"""Search-quality evaluation runner.

Embeds the WANDS queries, runs them through the Lakebase serving functions
(semantic + hybrid) and scores the rankings against the WANDS judgments. Runs
two query sets:

- **clean**: the original WANDS queries (measures overall relevance; the place
  where name-weighting via ``setweight``/``ts_rank_cd`` should show up), and
- **typo**: the same queries with seeded keyboard typos (measures robustness;
  the place where a ``pg_trgm`` fuzzy fallback should show up).

Writes a JSON + Markdown report under ``eval/results/`` and prints a comparison
table. Re-run after a search change and diff the numbers.

Example
-------
    python -m eval.run_eval \\
        --profile azure-video \\
        --instance projects/<id>/branches/<id>/endpoints/<id> \\
        --database appdb \\
        --tag baseline
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from . import metrics
from .data import Query, ensure_data, load_qrels, load_queries
from .search import SearchClient
from .typo import typo_queries

MODES = ("semantic", "hybrid")
_RESULTS_DIR = Path(__file__).parent / "results"


# --------------------------------------------------------------------------- #
# metric aggregation
# --------------------------------------------------------------------------- #
def aggregate(runs: dict[int, list[int]], qrels: dict[int, metrics.Qrels]) -> dict[str, float | None]:
    """Mean metrics over queries. ``runs`` maps query_id -> ranked product_ids."""
    acc: dict[str, list[float | None]] = defaultdict(list)
    for qid, ranked in runs.items():
        qr = qrels.get(qid, {})
        acc["ndcg@10"].append(metrics.ndcg_at_k(ranked, qr, 10))
        acc["ndcg@20"].append(metrics.ndcg_at_k(ranked, qr, 20))
        acc["recall@10_strict"].append(metrics.recall_at_k(ranked, qr, 10, strict=True))
        acc["recall@20_strict"].append(metrics.recall_at_k(ranked, qr, 20, strict=True))
        acc["recall@10_lenient"].append(metrics.recall_at_k(ranked, qr, 10, strict=False))
        acc["recall@20_lenient"].append(metrics.recall_at_k(ranked, qr, 20, strict=False))
        acc["precision@10_lenient"].append(metrics.precision_at_k(ranked, qr, 10, strict=False))
        acc["mrr_strict"].append(metrics.mrr(ranked, qr, strict=True))
        acc["map@20_lenient"].append(metrics.average_precision_at_k(ranked, qr, 20, strict=False))
    return {name: metrics.mean(vals) for name, vals in acc.items()}


def aggregate_by_class(
    runs: dict[int, list[int]],
    qrels: dict[int, metrics.Qrels],
    qclass: dict[int, str],
) -> dict[str, dict[str, float | None]]:
    """NDCG@10 / recall@20_lenient per query_class (compact per-facet view)."""
    by_class: dict[str, dict[int, list[int]]] = defaultdict(dict)
    for qid, ranked in runs.items():
        by_class[qclass.get(qid, "?")][qid] = ranked
    out: dict[str, dict[str, float | None]] = {}
    for cls, cls_runs in sorted(by_class.items()):
        agg = aggregate(cls_runs, qrels)
        out[cls] = {
            "n": len(cls_runs),
            "ndcg@10": agg["ndcg@10"],
            "recall@20_lenient": agg["recall@20_lenient"],
        }
    return out


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def _fmt(v: float | None) -> str:
    return "  -  " if v is None else f"{v:.4f}"


def render_table(set_name: str, per_mode: dict[str, dict[str, float | None]]) -> str:
    metric_names = list(next(iter(per_mode.values())).keys())
    width = max(len(m) for m in metric_names)
    header = f"{'metric'.ljust(width)} | " + " | ".join(f"{m:>9}" for m in per_mode)
    lines = [f"### {set_name}", "", header, "-" * len(header)]
    for metric_name in metric_names:
        row = f"{metric_name.ljust(width)} | " + " | ".join(
            f"{_fmt(per_mode[mode][metric_name]):>9}" for mode in per_mode
        )
        lines.append(row)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile", required=True, help="Databricks CLI profile")
    p.add_argument("--instance", required=True, help="full Lakebase endpoint resource name")
    p.add_argument("--database", default="appdb")
    p.add_argument("--embedding-endpoint", default="databricks-bge-large-en")
    p.add_argument("--limit", type=int, default=20, help="top-k retrieved per query")
    p.add_argument("--ef-search", type=int, default=20, help="HNSW ef_search (match app)")
    p.add_argument("--sample", type=int, default=0, help="evaluate only first N queries (0 = all)")
    p.add_argument("--typo-rate", type=float, default=0.15, help="per-word typo probability")
    p.add_argument("--seed", type=int, default=1234, help="typo RNG seed (reproducible)")
    p.add_argument("--skip-typo", action="store_true", help="only run the clean query set")
    p.add_argument("--tag", default="baseline", help="label for the output filenames")
    p.add_argument("--out", type=Path, default=_RESULTS_DIR)
    args = p.parse_args()

    ensure_data()
    queries: list[Query] = load_queries()
    qrels = load_qrels()
    if args.sample > 0:
        queries = queries[: args.sample]
    qclass = {q.query_id: q.query_class for q in queries}

    # Build query sets. Typo set keeps the ORIGINAL query_id (same info need).
    clean_texts = [q.text for q in queries]
    query_sets: dict[str, list[str]] = {"clean": clean_texts}
    if not args.skip_typo:
        query_sets["typo"] = typo_queries(clean_texts, rate=args.typo_rate, seed=args.seed)

    print(f">> connecting to Lakebase ({args.instance}) via profile {args.profile}")
    client = SearchClient.connect(
        profile=args.profile,
        instance=args.instance,
        database=args.database,
        embedding_endpoint=args.embedding_endpoint,
        ef_search=args.ef_search,
    )

    t0 = time.perf_counter()
    report: dict = {
        "meta": {
            "tag": args.tag,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_queries": len(queries),
            "limit": args.limit,
            "ef_search": args.ef_search,
            "typo_rate": args.typo_rate,
            "seed": args.seed,
            "embedding_endpoint": args.embedding_endpoint,
            "instance": args.instance,
            "database": args.database,
        },
        "sets": {},
    }
    rendered: list[str] = []

    try:
        for set_name, texts in query_sets.items():
            print(f">> [{set_name}] embedding {len(texts)} queries ...")
            vecs = client.embed(texts)

            per_mode: dict[str, dict[str, float | None]] = {}
            per_mode_by_class: dict[str, dict] = {}
            for mode in MODES:
                print(f">> [{set_name}] searching mode={mode} ...")
                runs: dict[int, list[int]] = {}
                for q, text, vec in zip(queries, texts, vecs):
                    runs[q.query_id] = client.run(mode, text, vec, args.limit)
                per_mode[mode] = aggregate(runs, qrels)
                per_mode_by_class[mode] = aggregate_by_class(runs, qrels, qclass)

            report["sets"][set_name] = {"overall": per_mode, "by_class": per_mode_by_class}
            rendered.append(render_table(set_name, per_mode))
    finally:
        client.close()

    elapsed = time.perf_counter() - t0
    report["meta"]["elapsed_s"] = round(elapsed, 1)

    # Write artifacts.
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    json_path = args.out / f"{args.tag}_{stamp}.json"
    md_path = args.out / f"{args.tag}_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2))

    md = [f"# Search-quality eval — `{args.tag}`", ""]
    md.append(f"- queries: {len(queries)}  |  limit: {args.limit}  |  ef_search: {args.ef_search}")
    md.append(f"- typo_rate: {args.typo_rate}  |  seed: {args.seed}  |  elapsed: {elapsed:.1f}s")
    md.append(f"- embedding: `{args.embedding_endpoint}`")
    md.append("")
    md.extend(f"```\n{block}\n```\n" for block in rendered)
    md_path.write_text("\n".join(md))

    print("\n" + "\n\n".join(rendered))
    print(f"\n>> wrote {json_path}")
    print(f">> wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
