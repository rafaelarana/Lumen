"""Load the WANDS ground-truth queries and relevance judgments.

WANDS (Wayfair ANnotation Dataset for Search, MIT license) ships three
tab-separated files (despite the ``.csv`` extension). We only need two of them
for evaluation — the products themselves already live in Lakebase:

- ``query.csv``  -> ``query_id``, ``query``, ``query_class``   (480 rows)
- ``label.csv``  -> ``id``, ``query_id``, ``product_id``, ``label``
                    where ``label`` ∈ {Exact, Partial, Irrelevant}  (233,448 rows)

The same files are loaded into ``lumen_bronze.{queries,labels}`` by
``notebooks/01_load_wands.py``; reading them straight from GitHub keeps the
harness independent of a running warehouse. ``product_id`` matches Lakebase
because the pipeline loads ``product.csv`` with the PK unchanged.
"""
from __future__ import annotations

import csv
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# WANDS gains follow the same convention as eval.metrics.
LABEL_TO_GAIN = {"Exact": 2, "Partial": 1, "Irrelevant": 0}

_RAW_BASE = "https://raw.githubusercontent.com/wayfair/WANDS/main/dataset"
_DATA_DIR = Path(__file__).parent / "data"
_FILES = ("query.csv", "label.csv")

Qrels = dict[int, int]  # {product_id: gain}


@dataclass(frozen=True)
class Query:
    query_id: int
    text: str
    query_class: str


def ensure_data(data_dir: Path = _DATA_DIR) -> Path:
    """Download and cache the WANDS query/label files. Idempotent."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in _FILES:
        dest = data_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        url = f"{_RAW_BASE}/{name}"
        print(f">> downloading {url}")
        urllib.request.urlretrieve(url, dest)  # noqa: S310 (trusted host)
    return data_dir


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def load_queries(data_dir: Path = _DATA_DIR) -> list[Query]:
    """All WANDS queries, ordered by query_id."""
    rows = _read_tsv(data_dir / "query.csv")
    out = [
        Query(
            query_id=int(r["query_id"]),
            text=r["query"].strip(),
            query_class=(r.get("query_class") or "").strip(),
        )
        for r in rows
    ]
    return sorted(out, key=lambda q: q.query_id)


def load_qrels(data_dir: Path = _DATA_DIR) -> dict[int, Qrels]:
    """Relevance judgments as ``{query_id: {product_id: gain}}``.

    Only judgments with a known label are kept; unknown/blank labels are
    skipped (they would map to gain 0 anyway, same as an unjudged product).
    """
    rows = _read_tsv(data_dir / "label.csv")
    qrels: dict[int, Qrels] = {}
    for r in rows:
        gain = LABEL_TO_GAIN.get((r.get("label") or "").strip())
        if gain is None:
            continue
        qid = int(r["query_id"])
        pid = int(r["product_id"])
        qrels.setdefault(qid, {})[pid] = gain
    return qrels


if __name__ == "__main__":
    ensure_data()
    qs = load_queries()
    qrels = load_qrels()
    n_judg = sum(len(v) for v in qrels.values())
    print(f"queries: {len(qs)}  | queries with qrels: {len(qrels)}  | judgments: {n_judg:,}")
    print(f"example: {qs[0]}")
    print(f"  judged products for q{qs[0].query_id}: {len(qrels.get(qs[0].query_id, {}))}")
