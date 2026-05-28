# Search-quality evaluation harness

Measures the **ranking quality** of the Lakebase serving functions
(`search_products_semantic` / `search_products_hybrid`) against the **WANDS**
ground-truth relevance judgments. Use it to set a baseline and to prove that a
search change (e.g. name-weighting or typo tolerance) actually improves
relevance instead of just "feeling" better.

## What it measures

Graded relevance from WANDS labels (`Exact`=2, `Partial`=1, `Irrelevant`/unjudged=0):

| Metric | Meaning |
|---|---|
| **NDCG@10 / @20** | primary; rewards putting the most-relevant items highest |
| **Recall@10 / @20** | fraction of relevant items found (strict = Exact only; lenient = Exact+Partial) |
| **Precision@10** | fraction of the top 10 that is relevant (lenient) |
| **MRR** | reciprocal rank of the first Exact hit |
| **MAP@20** | mean average precision (lenient) |

Reported per **mode** (semantic vs hybrid) and per **query_class** facet.

Two query sets are evaluated:

- **clean** — the 480 original WANDS queries. Where `setweight` / `ts_rank_cd`
  name-weighting should show up.
- **typo** — the same queries with seeded keyboard typos. Where a `pg_trgm`
  fuzzy fallback should show up. Deterministic given `--seed`.

The harness calls the SQL functions **directly** (psycopg), bypassing the
FastAPI app and its caches, so we measure pure ranking — not latency or caching.

## Design notes

- **Same embedding model** as the catalog (`databricks-bge-large-en`, see
  `notebooks/03_embed_catalog.py`) → fair comparison.
- **Same `hnsw.ef_search`** as the app (20, see `app/backend/lakebase.py`) →
  recall measured under serving conditions. Override with `--ef-search`.
- WANDS files are downloaded once and cached under `eval/data/` (gitignored).
- Results are written to `eval/results/<tag>_<timestamp>.{json,md}` (gitignored).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r eval/requirements.txt
```

You need a Databricks CLI profile with access to the workspace, the Model
Serving endpoint, and the Lakebase endpoint (the apply-time superuser identity,
same one used by `scripts/run_lakebase_sql.py`).

## Run

```bash
# Quick smoke test (10 queries, clean set only) — verifies auth + connectivity.
python -m eval.run_eval \
  --profile <cli-profile> \
  --instance projects/<id>/branches/<id>/endpoints/<id> \
  --database appdb \
  --sample 10 --skip-typo --tag smoke

# Full baseline (480 queries, clean + typo).
python -m eval.run_eval \
  --profile <cli-profile> \
  --instance projects/<id>/branches/<id>/endpoints/<id> \
  --database appdb \
  --tag baseline
```

Useful flags: `--limit` (top-k, default 20), `--typo-rate` (default 0.15),
`--seed` (default 1234), `--ef-search` (default 20), `--sample N`, `--skip-typo`.

## Unit-test the metrics

```bash
python -m eval.metrics    # self-test with known rankings
python -m eval.data       # download + summarize WANDS (480 queries)
python -m eval.typo       # preview typo injection
```

## Workflow: measure → change → re-measure

1. Run `--tag baseline` and commit the numbers.
2. Apply a search change (e.g. edit `notebooks/04_lakebase_bootstrap.sql`,
   re-run the bootstrap, refresh the materialized view).
3. Run `--tag <change-name>` and diff against baseline. Expect hybrid NDCG to
   rise on **clean** (name boost) and on **typo** (fuzzy fallback), with no
   regression on clean.
