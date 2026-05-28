# Lumen — Search

How product search works in Lumen, end to end: the two retrieval signals
(semantic + full-text), how they are fused (hybrid / RRF), the serving
functions and caches that back them, and the WANDS-based evaluation harness used
to measure ranking quality.

This document is the deep-dive companion to
[`Architecture.md`](./Architecture.md); see §4.6 (serving functions), §5.4
(caches) and §5.5 (API surface) there for the surrounding system.

## Table of contents

- [1. Overview](#1-overview)
- [2. Semantic search (vector)](#2-semantic-search-vector)
- [3. Full-text search (keyword)](#3-full-text-search-keyword)
- [4. Hybrid search (RRF fusion)](#4-hybrid-search-rrf-fusion)
- [5. Serving functions](#5-serving-functions)
- [6. Turbo path — layered caches](#6-turbo-path--layered-caches)
- [7. API surface](#7-api-surface)
- [8. Quality evaluation (WANDS)](#8-quality-evaluation-wands)
- [9. Status & remaining limitations](#9-status--remaining-limitations)

---

## 1. Overview

Lumen searches the **WANDS** catalog (~42,994 products, Wayfair ANnotation
Dataset for Search) stored in Lakebase Autoscale (managed Postgres). Two
retrieval signals are available, selectable per request via `mode`:

| Mode | Signal | Strength |
|---|---|---|
| `semantic` | vector cosine similarity over BGE-large embeddings (HNSW) | intent, synonyms, natural-language queries |
| `hybrid` | semantic **+** Postgres full-text search, fused with RRF | adds exact keyword / token precision |

Everything — data, vector index, full-text index and the fusion logic — lives in
**one Postgres** (`lumen_gold.products_mv`). There is no separate search engine
to operate or sync. The whole search is a SQL function call.

The source of truth for all of this is
[`notebooks/04_lakebase_bootstrap.sql`](../notebooks/04_lakebase_bootstrap.sql)
(materialized view, indexes, serving functions) and
[`app/backend/`](../app/backend) (embedding, pooling, caching, API).

---

## 2. Semantic search (vector)

**Embeddings.** Product text (`name | class | category_hierarchy | description |
features`) is embedded offline into a 1024-dim vector with
`databricks-bge-large-en` (`notebooks/03_embed_catalog.py`). Query text is
embedded at request time through the same Model Serving endpoint
(`app/backend/embed.py`).

**Index.** The materialized view carries the vector as a real `pgvector`
column, indexed with **HNSW** for approximate nearest-neighbor search
(`04_lakebase_bootstrap.sql:48-51`):

```sql
CREATE INDEX idx_products_mv_embedding
    ON lumen_gold.products_mv
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);
```

Per session the app sets `hnsw.ef_search = 20` (`app/backend/lakebase.py:66`) —
a lower value than the default 40, trading a little recall for lower, more
predictable latency.

**Ranking.** `search_products_semantic` orders by cosine distance (`<=>`) and
returns `similarity = 1 - distance` (`04_lakebase_bootstrap.sql:66-94`):

```sql
ORDER BY p.embedding <=> query_embedding
LIMIT p_limit;
```

---

## 3. Full-text search (keyword)

The keyword signal uses **PostgreSQL native full-text search** — no BM25 engine,
no external service. Three pieces:

**(a) Indexed document — `search_vector`.** Built once in the materialized view
(`04_lakebase_bootstrap.sql:34-39`):

```sql
to_tsvector('english',
    coalesce(product_name, '')        || ' ' ||
    coalesce(product_description, '') || ' ' ||
    coalesce(product_class, ''))      AS search_vector
```

`to_tsvector('english', …)` tokenizes, applies English stemming
(`running`→`run`) and drops stop-words. Because it lives in a materialized view,
it is computed at refresh time, not per query.

**(b) Index — GIN.** An inverted index (word → products) makes `@@` matching
fast (`04_lakebase_bootstrap.sql:58-59`):

```sql
CREATE INDEX idx_products_mv_fts
    ON lumen_gold.products_mv USING gin (search_vector);
```

**(c) Query processing & ranking.** Inside the hybrid function the user text is
normalized identically with `plainto_tsquery('english', …)` (tokenize + stem +
**AND** of terms), and matched with `@@` against the stored `search_vector`
(fast GIN membership test). Ranking is **field-weighted**: a weighted tsvector
is built inline (name=`A`, class=`B`, description=`C`) and scored with
`ts_rank_cd`, so a term in the product name outranks the same term buried in a
description:

```sql
WHERE p.search_vector @@ plainto_tsquery('english', query_text)
ORDER BY ts_rank_cd(
    '{0.1, 0.3, 0.6, 1.0}'::float4[],          -- weights {D, C, B, A}
    setweight(to_tsvector('english', coalesce(product_name, '')),        'A') ||
    setweight(to_tsvector('english', coalesce(product_class, '')),       'B') ||
    setweight(to_tsvector('english', coalesce(product_description, '')), 'C'),
    plainto_tsquery('english', query_text)
) DESC
```

> Note: Postgres FTS has no true IDF/BM25. `setweight` + `ts_rank_cd` approximate
> it with **field boosting** (name ≫ description) plus cover-density proximity —
> a real BM25 would need an extension. The weighted tsvector is computed inline
> rather than stored, so this needed no materialized-view rebuild. Strict `@@`
> matching is still exact on stemmed lexemes, so misspellings are handled by a
> separate `pg_trgm` fuzzy fallback — see §4.

---

## 4. Hybrid search (RRF fusion)

`search_products_hybrid` (`04_lakebase_bootstrap.sql`) runs the two
signals independently and fuses their **ranks** with **Reciprocal Rank Fusion
(RRF)**. Ranks (not raw scores) are fused because cosine distance and the text
score are not comparable.

1. **`vector_results`** — top `p_limit * 3` by cosine distance, numbered with
   `ROW_NUMBER()`.
2. **`text_results`** — top `p_limit * 3` by the name-weighted `ts_rank_cd`
   (§3c), numbered with `ROW_NUMBER()`.
3. **`combined`** — `FULL OUTER JOIN` on `product_id`, scored:

```sql
(p_vector_weight * (1.0 / (60 + COALESCE(v.rk, 1000)))) +
(p_text_weight   * (1.0 / (60 + COALESCE(t.rk, 1000)))) AS rrf_score
```

- `k = 60` — standard RRF damping constant.
- Missing rank → `1000` (penalty when a product appears in only one list).
- Defaults: `p_vector_weight = 0.7`, `p_text_weight = 0.3` — semantic leads,
  keyword refines. Both are function parameters.

Results are ordered by `rrf_score DESC` and the top `p_limit` returned as
`combined_score`.

**Typo fallback (`pg_trgm`).** Before fusing, the function counts strict-FTS
hits. When that count is below a threshold (3) — typically because the query is
misspelled and `@@` matches nothing — the `text_results` branch switches from
FTS to a **trigram word-similarity** match on the product name (`query <% name`,
backed by `idx_products_mv_name_trgm` and the `pg_trgm.word_similarity_threshold`
GUC). The vector branch and RRF fusion are unchanged, so a typo'd query still
returns a sensibly ranked list instead of relying on the vector signal alone.

---

## 5. Serving functions

All search is exposed as PL/pgSQL functions
(`04_lakebase_bootstrap.sql`, granted least-privilege to the App SP in
`scripts/run_lakebase_sql.py`):

| Function | Purpose |
|---|---|
| `search_products_semantic(vec, class, limit)` | vector-only search |
| `search_products_hybrid(text, vec, class, limit, vec_w, text_w)` | RRF hybrid search |
| `recommend_similar_products(product_id, limit, same_class)` | live "more like this" |
| `recommend_similar_products_fast(product_id, limit)` | precomputed neighbors (Turbo) |
| `list_product_classes(limit)` | facet values for the UI filter |
| `get_product(product_id)` | single-product lookup |

The backend dispatches by `mode` in `app/backend/main.py:140-167`
(`_run_search`).

---

## 6. Turbo path — layered caches

Two serving paths share the same ranking but differ on caching:

- **Standard** (`POST /api/search`) — always embeds the query (Model Serving)
  then calls the SQL function.
- **Turbo** (`POST /api/search/fast`) — three layers
  (`app/backend/result_cache.py`, `embed.py`, `main.py`):
  1. **L1 result cache** — `(product_id, score)` list keyed by
     `(normalized_query, mode, product_class)`.
  2. **L2 embed cache** — LRU of query→vector, skips re-embedding.
  3. **L3 full path** — Model Serving + HNSW/RRF for cold queries.

At startup `_preload_caches` (`main.py:63-85`) batch-embeds ~100 seed queries
(`loadgen.py`) and warms L1 for **both** modes. Caching affects latency only,
not ranking — so quality evaluation (§8) bypasses it.

---

## 7. API surface

| Endpoint | Notes |
|---|---|
| `POST /api/search` | Standard. Body: `q`, `mode` (`semantic`\|`hybrid`), `product_class?`, `limit`. |
| `POST /api/search/fast` | Turbo (layered cache). |
| `GET /api/classes` | Facet values. |
| `GET /api/cache/stats` | L1/L2 hit ratios. |
| `POST /api/benchmark/*` | In-app load generator (latency only). |

Response includes a latency breakdown (`embed_ms`, `db_ms`, `total_ms`,
`cache_hit`, `cache_layer`). Observed latency: see `Architecture.md §9`
(turbo:hybrid p50≈18 ms, p99≈50 ms).

---

## 8. Quality evaluation (WANDS)

WANDS ships graded relevance judgments, which Lumen uses to measure **ranking
quality** (not just latency). A standalone harness lives in
[`eval/`](../eval) (see [`eval/README.md`](../eval/README.md)).

**Ground truth.** `lumen_bronze.labels` — 480 queries × ~232K judgments,
`label ∈ {Exact, Partial, Irrelevant}` (gains 2 / 1 / 0). The harness reads the
WANDS `query.csv` + `label.csv` directly so it needs no warehouse;
`product_id` matches Lakebase.

**Metrics.** NDCG@10/@20 (primary, graded), Recall@10/@20 (strict = Exact,
lenient = Exact+Partial), Precision@10, MRR, MAP@20 — per mode and per
`query_class`.

**Method.** The harness calls the SQL functions **directly** (psycopg, same
`hnsw.ef_search` and embedding model as serving), bypassing the app and caches,
so it measures pure ranking. It evaluates two query sets:

- **clean** — the 480 WANDS queries (overall relevance).
- **typo** — the same queries with seeded keyboard typos (robustness;
  reproducible via `--seed`).

Run it:

```bash
pip install -r eval/requirements.txt
python -m eval.run_eval \
  --profile <cli-profile> \
  --instance projects/<id>/branches/<id>/endpoints/<id> \
  --database appdb --tag baseline
```

**Baseline vs tuned — 2026-05-28** (limit 20, `ef_search` 20, typo_rate 0.15,
seed 1234). "baseline" = unweighted `ts_rank`, no fallback; "tuned" = the §3c
name-weighting + §4 `pg_trgm` fallback now in production. The `semantic` column
is identical across runs (only the hybrid text branch changed) and is shown once
as a reference.

**Hybrid, clean set** (effect of name-weighting):

| Metric | semantic | hybrid baseline | hybrid tuned | Δ |
|---|---|---|---|---|
| NDCG@10 | 0.706 | 0.714 | **0.725** | +0.012 |
| precision@10 | 0.774 | 0.777 | **0.789** | +0.013 |
| MAP@20 | 0.695 | 0.697 | **0.706** | +0.009 |
| MRR | 0.708 | 0.726 | 0.707 | −0.019 |

**Hybrid, typo set** (effect of the `pg_trgm` fallback):

| Metric | semantic | hybrid baseline | hybrid tuned | Δ |
|---|---|---|---|---|
| NDCG@10 | 0.483 | 0.484 | **0.514** | +0.031 |
| MRR | 0.437 | 0.438 | **0.480** | +0.042 |
| precision@10 | 0.553 | 0.553 | **0.573** | +0.020 |
| MAP@20 | 0.471 | 0.471 | **0.489** | +0.018 |

Reading:

1. **Name-weighting (clean).** Hybrid's edge over semantic grew from +0.7 pt to
   +1.9 pt NDCG@10, with precision and MAP also up. Trade-off: MRR dipped ~0.019
   — the name boost improves the graded ranking overall but pushes the first
   *Exact* hit slightly later on some clean queries.
2. **Fuzzy fallback (typo).** The big win: baseline hybrid was no better than
   semantic under typos; tuned hybrid now clearly beats it (+3.1 pt NDCG@10,
   +4.2 pt MRR).
3. **Sanity.** The `semantic` column is unchanged baseline→tuned, confirming the
   change only touched the hybrid text branch.

Net positive; the change shipped. Re-run with a new `--tag` after future changes
and diff the JSON in `eval/results/`.

---

## 9. Status & remaining limitations

**Shipped (2026-05-28)** — the FTS tuning is implemented in
`notebooks/04_lakebase_bootstrap.sql` and applied to production via the
idempotent `scripts/apply_search_tuning.py` (adds the `pg_trgm` extension + a
trigram index and does `CREATE OR REPLACE` on the hybrid function — no MV
rebuild, no downtime). See the §8 delta.

| Was | Now |
|---|---|
| FTS treated name/class/description equally | name-weighted via `setweight` (A/B/C) + `ts_rank_cd` |
| No typo tolerance | `pg_trgm` trigram fuzzy **fallback** when strict FTS returns < 3 hits |

**Remaining limitations / future work:**

- **Still no true IDF/BM25** — the weighting is field boosting + cover density,
  not BM25. A real BM25 ranker would need an extension (e.g. ParadeDB
  `pg_search`).
- **Weighted tsvector computed inline** — fine at this candidate-set size; could
  be materialized into `products_mv` (a stored weighted column) if the text
  branch ever becomes a hotspot.
- **Fallback knobs are constants** — `min_fts_hits = 3` and
  `word_similarity_threshold = 0.3` are baked into the function; expose as
  parameters if tuning per query proves useful.
- **Eval requires a live endpoint** — the harness calls the real serving
  functions (read-only, runs in minutes).
