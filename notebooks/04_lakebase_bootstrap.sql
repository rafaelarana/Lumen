-- ============================================================================
-- 04 — Lakebase bootstrap
-- ============================================================================
-- Idempotent: applies extensions, the materialized view that types embeddings
-- correctly, indexes, and serving functions. The Synced Table replicates the
-- source Delta into lumen_gold.products_synced with embedding stored as jsonb
-- (Synced Tables don't know about pgvector); we materialize a casted view on
-- top for HNSW indexing.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 1. Extensions
-- ----------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS databricks_auth;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- trigram fuzzy match (typo fallback)

-- ----------------------------------------------------------------------------
-- 2. Materialized view with proper vector type + tsvector for FTS
-- ----------------------------------------------------------------------------
DROP MATERIALIZED VIEW IF EXISTS lumen_gold.products_mv CASCADE;

CREATE MATERIALIZED VIEW lumen_gold.products_mv AS
SELECT
    product_id,
    product_name,
    product_class,
    category_hierarchy,
    product_description,
    product_features,
    average_rating,
    review_count,
    embedding::text::vector(1024) AS embedding,
    to_tsvector(
        'english',
        coalesce(product_name, '')         || ' ' ||
        coalesce(product_description, '')  || ' ' ||
        coalesce(product_class, '')
    ) AS search_vector
FROM lumen_gold.products_synced
WHERE embedding IS NOT NULL;

-- Unique index on PK enables REFRESH MATERIALIZED VIEW CONCURRENTLY
CREATE UNIQUE INDEX idx_products_mv_pk
    ON lumen_gold.products_mv (product_id);

-- HNSW for vector cosine similarity (BGE-large is 1024-dim)
CREATE INDEX idx_products_mv_embedding
    ON lumen_gold.products_mv
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);

-- Pre-filter by product_class (the WANDS analog of "distributor")
CREATE INDEX idx_products_mv_class
    ON lumen_gold.products_mv (product_class);

-- Full-text search GIN index
CREATE INDEX idx_products_mv_fts
    ON lumen_gold.products_mv USING gin (search_vector);

-- Trigram index on the product name — powers the pg_trgm fuzzy fallback in
-- search_products_hybrid (the `<%` word-similarity operator) for typo'd queries.
CREATE INDEX idx_products_mv_name_trgm
    ON lumen_gold.products_mv USING gin (product_name gin_trgm_ops);

-- ----------------------------------------------------------------------------
-- 3. Serving functions
-- ----------------------------------------------------------------------------

-- 3a. Pure semantic search ----------------------------------------------------
CREATE OR REPLACE FUNCTION search_products_semantic(
    query_embedding vector(1024),
    p_class TEXT DEFAULT NULL,
    p_limit INT DEFAULT 20
) RETURNS TABLE (
    product_id INT,
    product_name TEXT,
    product_class TEXT,
    category_hierarchy TEXT,
    average_rating DOUBLE PRECISION,
    review_count INT,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        p.product_id,
        p.product_name,
        p.product_class,
        p.category_hierarchy,
        p.average_rating,
        p.review_count,
        (1 - (p.embedding <=> query_embedding))::float AS similarity
    FROM lumen_gold.products_mv p
    WHERE (p_class IS NULL OR p.product_class = p_class)
    ORDER BY p.embedding <=> query_embedding
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE;

-- 3b. Hybrid search (vector + FTS, RRF combine) -------------------------------
-- The text branch is field-weighted — the product NAME is boosted over class
-- and description via setweight + ts_rank_cd — and falls back to a pg_trgm
-- fuzzy match on the name when strict FTS finds too few hits (typo tolerance).
-- The weighted tsvector is computed INLINE (over the small candidate set the
-- GIN-indexed @@ filter selects), so this needs no materialized-view rebuild;
-- it could be materialized into products_mv later for efficiency.
-- Signature is unchanged (6 args) so the GRANT and app call site still match;
-- the fallback knobs are kept as local constants.
CREATE OR REPLACE FUNCTION search_products_hybrid(
    query_text TEXT,
    query_embedding vector(1024),
    p_class TEXT DEFAULT NULL,
    p_limit INT DEFAULT 20,
    p_vector_weight FLOAT DEFAULT 0.7,
    p_text_weight FLOAT DEFAULT 0.3
) RETURNS TABLE (
    product_id INT,
    product_name TEXT,
    product_class TEXT,
    category_hierarchy TEXT,
    average_rating DOUBLE PRECISION,
    review_count INT,
    combined_score FLOAT
) AS $$
DECLARE
    c_min_fts_hits CONSTANT INT   := 3;    -- below this, use the fuzzy fallback
    c_trgm_thresh  CONSTANT FLOAT := 0.3;  -- pg_trgm word_similarity threshold
    -- ts_rank_cd weights are ordered {D, C, B, A}: name(A)=1.0 dominates,
    -- class(B)=0.6, description(C)=0.3.
    c_weights      CONSTANT float4[] := '{0.1, 0.3, 0.6, 1.0}'::float4[];
    v_query        tsquery := plainto_tsquery('english', query_text);
    v_fts_hits     INT;
BEGIN
    SELECT count(*) INTO v_fts_hits
    FROM lumen_gold.products_mv p
    WHERE p.search_vector @@ v_query
      AND (p_class IS NULL OR p.product_class = p_class);

    IF v_fts_hits >= c_min_fts_hits THEN
        -- Strict FTS branch, name-weighted ranking.
        RETURN QUERY
        WITH vector_results AS (
            SELECT p.product_id,
                   ROW_NUMBER() OVER (ORDER BY p.embedding <=> query_embedding) AS rk
            FROM lumen_gold.products_mv p
            WHERE (p_class IS NULL OR p.product_class = p_class)
            ORDER BY p.embedding <=> query_embedding
            LIMIT p_limit * 3
        ),
        text_results AS (
            SELECT p.product_id,
                   ROW_NUMBER() OVER (
                       ORDER BY ts_rank_cd(
                           c_weights,
                           setweight(to_tsvector('english', coalesce(p.product_name, '')),        'A') ||
                           setweight(to_tsvector('english', coalesce(p.product_class, '')),       'B') ||
                           setweight(to_tsvector('english', coalesce(p.product_description, '')), 'C'),
                           v_query
                       ) DESC
                   ) AS rk
            FROM lumen_gold.products_mv p
            WHERE p.search_vector @@ v_query
              AND (p_class IS NULL OR p.product_class = p_class)
            LIMIT p_limit * 3
        ),
        combined AS (
            SELECT COALESCE(v.product_id, t.product_id) AS product_id,
                   (p_vector_weight * (1.0 / (60 + COALESCE(v.rk, 1000)))) +
                   (p_text_weight   * (1.0 / (60 + COALESCE(t.rk, 1000)))) AS rrf_score
            FROM vector_results v
            FULL OUTER JOIN text_results t USING (product_id)
        )
        SELECT p.product_id, p.product_name, p.product_class, p.category_hierarchy,
               p.average_rating, p.review_count, c.rrf_score::float AS combined_score
        FROM combined c
        JOIN lumen_gold.products_mv p USING (product_id)
        ORDER BY c.rrf_score DESC
        LIMIT p_limit;
    ELSE
        -- Fuzzy fallback: pg_trgm word-similarity on the product name (typos).
        -- Transaction-local GUC drives the `<%` operator + the trigram index.
        PERFORM set_config('pg_trgm.word_similarity_threshold', c_trgm_thresh::text, true);
        RETURN QUERY
        WITH vector_results AS (
            SELECT p.product_id,
                   ROW_NUMBER() OVER (ORDER BY p.embedding <=> query_embedding) AS rk
            FROM lumen_gold.products_mv p
            WHERE (p_class IS NULL OR p.product_class = p_class)
            ORDER BY p.embedding <=> query_embedding
            LIMIT p_limit * 3
        ),
        text_results AS (
            SELECT p.product_id,
                   ROW_NUMBER() OVER (
                       ORDER BY word_similarity(query_text, p.product_name) DESC
                   ) AS rk
            FROM lumen_gold.products_mv p
            WHERE query_text <% p.product_name
              AND (p_class IS NULL OR p.product_class = p_class)
            ORDER BY word_similarity(query_text, p.product_name) DESC
            LIMIT p_limit * 3
        ),
        combined AS (
            SELECT COALESCE(v.product_id, t.product_id) AS product_id,
                   (p_vector_weight * (1.0 / (60 + COALESCE(v.rk, 1000)))) +
                   (p_text_weight   * (1.0 / (60 + COALESCE(t.rk, 1000)))) AS rrf_score
            FROM vector_results v
            FULL OUTER JOIN text_results t USING (product_id)
        )
        SELECT p.product_id, p.product_name, p.product_class, p.category_hierarchy,
               p.average_rating, p.review_count, c.rrf_score::float AS combined_score
        FROM combined c
        JOIN lumen_gold.products_mv p USING (product_id)
        ORDER BY c.rrf_score DESC
        LIMIT p_limit;
    END IF;
END;
$$ LANGUAGE plpgsql VOLATILE;

-- 3c. Similar-product recommender --------------------------------------------
CREATE OR REPLACE FUNCTION recommend_similar_products(
    p_product_id INT,
    p_limit INT DEFAULT 10,
    p_same_class BOOLEAN DEFAULT FALSE
) RETURNS TABLE (
    product_id INT,
    product_name TEXT,
    product_class TEXT,
    average_rating DOUBLE PRECISION,
    review_count INT,
    similarity FLOAT
) AS $$
DECLARE
    src_embedding vector(1024);
    src_class TEXT;
BEGIN
    SELECT p.embedding, p.product_class
      INTO src_embedding, src_class
    FROM lumen_gold.products_mv p
    WHERE p.product_id = p_product_id;

    IF src_embedding IS NULL THEN
        RETURN;
    END IF;

    RETURN QUERY
    SELECT
        p.product_id,
        p.product_name,
        p.product_class,
        p.average_rating,
        p.review_count,
        (1 - (p.embedding <=> src_embedding))::float AS similarity
    FROM lumen_gold.products_mv p
    WHERE p.product_id <> p_product_id
      AND (NOT p_same_class OR p.product_class = src_class)
    ORDER BY p.embedding <=> src_embedding
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE;

-- 3d. Distinct classes (for the UI filter) -----------------------------------
CREATE OR REPLACE FUNCTION list_product_classes(p_limit INT DEFAULT 50)
RETURNS TABLE (product_class TEXT, n BIGINT) AS $$
BEGIN
    RETURN QUERY
    SELECT p.product_class, COUNT(*)::BIGINT AS n
    FROM lumen_gold.products_mv p
    WHERE p.product_class IS NOT NULL AND p.product_class <> ''
    GROUP BY p.product_class
    ORDER BY n DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE;

-- 3e. Single-product detail lookup -------------------------------------------
CREATE OR REPLACE FUNCTION get_product(p_product_id INT)
RETURNS TABLE (
    product_id INT,
    product_name TEXT,
    product_class TEXT,
    category_hierarchy TEXT,
    product_description TEXT,
    product_features TEXT,
    average_rating DOUBLE PRECISION,
    review_count INT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        p.product_id, p.product_name, p.product_class, p.category_hierarchy,
        p.product_description, p.product_features,
        p.average_rating, p.review_count
    FROM lumen_gold.products_mv p
    WHERE p.product_id = p_product_id;
END;
$$ LANGUAGE plpgsql STABLE;

-- ============================================================================
-- 4. TURBO MODE — precomputed neighbors for "similar products"
-- ============================================================================
-- Materialize the top-20 nearest neighbors for every product once. /similar
-- becomes a PK lookup + array unnest instead of an HNSW scan + embedding
-- fetch. ~30s to build for 43K rows.
--
-- The /search "Turbo" path is implemented entirely in the app layer (LRU
-- cache around the embedding call) and reuses the existing search functions,
-- so no SQL change is needed for it.
-- ============================================================================

DROP MATERIALIZED VIEW IF EXISTS lumen_gold.similar_top_k CASCADE;

CREATE MATERIALIZED VIEW lumen_gold.similar_top_k AS
SELECT
    p.product_id,
    ARRAY(
        SELECT q.product_id
        FROM lumen_gold.products_mv q
        WHERE q.product_id <> p.product_id
        ORDER BY q.embedding <=> p.embedding
        LIMIT 20
    ) AS neighbors
FROM lumen_gold.products_mv p;

CREATE UNIQUE INDEX idx_similar_top_k_pk
    ON lumen_gold.similar_top_k (product_id);

-- Fast recommender: rank from precomputed array, no HNSW lookup.
CREATE OR REPLACE FUNCTION recommend_similar_products_fast(
    p_product_id INT,
    p_limit INT DEFAULT 10
) RETURNS TABLE (
    product_id INT,
    product_name TEXT,
    product_class TEXT,
    average_rating DOUBLE PRECISION,
    review_count INT,
    rank INT
) AS $$
BEGIN
    RETURN QUERY
    WITH ranked AS (
        SELECT n.neighbor_id::INT AS pid, n.ord::INT AS rk
        FROM lumen_gold.similar_top_k stk
        CROSS JOIN LATERAL unnest(stk.neighbors[1:p_limit])
            WITH ORDINALITY AS n(neighbor_id, ord)
        WHERE stk.product_id = p_product_id
    )
    SELECT
        mv.product_id,
        mv.product_name,
        mv.product_class,
        mv.average_rating,
        mv.review_count,
        r.rk
    FROM ranked r
    JOIN lumen_gold.products_mv mv ON mv.product_id = r.pid
    ORDER BY r.rk;
END;
$$ LANGUAGE plpgsql STABLE;
