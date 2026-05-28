#!/usr/bin/env python3
"""Apply the FTS search tuning to a Lakebase branch — idempotently, no downtime.

Applies ONLY the additive pieces of the search tuning so there is no
materialized-view rebuild and no HNSW reindex:

  1. CREATE EXTENSION IF NOT EXISTS pg_trgm
  2. CREATE INDEX IF NOT EXISTS idx_products_mv_name_trgm (trigram on name)
  3. CREATE OR REPLACE FUNCTION search_products_hybrid(...)  -- name-weighted
     ts_rank_cd + pg_trgm fuzzy fallback

The function body is extracted verbatim from
``notebooks/04_lakebase_bootstrap.sql`` so that file stays the single source of
truth. Connection/auth mirrors ``scripts/run_lakebase_sql.py``.

Usage:
    python3 scripts/apply_search_tuning.py \\
        --profile azure-video \\
        --instance projects/<id>/branches/<id>/endpoints/<id> \\
        --database appdb
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

try:
    import psycopg
    from databricks.sdk import WorkspaceClient
except ImportError:  # pragma: no cover
    print("Missing deps: pip install 'databricks-sdk' 'psycopg[binary]'", file=sys.stderr)
    sys.exit(2)

_BOOTSTRAP = Path(__file__).resolve().parent.parent / "notebooks" / "04_lakebase_bootstrap.sql"

_EXTENSION = "CREATE EXTENSION IF NOT EXISTS pg_trgm;"
_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_products_mv_name_trgm "
    "ON lumen_gold.products_mv USING gin (product_name gin_trgm_ops);"
)


def extract_hybrid_function(sql_text: str) -> str:
    """Slice the search_products_hybrid definition out of the bootstrap SQL."""
    start = sql_text.index("CREATE OR REPLACE FUNCTION search_products_hybrid(")
    end_marker = "$$ LANGUAGE plpgsql VOLATILE;"
    end = sql_text.index(end_marker, start) + len(end_marker)
    return sql_text[start:end]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", required=True)
    p.add_argument("--instance", required=True, help="full endpoint resource name")
    p.add_argument("--database", required=True)
    args = p.parse_args()

    func_sql = extract_hybrid_function(_BOOTSTRAP.read_text())
    # Sanity: must be the tuned version, not the old ts_rank one.
    if "ts_rank_cd" not in func_sql or "word_similarity" not in func_sql:
        print("ERROR: extracted function is not the tuned version.", file=sys.stderr)
        return 1

    os.environ["DATABRICKS_CONFIG_PROFILE"] = args.profile
    ws = WorkspaceClient()
    endpoint = ws.api_client.do("GET", f"/api/2.0/postgres/{args.instance}")
    dns = endpoint["status"]["hosts"]["host"]
    cred = ws.api_client.do("POST", "/api/2.0/postgres/credentials", body={"endpoint": args.instance})
    user = ws.current_user.me().user_name
    print(f">> Connecting to {args.database} on {dns} as {user}")

    conn = psycopg.connect(
        host=dns, port=5432, dbname=args.database, user=user,
        password=cred["token"], sslmode="require", autocommit=True,
    )
    with conn.cursor() as cur:
        print(">> 1/3 pg_trgm extension")
        cur.execute(_EXTENSION)
        print(">> 2/3 trigram index on product_name (idempotent)")
        cur.execute(_INDEX)
        print(f">> 3/3 search_products_hybrid ({len(func_sql):,} bytes)")
        cur.execute(func_sql)
    conn.close()
    print(">> Search tuning applied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
