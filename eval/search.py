"""Search client for the eval harness: embed queries + call Lakebase functions.

Talks to the *same* serving functions the app uses (``search_products_semantic``
/ ``search_products_hybrid``) but **directly over psycopg**, bypassing the
FastAPI app, its caches and the Apps edge — so we measure the ranking the SQL
functions produce, with nothing in between.

Connection follows the pattern in ``scripts/run_lakebase_sql.py``: resolve the
endpoint DNS, mint a fresh OAuth token, and connect as the current user (the
apply-time superuser). Embeddings use the same Model Serving endpoint and model
(BGE-large) that embedded the catalog in ``notebooks/03_embed_catalog.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

import psycopg
from databricks.sdk import WorkspaceClient
from pgvector.psycopg import register_vector

_EMBED_BATCH = 128  # Model Serving accepts batches; keep requests modest.


@dataclass
class SearchClient:
    ws: WorkspaceClient
    conn: psycopg.Connection
    embedding_endpoint: str

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    @classmethod
    def connect(
        cls,
        *,
        profile: str,
        instance: str,
        database: str,
        embedding_endpoint: str = "databricks-bge-large-en",
        ef_search: int = 20,
    ) -> "SearchClient":
        """Open a Lakebase connection using a Databricks CLI ``profile``.

        ``instance`` is the full endpoint resource name, e.g.
        ``projects/<id>/branches/<id>/endpoints/<id>``.
        """
        import os

        os.environ["DATABRICKS_CONFIG_PROFILE"] = profile
        ws = WorkspaceClient()

        endpoint = ws.api_client.do("GET", f"/api/2.0/postgres/{instance}")
        dns = endpoint["status"]["hosts"]["host"]
        cred = ws.api_client.do(
            "POST", "/api/2.0/postgres/credentials", body={"endpoint": instance}
        )
        user = ws.current_user.me().user_name

        conn = psycopg.connect(
            host=dns,
            port=5432,
            dbname=database,
            user=user,
            password=cred["token"],
            sslmode="require",
            autocommit=True,
        )
        register_vector(conn)
        # Match the app's HNSW session tuning (lakebase.py:66) so recall is
        # measured under the same conditions the app serves.
        conn.execute(f"SET hnsw.ef_search = {int(ef_search)}")
        return cls(ws=ws, conn=conn, embedding_endpoint=embedding_endpoint)

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------ #
    # embedding
    # ------------------------------------------------------------------ #
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed queries in batches (one Model Serving call per batch)."""
        out: list[list[float]] = []
        for start in range(0, len(texts), _EMBED_BATCH):
            chunk = texts[start : start + _EMBED_BATCH]
            resp = self.ws.serving_endpoints.query(
                name=self.embedding_endpoint, input=chunk
            )
            for elt in resp.data:
                emb = elt.embedding if hasattr(elt, "embedding") else elt["embedding"]
                # Coerce to float: Model Serving occasionally returns an int
                # element (e.g. 0), and pgvector refuses mixed-type lists.
                out.append([float(x) for x in emb])
        return out

    # ------------------------------------------------------------------ #
    # search (returns ranked product_ids, best first)
    # ------------------------------------------------------------------ #
    def semantic(self, qvec: list[float], limit: int, product_class: str | None = None) -> list[int]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT product_id FROM search_products_semantic(%s::vector(1024), %s, %s)",
                [qvec, product_class, limit],
            )
            return [row[0] for row in cur.fetchall()]

    def hybrid(
        self,
        query_text: str,
        qvec: list[float],
        limit: int,
        product_class: str | None = None,
    ) -> list[int]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT product_id FROM search_products_hybrid(%s, %s::vector(1024), %s, %s)",
                [query_text, qvec, product_class, limit],
            )
            return [row[0] for row in cur.fetchall()]

    def run(self, mode: str, query_text: str, qvec: list[float], limit: int) -> list[int]:
        """Dispatch by mode, mirroring app/backend/main.py:_run_search."""
        if mode == "hybrid":
            return self.hybrid(query_text, qvec, limit)
        return self.semantic(qvec, limit)
