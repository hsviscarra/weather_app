"""
Lakebase (Databricks-managed Postgres) connection helper.

Connects using a single LAKEBASE_URL (a standard Postgres connection URL,
e.g. postgresql://role:password@host:5432/databricks_postgres?sslmode=require)
pointing at a native Postgres role with a static, non-expiring password.
Matches the connection pattern from the course's day-2 template
(databricks-lakebase-app-day-2/lakebase.py).
"""

import base64
import os
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import RealDictCursor

_SCOPE = os.environ.get("LAKEBASE_SECRET_SCOPE", "database")
_KEY = os.environ.get("LAKEBASE_SECRET_KEY", "lakebase-url")


def _lakebase_url() -> str:
    """Resolve the Lakebase connection URL: LAKEBASE_URL env var first
    (local dev only - never set on the deployed app), falling back to the
    Databricks secret scope via the SDK (matches app.yaml in production).

    WorkspaceClient() is constructed lazily, only when actually needed for
    the secret-fetch fallback - constructing it eagerly at import time would
    pay its auth-negotiation cost (which can hang for a long time against
    unreachable/invalid credentials) even on local runs that never use it.
    """
    env_url = os.environ.get("LAKEBASE_URL")
    if env_url:
        return env_url

    from databricks.sdk import WorkspaceClient

    secret = WorkspaceClient().secrets.get_secret(scope=_SCOPE, key=_KEY)
    return base64.b64decode(secret.value).decode("utf-8")


@contextmanager
def get_connection():
    """Yield a psycopg2 connection with a RealDictCursor factory."""
    conn = psycopg2.connect(_lakebase_url(), cursor_factory=RealDictCursor)
    try:
        yield conn
    finally:
        conn.close()


def run_query(sql: str, params=None) -> list[dict]:
    """Run a read query against Lakebase and return rows as list[dict]."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


def run_query_one(sql: str, params=None) -> dict | None:
    """Run a read query and return a single row (or None)."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()


def run_write(sql: str, params=None, returning: bool = False):
    """Run an INSERT/UPDATE/DELETE against Lakebase.

    Returns the RETURNING row (as a dict) if `returning` is True,
    otherwise the number of affected rows.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            result = cur.fetchone() if returning else cur.rowcount
            conn.commit()
            return result
