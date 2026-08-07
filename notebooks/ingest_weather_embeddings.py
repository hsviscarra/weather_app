# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Ingest Weather Documents -> Vector Embeddings (Lakebase)
# MAGIC
# MAGIC This notebook is part of the **Context Engineering on Databricks** course,
# MAGIC adapted for weather data (National Weather Service alerts + forecasts)
# MAGIC instead of stock ticker news.
# MAGIC
# MAGIC Unlike the stock-news template's embeddings notebook, this is a **plain
# MAGIC Python script using `psycopg2`** - no Spark, no `spark.read.jdbc` /
# MAGIC `spark.write.format("postgresql")`. Spark JDBC writes are not reliable
# MAGIC against this Lakebase instance, so every read and write here goes
# MAGIC through a psycopg2 connection, matching `lakebase.py`'s
# MAGIC `get_connection()` pattern (duplicated below rather than imported, so
# MAGIC this notebook is self-contained and doesn't depend on the parent
# MAGIC project folder being on `sys.path`).
# MAGIC
# MAGIC It:
# MAGIC 1. Reads rows from `weather_documents` that don't have any chunks in
# MAGIC    `weather_embeddings` yet (harvesting itself happens separately, via
# MAGIC    the Flask app's `POST /weather/sync` - this notebook only embeds
# MAGIC    whatever's already been synced).
# MAGIC 2. Chunks each document's `headline + narrative_text` with a sliding
# MAGIC    window (`CHUNK_SIZE=800`, `CHUNK_OVERLAP=100` by default) - most NWS
# MAGIC    text is short enough to stay a single chunk; longer alert
# MAGIC    descriptions split into several overlapping chunks.
# MAGIC 3. Embeds each chunk with `sentence-transformers/all-MiniLM-L6-v2`
# MAGIC    (384-dim, same model + dimension as `app.py`'s `/weather/search`,
# MAGIC    so query vectors and stored vectors stay comparable).
# MAGIC 4. Writes chunks into `weather_embeddings` via
# MAGIC    `psycopg2.extras.execute_values`, batched, casting each embedding to
# MAGIC    `%s::vector` directly in the SQL template.
# MAGIC
# MAGIC **The embedding model must load from a Unity Catalog Volume, not
# MAGIC download from Hugging Face at runtime** - serverless compute has no
# MAGIC reliable internet egress for that download. The very first time you run
# MAGIC this (or `app.py`), do so from compute that DOES have internet access
# MAGIC (e.g. a regular all-purpose cluster) so the model gets cached into the
# MAGIC Volume once; every run after that (including on serverless) reads the
# MAGIC already-cached files with `HF_HUB_OFFLINE=1` forced, so it never
# MAGIC attempts a network call at all.
# MAGIC
# MAGIC It re-uses the SAME Lakebase secret (scope `database`, key
# MAGIC `lakebase-url`) that `lakebase.py` uses in the Flask app, so no extra
# MAGIC secrets need to be created for this notebook.

# COMMAND ----------

# MAGIC %pip install -q sentence-transformers

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config

# COMMAND ----------

dbutils.widgets.text("documents_table_name", "weather_documents", "Source table (raw documents)")
dbutils.widgets.text("embeddings_table_name", "weather_embeddings", "Destination table (chunk vectors)")
dbutils.widgets.text("embedding_model", "sentence-transformers/all-MiniLM-L6-v2", "Embedding model")
dbutils.widgets.text("model_cache_path", "/Volumes/hv_external_catalog/weather_schema/ml_models", "Unity Catalog Volume for the cached model")
dbutils.widgets.text("chunk_size", "800", "Chunk size (chars)")
dbutils.widgets.text("chunk_overlap", "100", "Chunk overlap (chars)")
dbutils.widgets.text("batch_size", "100", "Rows per psycopg2 execute_values batch")

DOCUMENTS_TABLE_NAME = dbutils.widgets.get("documents_table_name")
EMBEDDINGS_TABLE_NAME = dbutils.widgets.get("embeddings_table_name")
EMBEDDING_MODEL_NAME = dbutils.widgets.get("embedding_model")
MODEL_CACHE_PATH = dbutils.widgets.get("model_cache_path")
CHUNK_SIZE = int(dbutils.widgets.get("chunk_size"))
CHUNK_OVERLAP = int(dbutils.widgets.get("chunk_overlap"))
BATCH_SIZE = int(dbutils.widgets.get("batch_size"))

# Must match app.py's EMBEDDING_MODEL_NAME / weather_embeddings' vector()
# column width. sentence-transformers/all-MiniLM-L6-v2 -> 384.
EMBEDDING_DIM = 384

print(f"Using model {EMBEDDING_MODEL_NAME!r} -> {EMBEDDING_DIM}-dim vectors, cached at {MODEL_CACHE_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resolve the Lakebase connection URL
# MAGIC
# MAGIC Same secret, same decoding scheme as `lakebase.py`.

# COMMAND ----------

import base64

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()


def get_lakebase_url() -> str:
    secret = w.secrets.get_secret(scope="database", key="lakebase-url")
    return base64.b64decode(secret.value).decode("utf-8")


LAKEBASE_URL = get_lakebase_url()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Connection helper (same pattern as lakebase.py's get_connection())

# COMMAND ----------

from contextlib import contextmanager

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values


@contextmanager
def get_connection():
    conn = psycopg2.connect(LAKEBASE_URL, cursor_factory=RealDictCursor)
    try:
        yield conn
    finally:
        conn.close()


with get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) AS n FROM {DOCUMENTS_TABLE_NAME}")
        print(f"Connection successful - {DOCUMENTS_TABLE_NAME} has {cur.fetchone()['n']} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ensure weather_embeddings + pgvector exist
# MAGIC
# MAGIC Mirrors `sql/02_setup_weather_embeddings_table.sql`, so this script is
# MAGIC self-sufficient even if that file wasn't run manually first.

# COMMAND ----------

with get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {EMBEDDINGS_TABLE_NAME} (
                id           TEXT PRIMARY KEY,
                document_id  TEXT NOT NULL REFERENCES {DOCUMENTS_TABLE_NAME}(id) ON DELETE CASCADE,
                chunk_index  INT NOT NULL,
                chunk_text   TEXT NOT NULL,
                embedding    VECTOR({EMBEDDING_DIM}) NOT NULL,
                model_name   TEXT NOT NULL,
                created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{EMBEDDINGS_TABLE_NAME}_embedding "
            f"ON {EMBEDDINGS_TABLE_NAME} USING hnsw (embedding vector_cosine_ops)"
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{EMBEDDINGS_TABLE_NAME}_document_id "
            f"ON {EMBEDDINGS_TABLE_NAME} (document_id)"
        )
        conn.commit()

print(f"{EMBEDDINGS_TABLE_NAME} ready (pgvector extension + HNSW index confirmed)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read unembedded documents
# MAGIC
# MAGIC Only documents with zero existing rows in `weather_embeddings` - a
# MAGIC re-run only processes what's new since the last run.

# COMMAND ----------

with get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT d.id, d.headline, d.narrative_text
            FROM {DOCUMENTS_TABLE_NAME} d
            WHERE NOT EXISTS (
                SELECT 1 FROM {EMBEDDINGS_TABLE_NAME} e WHERE e.document_id = d.id
            )
            AND coalesce(d.narrative_text, '') != ''
            """
        )
        documents = cur.fetchall()

print(f"Found {len(documents)} unembedded documents")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chunk each document's text
# MAGIC
# MAGIC Sliding window over `headline + narrative_text`. No external
# MAGIC article-URL fetch (unlike the stock-news template) - `narrative_text`
# MAGIC already IS the full content.

# COMMAND ----------


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    step = max(chunk_size - chunk_overlap, 1)
    chunks = []
    for start in range(0, len(text), step):
        piece = text[start : start + chunk_size].strip()
        if piece:
            chunks.append(piece)
        if start + chunk_size >= len(text):
            break
    return chunks


chunk_rows = []  # (document_id, chunk_index, chunk_text)
for doc in documents:
    full_text = f"{doc['headline'] or ''}. {doc['narrative_text'] or ''}".strip()
    for chunk_index, chunk in enumerate(chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)):
        chunk_rows.append((doc["id"], chunk_index, chunk))

print(f"Split {len(documents)} documents into {len(chunk_rows)} chunks")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the embedding model from the Volume (offline - no download)
# MAGIC
# MAGIC `HF_HUB_OFFLINE=1` forces sentence-transformers/huggingface_hub to
# MAGIC never attempt a network call, even a version-check ping - it either
# MAGIC finds the model already cached at `MODEL_CACHE_PATH` or raises
# MAGIC immediately, rather than hanging or silently trying (and failing) to
# MAGIC reach Hugging Face from serverless compute.
# MAGIC
# MAGIC If this raises `OSError` on first run, the model isn't cached in the
# MAGIC Volume yet - run this notebook once from compute with internet access
# MAGIC (e.g. a regular all-purpose cluster, with `HF_HUB_OFFLINE` unset) to
# MAGIC populate the Volume, then switch back to serverless.

# COMMAND ----------

import os

os.environ["HF_HUB_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer

print(f"Loading {EMBEDDING_MODEL_NAME} from {MODEL_CACHE_PATH} (offline)...")
model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder=MODEL_CACHE_PATH)
actual_dim = model.get_sentence_embedding_dimension()
if actual_dim != EMBEDDING_DIM:
    raise ValueError(
        f"Model dimension {actual_dim} != expected {EMBEDDING_DIM} - "
        f"update EMBEDDING_DIM here and weather_embeddings' vector() column."
    )
print(f"Model loaded, {actual_dim}-dim vectors")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Embed + write chunks in batches
# MAGIC
# MAGIC `execute_values` with a custom `template` casts each embedding to
# MAGIC pgvector's type inline (`%s::vector`) - no Spark JDBC array-then-cast
# MAGIC two-step needed.

# COMMAND ----------


def to_vector_literal(vec) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


written = 0
with get_connection() as conn:
    with conn.cursor() as cur:
        for i in range(0, len(chunk_rows), BATCH_SIZE):
            batch = chunk_rows[i : i + BATCH_SIZE]
            texts = [c[2] for c in batch]
            vectors = model.encode(texts, show_progress_bar=False)

            values = [
                (
                    f"{doc_id}_{chunk_index}",
                    doc_id,
                    chunk_index,
                    chunk_text_val,
                    to_vector_literal(vector),
                    EMBEDDING_MODEL_NAME,
                )
                for (doc_id, chunk_index, chunk_text_val), vector in zip(batch, vectors)
            ]

            execute_values(
                cur,
                f"""
                INSERT INTO {EMBEDDINGS_TABLE_NAME}
                    (id, document_id, chunk_index, chunk_text, embedding, model_name)
                VALUES %s
                ON CONFLICT (id) DO UPDATE
                    SET chunk_text = EXCLUDED.chunk_text,
                        embedding = EXCLUDED.embedding,
                        model_name = EXCLUDED.model_name
                """,
                values,
                template="(%s, %s, %s, %s, %s::vector, %s)",
            )
            conn.commit()
            written += len(values)
            print(f"Wrote batch {i // BATCH_SIZE + 1}: {len(values)} chunks (running total {written})")

print(f"Done - wrote {written} chunk embeddings to {EMBEDDINGS_TABLE_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify

# COMMAND ----------

with get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) AS n FROM {DOCUMENTS_TABLE_NAME}")
        print(f"{DOCUMENTS_TABLE_NAME}: {cur.fetchone()['n']} rows")

        cur.execute(f"SELECT count(*) AS n, count(DISTINCT document_id) AS docs FROM {EMBEDDINGS_TABLE_NAME}")
        row = cur.fetchone()
        print(f"{EMBEDDINGS_TABLE_NAME}: {row['n']} chunk rows across {row['docs']} documents")

        cur.execute(
            f"SELECT id, document_id, chunk_index, model_name, created_at "
            f"FROM {EMBEDDINGS_TABLE_NAME} ORDER BY created_at DESC LIMIT 5"
        )
        print("\nMost recent chunks:")
        for r in cur.fetchall():
            print(dict(r))
