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
# MAGIC It:
# MAGIC 1. Reads the `weather_watchlist` table in Lakebase to find out which
# MAGIC    locations (lat/lon) are currently being tracked.
# MAGIC 2. Fetches active alerts (`GET /alerts/active?point=`) and forecast
# MAGIC    periods (`GET /points/...` -> `GET /gridpoints/.../forecast`) for
# MAGIC    those locations directly from the free, keyless NWS API
# MAGIC    (api.weather.gov - see weather_client.py for the same call shape
# MAGIC    used by the Flask app's `POST /sync` route), and upserts the results
# MAGIC    into the `weather_documents` table.
# MAGIC 3. Computes a sentence embedding for each document's narrative text
# MAGIC    using Spark, distributed across the cluster via a pandas UDF, and
# MAGIC    writes them into a `weather_documents_embeddings` table using the
# MAGIC    `pgvector` Postgres extension so downstream RAG / context-engineering
# MAGIC    exercises can run similarity search directly in Postgres.
# MAGIC
# MAGIC It re-uses the SAME Lakebase secret (scope `database`, key `lakebase-url`)
# MAGIC that `lakebase.py` uses in the Flask app, so no extra secrets need to be
# MAGIC created for this notebook. Unlike the stock-news version of this
# MAGIC notebook, there's no chunking/full-article-scraping step - alert and
# MAGIC forecast text is already short and self-contained, so one embedding per
# MAGIC document is enough. There's also no strict rate limit to work around -
# MAGIC the NWS API is free and keyless, it just requires a descriptive
# MAGIC User-Agent header.

# COMMAND ----------

# MAGIC %pip install -q sentence-transformers requests

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config
# MAGIC
# MAGIC Widgets let you override the source/destination table names and the
# MAGIC embedding model without editing the notebook - useful when running this
# MAGIC as a scheduled Databricks Job.

# COMMAND ----------

dbutils.widgets.text("watchlist_table_name", "weather_watchlist", "Source table (watched locations)")
dbutils.widgets.text("documents_table_name", "weather_documents", "Destination table (raw documents)")
dbutils.widgets.text("embeddings_table_name", "weather_documents_embeddings", "Destination table (vectors)")
dbutils.widgets.text("embedding_model", "sentence-transformers/all-MiniLM-L6-v2", "Embedding model")
dbutils.widgets.text("nws_user_agent", "(weather_app, you@example.com)", "NWS API User-Agent (required by api.weather.gov)")

WATCHLIST_TABLE_NAME = dbutils.widgets.get("watchlist_table_name")
DOCUMENTS_TABLE_NAME = dbutils.widgets.get("documents_table_name")
EMBEDDINGS_TABLE_NAME = dbutils.widgets.get("embeddings_table_name")
EMBEDDING_MODEL_NAME = dbutils.widgets.get("embedding_model")
NWS_USER_AGENT = dbutils.widgets.get("nws_user_agent")
NWS_BASE_URL = "https://api.weather.gov"

# Different sentence-transformers models emit different vector sizes, and the
# pgvector column type (VECTOR(N)) must match exactly. Rather than hardcoding
# one dimension, switch on the model name so swapping EMBEDDING_MODEL_NAME via
# the widget above automatically resizes the destination table's vector column.
match EMBEDDING_MODEL_NAME:
    case "sentence-transformers/all-MiniLM-L6-v2":
        EMBEDDING_DIM = 384
    case "sentence-transformers/all-MiniLM-L12-v2":
        EMBEDDING_DIM = 384
    case "sentence-transformers/all-mpnet-base-v2":
        EMBEDDING_DIM = 768
    case "sentence-transformers/paraphrase-multilingual-mpnet-base-v2":
        EMBEDDING_DIM = 768
    case "BAAI/bge-small-en-v1.5":
        EMBEDDING_DIM = 384
    case "BAAI/bge-base-en-v1.5":
        EMBEDDING_DIM = 768
    case "BAAI/bge-large-en-v1.5":
        EMBEDDING_DIM = 1024
    case _:
        raise ValueError(
            f"Unknown embedding model {EMBEDDING_MODEL_NAME!r} - add its output "
            "dimension to the match/case block above before running this notebook."
        )

print(f"Using model {EMBEDDING_MODEL_NAME!r} -> {EMBEDDING_DIM}-dim vectors")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resolve the Lakebase connection URL
# MAGIC
# MAGIC Same secret, same decoding scheme as `lakebase.py`: a single base64-encoded
# MAGIC Postgres URL (`postgresql://role:password@host:5432/db?sslmode=require`)
# MAGIC stored in a Databricks secret scope. We parse it into the pieces Spark's
# MAGIC JDBC reader/writer need (url/user/password).

# COMMAND ----------

import base64
from urllib.parse import urlparse

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()


def get_lakebase_url() -> str:
    secret = w.secrets.get_secret(scope="database", key="lakebase-url")
    return base64.b64decode(secret.value).decode("utf-8")


lakebase_url = get_lakebase_url()
parsed = urlparse(lakebase_url)

jdbc_url = f"jdbc:postgresql://{parsed.hostname}:{parsed.port or 5432}{parsed.path}"
jdbc_properties = {
    "user": parsed.username,
    "password": parsed.password,
    "driver": "org.postgresql.Driver",
    "sslmode": "require",
}
print(f"Connecting to: {parsed.hostname}:{parsed.port or 5432}{parsed.path}")

# COMMAND ----------

# DBTITLE 1,Test JDBC Connection
try:
    test_df = spark.read.jdbc(url=jdbc_url, table=WATCHLIST_TABLE_NAME, properties=jdbc_properties)
    count = test_df.count()
    print(f"Connection successful! Found {count} rows in {WATCHLIST_TABLE_NAME}")
    test_df.show(5)
except Exception as e:
    print(f"Connection failed: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Database Setup Instructions
# MAGIC
# MAGIC Before running this notebook, manually create the required tables in
# MAGIC your Lakebase Postgres database:
# MAGIC
# MAGIC 1. Run `sql/01_setup_weather_documents_table.sql` to create `weather_documents`
# MAGIC 2. Run `sql/02_setup_weather_documents_embeddings_table.sql` to create
# MAGIC    `weather_documents_embeddings` - replace `{{EMBEDDING_DIM}}` with the
# MAGIC    value printed above (384 for the default model).
# MAGIC
# MAGIC `weather_watchlist` doesn't need a manual setup script - the Flask app
# MAGIC creates it automatically (`ensure_watchlist_table()` in app.py) the
# MAGIC first time it runs.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fetch alerts + forecasts from the NWS API for watchlisted locations
# MAGIC
# MAGIC This ETL is self-contained: it queries the `weather_watchlist` table
# MAGIC directly (not the Flask app's `/sync` route) so the scheduled Job doesn't
# MAGIC depend on anyone having triggered a sync from the UI. Requests are made
# MAGIC serially with a short pause between locations as a courtesy to the API -
# MAGIC unlike the Massive API in the stock-news version of this notebook, NWS
# MAGIC has no strict quota to enforce.

# COMMAND ----------

import hashlib
import json as _json
import time

import requests

session = requests.Session()
session.headers.update({"User-Agent": NWS_USER_AGENT, "Accept": "application/geo+json"})


def get_watchlist_locations() -> list[dict]:
    """Distinct watched locations across all users - these are the only
    locations we fetch alerts/forecasts for."""
    df = spark.read.jdbc(url=jdbc_url, table=WATCHLIST_TABLE_NAME, properties=jdbc_properties)
    return [row.asDict() for row in df.select("label", "lat", "lon").distinct().collect()]


def get_active_alerts(lat: float, lon: float) -> list[dict]:
    """Single GET /alerts/active?point= call (mirrors
    WeatherClient.get_active_alerts in weather_client.py)."""
    resp = session.get(f"{NWS_BASE_URL}/alerts/active", params={"point": f"{lat},{lon}"}, timeout=30)
    resp.raise_for_status()
    return resp.json().get("features", [])


def get_forecast_periods(lat: float, lon: float) -> list[dict]:
    """Two calls: resolve the point to its forecast grid, then fetch periods
    (mirrors WeatherClient.get_forecast_periods)."""
    point = session.get(f"{NWS_BASE_URL}/points/{lat},{lon}", timeout=30)
    point.raise_for_status()
    forecast_url = point.json()["properties"]["forecast"]
    resp = session.get(forecast_url, timeout=30)
    resp.raise_for_status()
    return resp.json()["properties"]["periods"]


def forecast_doc_id(lat: float, lon: float, period: dict) -> str:
    """Stable dedup key for a forecast period (no natural id in the API
    response) - hash location + start time + period name."""
    raw = f"{lat:.4f},{lon:.4f}:{period.get('startTime', '')}:{period.get('name', '')}"
    return "forecast:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


rows = []
locations = get_watchlist_locations()
print(f"Found {len(locations)} distinct watchlisted locations")

for i, loc in enumerate(locations):
    if i > 0:
        time.sleep(1)  # light courtesy pause between locations
    label, lat, lon = loc["label"], float(loc["lat"]), float(loc["lon"])

    try:
        for feature in get_active_alerts(lat, lon):
            props = feature.get("properties", {})
            rows.append(
                {
                    "id": props.get("id"),
                    "location": label,
                    "source_type": "alert",
                    "headline": props.get("headline") or props.get("event", ""),
                    "narrative_text": props.get("description") or props.get("instruction") or "",
                    "issued_at": props.get("sent"),
                    "effective_at": props.get("effective"),
                    "payload": _json.dumps(feature),
                }
            )
    except Exception as exc:
        print(f"Skipping alerts for {label}: {exc}")

    try:
        for period in get_forecast_periods(lat, lon):
            name = period.get("name", "")
            short = period.get("shortForecast", "")
            rows.append(
                {
                    "id": forecast_doc_id(lat, lon, period),
                    "location": label,
                    "source_type": "forecast",
                    "headline": f"{name}: {short}" if short else name,
                    "narrative_text": period.get("detailedForecast") or "",
                    "issued_at": period.get("startTime"),
                    "effective_at": period.get("endTime"),
                    "payload": _json.dumps(period),
                }
            )
    except Exception as exc:
        print(f"Skipping forecast for {label}: {exc}")

print(f"Fetched {len(rows)} documents across {len(locations)} locations")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Upsert raw documents into Lakebase
# MAGIC
# MAGIC Deduplicates against existing IDs (left anti-join) so re-running the
# MAGIC job doesn't create duplicate rows for alerts/forecast periods already
# MAGIC synced.

# COMMAND ----------

from pyspark.sql.functions import current_timestamp, to_timestamp

if rows:
    new_df = spark.createDataFrame(rows).withColumn("synced_at", current_timestamp())
    for ts_col in ("issued_at", "effective_at"):
        new_df = new_df.withColumn(ts_col, to_timestamp(ts_col))

    try:
        existing_ids = (
            spark.read.jdbc(url=jdbc_url, table=DOCUMENTS_TABLE_NAME, properties=jdbc_properties)
            .select("id")
            .distinct()
        )
        new_df = new_df.join(existing_ids, on="id", how="left_anti")
    except Exception:
        # Table might not exist yet or be empty - that's fine, write all rows.
        pass

    synced_count = new_df.count()
    if synced_count > 0:
        new_df.write.format("postgresql") \
            .option("host", parsed.hostname) \
            .option("port", parsed.port or 5432) \
            .option("database", parsed.path.lstrip("/")) \
            .option("dbtable", DOCUMENTS_TABLE_NAME) \
            .option("user", parsed.username) \
            .option("password", parsed.password) \
            .mode("append") \
            .save()
        print(f"Wrote {synced_count} new documents to {DOCUMENTS_TABLE_NAME}")
    else:
        print("No new documents (all already synced).")
else:
    print("No documents fetched - is weather_watchlist empty?")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load documents with Spark
# MAGIC
# MAGIC Reads the whole `weather_documents` table (just synced above) via JDBC
# MAGIC into a Spark DataFrame so embedding computation can be distributed
# MAGIC across the cluster.

# COMMAND ----------

docs_df = (
    spark.read.jdbc(url=jdbc_url, table=DOCUMENTS_TABLE_NAME, properties=jdbc_properties)
    .selectExpr(
        "id",
        "location",
        "headline",
        "issued_at",
        # Embed on headline + narrative_text together for richer context.
        "trim(concat(coalesce(headline, ''), '. ', coalesce(narrative_text, ''))) AS embedding_text",
    )
    .filter("embedding_text IS NOT NULL AND embedding_text != ''")
)

print(f"Loaded {docs_df.count()} weather documents from {DOCUMENTS_TABLE_NAME}")
display(docs_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute embeddings (distributed pandas UDF)
# MAGIC
# MAGIC Loads the sentence-transformers model once per executor process (not per
# MAGIC row) and applies it in batches via `mapInPandas`, which scales across
# MAGIC however many workers the cluster has.

# COMMAND ----------

# DBTITLE 1,Download embedding model to a Volume
from sentence_transformers import SentenceTransformer

# Unity Catalog Volume path (writable, accessible by all workers) - adjust
# the catalog/schema to one your workspace actually has.
MODEL_CACHE_PATH = "/Volumes/hv_external_catalog/weather_schema/ml_models"

print(f"Downloading {EMBEDDING_MODEL_NAME} to {MODEL_CACHE_PATH}...")
model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder=MODEL_CACHE_PATH)
print(f"Model downloaded and cached at {MODEL_CACHE_PATH}")

# COMMAND ----------

from typing import Iterator

import pandas as pd
from pyspark.sql.types import ArrayType, FloatType, StringType, StructField, StructType

embeddings_schema = StructType(
    [
        StructField("id", StringType(), False),
        StructField("location", StringType(), False),
        StructField("headline", StringType(), False),
        StructField("issued_at", StringType(), True),
        StructField("embedding", ArrayType(FloatType()), False),
    ]
)


def embed_partitions(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    """Runs once per Spark partition/task: load the model once, then embed
    every batch of rows handed to this partition."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder=MODEL_CACHE_PATH)

    for batch in iterator:
        vectors = model.encode(batch["embedding_text"].tolist(), show_progress_bar=False)
        yield pd.DataFrame(
            {
                "id": batch["id"],
                "location": batch["location"],
                "headline": batch["headline"],
                "issued_at": batch["issued_at"].astype(str),
                "embedding": [v.tolist() for v in vectors],
            }
        )


embeddings_df = docs_df.mapInPandas(embed_partitions, schema=embeddings_schema)

print(f"Computed {embeddings_df.count()} embeddings using {EMBEDDING_MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ensure the pgvector destination table exists
# MAGIC
# MAGIC `pgvector` isn't a JDBC-native type, but plain SQL text (`vector(N)`,
# MAGIC `::vector` casts) works fine over a raw JDBC connection - no psycopg2
# MAGIC needed.

# COMMAND ----------

print(f"Required EMBEDDING_DIM for SQL setup: {EMBEDDING_DIM}")
print(f"Table name: {EMBEDDINGS_TABLE_NAME}")
print("\nRun sql/02_setup_weather_documents_embeddings_table.sql in your Lakebase database before continuing.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Upsert embeddings into Lakebase
# MAGIC
# MAGIC Written via JDBC batch writes. Each embedding lands as a
# MAGIC `DOUBLE PRECISION[]` array - cast to Postgres' `vector` type via the
# MAGIC `::vector` SQL step printed at the end of this cell.

# COMMAND ----------

from pyspark.sql.functions import col, current_timestamp, expr, lit, to_timestamp

embeddings_with_meta = (
    embeddings_df.withColumn("model_name", lit(EMBEDDING_MODEL_NAME))
    .withColumn("embedded_at", current_timestamp())
    .withColumn("issued_at", to_timestamp(col("issued_at")))
)

# Convert embedding array from ArrayType(FloatType) to ArrayType(DoubleType) -
# Postgres JDBC expects DOUBLE PRECISION[] for vector columns.
embeddings_final = embeddings_with_meta.withColumn(
    "embedding", expr("transform(embedding, x -> cast(x as double))")
)

# Read existing embedding IDs to avoid duplicates.
try:
    existing_ids = (
        spark.read.jdbc(url=jdbc_url, table=EMBEDDINGS_TABLE_NAME, properties=jdbc_properties)
        .select("id")
        .distinct()
    )
    new_embeddings = embeddings_final.join(existing_ids, on="id", how="left_anti")
except Exception:
    new_embeddings = embeddings_final

embedding_count = new_embeddings.count()
if embedding_count > 0:
    new_embeddings.write.format("postgresql") \
        .option("host", parsed.hostname) \
        .option("port", parsed.port or 5432) \
        .option("database", parsed.path.lstrip("/")) \
        .option("dbtable", EMBEDDINGS_TABLE_NAME) \
        .option("user", parsed.username) \
        .option("password", parsed.password) \
        .mode("append") \
        .save()
    print(f"Wrote {embedding_count} new embeddings to {EMBEDDINGS_TABLE_NAME}")
    print("\nIMPORTANT: Run this SQL in your Lakebase database to cast arrays to vectors:")
    print(f"  UPDATE {EMBEDDINGS_TABLE_NAME} SET embedding = embedding::vector WHERE embedding IS NOT NULL;")
else:
    print("No new embeddings to write (all already exist).")

# COMMAND ----------

# DBTITLE 1,Verify All Tables
print("CHECKING TABLES FOR RECORDS")
print("=" * 100)

print(f"\n{DOCUMENTS_TABLE_NAME} (Raw Alerts + Forecasts)")
print("-" * 100)
try:
    docs_check = spark.read.jdbc(url=jdbc_url, table=DOCUMENTS_TABLE_NAME, properties=jdbc_properties)
    docs_count = docs_check.count()
    print(f"Total rows: {docs_count}")
    if docs_count > 0:
        print(f"\nDistinct locations: {docs_check.select('location').distinct().count()}")
        print("\nBy source_type:")
        docs_check.groupBy("source_type").count().show()
        print("\nSample records:")
        docs_check.select("id", "location", "source_type", "headline", "synced_at").show(5, truncate=50)
    else:
        print("No records found!")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 100)
print(f"{EMBEDDINGS_TABLE_NAME} (Vectors)")
print("-" * 100)
try:
    embeddings_check = spark.read.jdbc(url=jdbc_url, table=EMBEDDINGS_TABLE_NAME, properties=jdbc_properties)
    embeddings_count = embeddings_check.count()
    print(f"Total rows: {embeddings_count}")
    if embeddings_count > 0:
        print("\nSample records:")
        embeddings_check.select("id", "location", "headline", "model_name", "embedded_at").show(5, truncate=50)
        first_embedding = embeddings_check.select("embedding").first()
        if first_embedding:
            emb_array = first_embedding["embedding"]
            print(f"\nEmbedding dimension: {len(emb_array)} (should be {EMBEDDING_DIM})")
    else:
        print("No records found!")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 100)
print("VERIFICATION COMPLETE")
print("=" * 100)
