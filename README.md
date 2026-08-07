# Weather Watchlist (Lakebase + NWS API + Semantic Search)

A small Databricks App backed by Lakebase, structured the same way as the course's
[`databricks-lakebase-app-day-2`](https://github.com/hsviscarra/databricks-lakebase-app-day-2)
template, but pulling weather alerts and forecasts from the free, keyless
[National Weather Service API](https://www.weather.gov/documentation/services-web-api)
instead of stock news from Massive.

It harvests alerts/forecasts into a raw document store, chunks and embeds them
into a pgvector table via a scheduled notebook, and exposes a semantic search
endpoint that embeds a query at request time and finds the closest chunks.

## Files

- `app.py` - Flask app: `/healthz`, `/watchlist` (GET/POST/DELETE), `/weather/sync` (POST), `/documents` (GET), `/weather/search` (POST)
- `lakebase.py` - Lakebase connection helper (same pattern as the day-2 template; auth details below)
- `weather_client.py` - NWS API client (`get_active_alerts`, `get_forecast_periods`) + Nominatim geocoding (`geocode`)
- `setup_secrets.py` - One-time script to create the `database` secret scope and store the Lakebase URL
- `app.yaml` - Databricks App deployment config
- `templates/index.html` - Watchlist UI (add/remove locations, sync, view recent documents)
- `sql/01_setup_weather_documents_table.sql` - Raw document store DDL
- `sql/02_setup_weather_embeddings_table.sql` - pgvector chunk-embeddings table DDL
- `notebooks/ingest_weather_embeddings.py` - Self-contained ETL: reads locations from `weather_watchlist`, fetches alerts/forecasts from NWS, upserts into `weather_documents`, chunks + embeds into `weather_embeddings`
- `databricks.yml` + `resources/ingest_weather_embeddings_job.yml` - Asset Bundle config that schedules the notebook above as an hourly Workflow

## How Lakebase authentication works

`lakebase.py` resolves one Postgres connection URL two ways, in order:

1. **`LAKEBASE_URL` environment variable** - used for local development (`.env`). Never set on the deployed app.
2. **Databricks secret scope** - `WorkspaceClient().secrets.get_secret(scope="database", key="lakebase-url")`, decoding the base64 value the Secrets API returns. `WorkspaceClient()` is constructed *lazily*, only inside this fallback branch - constructing it eagerly at import time was measured to hang for a long time against unreachable/invalid credentials, which would otherwise slow every cold start even when `LAKEBASE_URL` already satisfies the connection.

This is a **native Postgres role with a static password** (set up once via `setup_secrets.py`), not an OAuth token that needs refreshing.

For the deployed app to reach step 2, its service principal needs read access to the secret - grant this via **App resources -> Add resource -> Secret** (scope `database`, key `lakebase-url`, permission **Can read**) in the Databricks Apps UI when creating/configuring the app, rather than a manual ACL command.

## Tables

```
weather_documents            -- raw alerts + forecast periods, one row each
  id              TEXT PRIMARY KEY   -- alert's own id, or hash(location + period startTime + name) for forecasts
  location        TEXT NOT NULL      -- e.g. "Chicago, IL" or a watchlist label
  source_type     TEXT NOT NULL      -- 'alert' | 'forecast'
  headline        TEXT NOT NULL      -- alert's headline/event, or "<period name>: <shortForecast>"
  narrative_text  TEXT               -- alert's description/instruction, or forecast's detailedForecast
  issued_at       TIMESTAMPTZ        -- alert's sent time, or forecast period's startTime
  effective_at    TIMESTAMPTZ        -- alert's effective time, or forecast period's endTime
  payload         JSONB NOT NULL     -- raw API response, for provenance
  synced_at       TIMESTAMPTZ NOT NULL DEFAULT now()

weather_embeddings            -- one row per CHUNK of a document's text, not one row per document
  id           TEXT PRIMARY KEY      -- "<document_id>_<chunk_index>"
  document_id  TEXT NOT NULL REFERENCES weather_documents(id) ON DELETE CASCADE
  chunk_index  INT NOT NULL
  chunk_text   TEXT NOT NULL
  embedding    VECTOR(384) NOT NULL  -- dimension must match EMBEDDING_MODEL_NAME
  model_name   TEXT NOT NULL
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now()

weather_watchlist              -- created automatically by app.py, not a manual SQL file
  label      TEXT NOT NULL
  lat        NUMERIC NOT NULL
  lon        NUMERIC NOT NULL
  email      TEXT NOT NULL
  added_at   TIMESTAMPTZ NOT NULL DEFAULT now()
  PRIMARY KEY (lat, lon, email)
```

Most forecast periods and many alerts are short enough to produce exactly
one chunk (`chunk_index = 0`); long multi-paragraph alert descriptions split
into several overlapping chunks. The table structure supports both without
special-casing.

## How the harvest works - `POST /weather/sync`

```
Body (optional JSON): {"locations": ["Chicago, IL", "Austin, TX"], "limit": 50}
```

- **`locations`** - place names. Each is geocoded to lat/lon via the free,
  keyless [Nominatim](https://nominatim.org/) (OpenStreetMap) API
  (`weather_client.geocode`), then alerts + forecast periods are fetched for
  that point from NWS. Geocoding calls are paced 1/sec to respect Nominatim's
  usage policy. A location that fails to geocode is skipped (logged, not
  fatal to the request).

  **Known risk**: Nominatim's public demo server actively blocks requests
  from cloud/data-center IP ranges (`403 Access denied`, verified during
  development - not a header/rate issue, a policy-level IP block). Since
  Databricks compute also runs from cloud IP ranges, `POST /weather/sync`
  with `locations` may return `{"synced": 0, ...}` in your workspace too. If
  that happens, use the reliable fallback below instead of debugging
  Nominatim further, or point `GEOCODE_API_BASE_URL` at a geocoder that
  explicitly allows your traffic (self-hosted Nominatim, a paid geocoding
  API, etc).
- **If `locations` is omitted** (or an empty body, e.g. no JSON at all) - falls back to the deployed app's `weather_watchlist` entries, which are already lat/lon and need no geocoding, so this path never depends on Nominatim. This is what the UI's "Sync now" button calls, and the one to use if geocoding is unreliable in your environment.
- **`limit`** - max alerts *and* max forecast periods to persist per location (applied independently to each `source_type`, not shared between them). Defaults to 50, clamped to `[1, 200]`.

Each alert/forecast period is upserted into `weather_documents` by its stable
id (`ON CONFLICT (id) DO UPDATE`), so re-syncing the same location doesn't
create duplicates - it refreshes `synced_at` and any changed fields instead.

## How vectorization works

Two separate paths write and read vectors, both loading the **same model
from the same place** so query vectors and stored vectors are comparable:

- **Write (batch)** - `notebooks/ingest_weather_embeddings.py`, run in Databricks/Spark, same pattern as the project template: reads `weather_documents` via JDBC, chunks `headline + narrative_text` into overlapping windows, embeds each chunk with a distributed pandas UDF (`sentence-transformers`), and writes to `weather_embeddings` via JDBC batch writes. `pgvector`'s `VECTOR` type isn't JDBC-native, so vectors land as `DOUBLE PRECISION[]` and a `UPDATE ... SET embedding = embedding::vector` cast step (printed by the notebook) finishes the job.
- **Read (live)** - `app.py`'s `POST /weather/search` embeds the incoming query with the same model, on the Flask process itself, then runs a pgvector nearest-neighbor query.

**The model lives in a Unity Catalog Volume** (`MODEL_CACHE_PATH`, default
`/Volumes/hv_external_catalog/weather_schema/ml_models` - adjust to a
catalog/schema your workspace actually has). Both the notebook and the Flask
app pass this same path as `cache_folder` to `SentenceTransformer(...)`, so
whichever one downloads the model first (usually the notebook, since it runs
first) leaves it cached there, and the other reads the same files instead of
re-fetching from Hugging Face. For the deployed app to read that Volume, attach it as a resource in the Apps UI (**App resources -> Add resource -> Volume**, alongside the Secret resource from the auth section above).

In `app.py`, the model is loaded **lazily and cached in memory**
(`_get_embedding_model()`) - Flask starts instantly and only pays the load
cost the first time `/weather/search` is actually called, not on every
request or on cold start if search is never used.

## How the write works

Same as the day-2 template: JDBC batch writes from Spark, with a left-anti
join against existing ids to avoid re-inserting chunks that are already
embedded, followed by a manual `::vector` cast step since Spark's JDBC
writer can't target `pgvector`'s native type directly. See
"Upsert embeddings into Lakebase" in the notebook.

## The retrieval endpoint - `POST /weather/search`

```
Body: {"query": "risk of flooding near rivers", "top_k": 5}
```

1. Validates `query` (non-empty string) and clamps `top_k` to `[1, 20]`.
2. Embeds `query` with `_get_embedding_model()`.
3. Runs a pgvector cosine-distance nearest-neighbor query against
   `weather_embeddings`, joined back to `weather_documents` for context
   (`location`, `source_type`, `headline`, `issued_at`), ordered by
   `embedding <=> query_vector` and limited to `top_k`.
4. Returns `{"results": [...], "query": ..., "top_k": ...}`, where each
   result includes a `similarity` score (`1 - cosine_distance`, so higher is
   more relevant).

The query vector is passed as a bound parameter (a bracketed string literal
cast with `::vector`), not string-interpolated into the SQL - psycopg2 has
no native pgvector adapter, but this pattern is still fully parameterized
and not a SQL-injection risk.

## Edge cases

| Case | Behavior |
|---|---|
| `weather_embeddings` table doesn't exist yet (SQL setup never run) | `POST /weather/search` catches `psycopg2.errors.UndefinedTable` and returns `{"results": [], "message": "..."}` with a `200`, not an error - this is an expected pre-ingestion state. |
| `weather_embeddings` exists but is empty (setup run, notebook never run) | Query returns zero rows; same `{"results": [], "message": "No embeddings synced yet."}` response. |
| Missing/malformed `query` | `400` with `{"error": "'query' is required and must be a non-empty string"}`. Non-string or empty-after-strip both count as missing. |
| `top_k` out of bounds (e.g. `0`, `500`, negative) | **Clamped**, not rejected - `max(1, min(top_k, 20))`. A non-integer `top_k` (e.g. a string that can't parse) is a `400`. |
| Embedding model fails to load (e.g. Volume not attached, model files missing) | `503` with `{"error": "Search is temporarily unavailable ..."}`, logged server-side with the full traceback. |
| `POST /weather/sync` with a `locations` entry that fails to geocode | That location is skipped (logged as a warning) and the rest of the batch still runs - one bad location name doesn't fail the whole sync. |
| `POST /weather/sync` with no `locations` and an empty watchlist | `200` with `{"synced": 0, "locations": 0, "message": "No locations given and watchlist is empty"}`. |

## Step-by-step setup

### 1. Create a Lakebase instance and a native-password role

Same as the day-2 template: **Catalog -> Lakebase -> Create Lakebase instance**, then under **Roles & Databases**, enable password auth and create a role. Copy the connection URL - you'll paste it into `setup_secrets.py` below.

### 2. Create the schema

Run manually in the Lakebase SQL editor (or via `psql`):
1. `sql/01_setup_weather_documents_table.sql`
2. `sql/02_setup_weather_embeddings_table.sql`

`weather_watchlist` doesn't need a manual script - the Flask app creates it on first run.

### 3. Store your secret

From a Databricks notebook attached to any running cluster:

```python
%sh python setup_secrets.py
```

If the `database` scope already exists from a previous project, `setup_secrets.py` skips scope creation gracefully by default (`create_scope` is commented out - only uncomment for the very first run ever in a workspace).

### 4. Create a Unity Catalog Volume for the embedding model (optional but recommended)

If you don't already have one, create a Volume (**Catalog -> your catalog -> your schema -> Create -> Volume**) and update `MODEL_CACHE_PATH` in `app.yaml` and the `nws_user_agent`/model widgets in `resources/ingest_weather_embeddings_job.yml` to match its path. Without this, both the notebook and the app still work - they just each download the model from Hugging Face independently instead of sharing a cached copy.

### 5. Grant the deployed app access to the secret (and the Volume)

In the Databricks Apps UI, **App resources -> Add resource**:
- **Secret**: scope `database`, key `lakebase-url`, permission **Can read**.
- **Volume**: the one from step 4, if you created it.

### 6. Run locally (optional)

```bash
cp .env.example .env   # paste your Lakebase URL into LAKEBASE_URL
pip install -r requirements.txt
python app.py
```

Visit `http://localhost:8000`. `/weather/sync` and NWS calls work locally exactly as deployed, since NWS needs no key. `/weather/search` also works locally - `sentence-transformers` downloads the model straight from Hugging Face into its default local cache the first time it's called, since there's no Volume mount outside Databricks.

### 7. Deploy as a Databricks App

Link the GitHub repo directly in the Databricks Apps "Create new app" wizard's Git step, attach the resources from step 5, and deploy.

### 8. Verify

1. Open the deployed app, add a location, click **Sync now** (or `curl -X POST <app-url>/weather/sync -d '{"locations":["Chicago, IL"]}'`).
2. Confirm rows in `weather_documents`: `SELECT * FROM weather_documents ORDER BY synced_at DESC;`
3. Run the notebook (manually first, see below) to populate `weather_embeddings`.
4. `curl -X POST <app-url>/weather/search -d '{"query":"flooding risk","top_k":3}'`

## Scheduling the embeddings notebook

Same two options as the day-2 template:

**Option A - Asset Bundle (recommended, version-controlled):**
```bash
databricks bundle deploy -t dev
databricks bundle run ingest_weather_embeddings_job -t dev   # test once manually
```
Then flip `pause_status` to `UNPAUSED` in `resources/ingest_weather_embeddings_job.yml` and redeploy.

**Option B - Workflows UI:** create a Job with a Notebook task pointing at `notebooks/ingest_weather_embeddings.py`, set the same `base_parameters` shown in `resources/ingest_weather_embeddings_job.yml`, and add an hourly schedule.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/healthz` | health check |
| GET | `/watchlist` | current user's watched locations |
| POST | `/watchlist` | add/relabel a location `{label?, lat, lon}` |
| DELETE | `/watchlist` | remove a location `{lat, lon}` |
| POST | `/weather/sync` | fetch alerts + forecasts, upsert into `weather_documents` - `{locations?, limit?}` |
| GET | `/documents?location=&source_type=&limit=` | read synced documents |
| POST | `/weather/search` | semantic search over `weather_embeddings` - `{query, top_k?}` |
