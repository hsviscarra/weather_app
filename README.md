# Weather Watchlist (Lakebase + NWS API)

A small Databricks App backed by Lakebase, structured the same way as the course's
[`databricks-lakebase-app-day-2`](https://github.com/hsviscarra/databricks-lakebase-app-day-2)
template, but pulling weather alerts and forecasts from the free, keyless
[National Weather Service API](https://www.weather.gov/documentation/services-web-api)
instead of stock news from Massive.

Users track locations (lat/lon). Syncing pulls active alerts and forecast
periods for each watched location and normalizes them into a raw document
store (`weather_documents`), which a scheduled notebook turns into vector
embeddings (`weather_documents_embeddings`) for downstream RAG /
context-engineering exercises.

## Files

- `app.py` - Flask app: `/healthz`, `/watchlist` (GET/POST/DELETE), `/sync` (POST), `/documents` (GET)
- `lakebase.py` - Lakebase connection helper (same pattern as the day-2 template)
- `weather_client.py` - NWS API client: `get_active_alerts`, `get_forecast_periods`
- `setup_secrets.py` - One-time script to create the `database` secret scope and store the Lakebase URL
- `app.yaml` - Databricks App deployment config
- `templates/index.html` - Watchlist UI (add/remove locations, sync, view recent documents)
- `sql/01_setup_weather_documents_table.sql` - Raw document store DDL
- `sql/02_setup_weather_documents_embeddings_table.sql` - pgvector embeddings table DDL
- `notebooks/ingest_weather_embeddings.py` - Self-contained ETL: reads locations from `weather_watchlist`, fetches alerts/forecasts from NWS, upserts into `weather_documents`, computes embeddings into `weather_documents_embeddings`
- `databricks.yml` + `resources/ingest_weather_embeddings_job.yml` - Asset Bundle config that schedules the notebook above as an hourly Workflow
- `.env.example` - Local dev env var template

## Schema

```
weather_documents
  id              TEXT PRIMARY KEY   -- alert's own id, or a hash of location+startTime for forecast periods
  location        TEXT NOT NULL      -- watchlist label, e.g. "Sacramento, CA"
  source_type     TEXT NOT NULL      -- 'alert' | 'forecast'
  headline        TEXT NOT NULL      -- alert's headline/event, or "<period name>: <shortForecast>"
  narrative_text  TEXT               -- alert's description/instruction, or forecast's detailedForecast
  issued_at       TIMESTAMPTZ        -- alert's sent time, or forecast period's startTime
  effective_at    TIMESTAMPTZ        -- alert's effective time, or forecast period's endTime
  payload         JSONB NOT NULL     -- raw API response, for provenance
  synced_at       TIMESTAMPTZ NOT NULL DEFAULT now()

weather_documents_embeddings
  id           TEXT PRIMARY KEY      -- same id as weather_documents
  location     TEXT NOT NULL
  headline     TEXT NOT NULL
  issued_at    TIMESTAMPTZ
  embedding    VECTOR({{EMBEDDING_DIM}}) NOT NULL
  model_name   TEXT NOT NULL
  embedded_at  TIMESTAMPTZ NOT NULL DEFAULT now()

weather_watchlist                    -- created automatically by app.py, not a manual SQL file
  label      TEXT NOT NULL
  lat        NUMERIC NOT NULL
  lon        NUMERIC NOT NULL
  email      TEXT NOT NULL
  added_at   TIMESTAMPTZ NOT NULL DEFAULT now()
  PRIMARY KEY (lat, lon, email)
```

## Step-by-step setup

### 1. Create a Lakebase instance and a native-password role

Same as the day-2 template: **Catalog → Lakebase → Create Lakebase instance**,
then under **Roles & Databases**, enable password auth and create a role.
Copy the connection URL - you'll paste it into `setup_secrets.py` below.

### 2. Create the schema

Run these manually in the Lakebase SQL editor (or via `psql`):
1. `sql/01_setup_weather_documents_table.sql`
2. `sql/02_setup_weather_documents_embeddings_table.sql` - replace `{{EMBEDDING_DIM}}` with `384` (the default model's dimension)

`weather_watchlist` doesn't need a manual script - the Flask app creates it on first run.

### 3. Store your secret

From a Databricks notebook attached to any running cluster:

```python
%sh python setup_secrets.py
```

Prompts for your Lakebase connection URL and stores it as secret `database/lakebase-url`.
(If the scope was already created in a previous project, `setup_secrets.py`
skips scope creation gracefully - the `create_scope` line is commented out by
default and only needs uncommenting the very first time it's ever run in a
workspace.)

### 4. Grant the deployed app read access to the secret

In the Databricks Apps UI, when creating/configuring the app, use **App resources → Add resource → Secret**, scope `database`, key `lakebase-url`, permission **Can read**. This grants the app's service principal access without any manual ACL command.

### 5. Run locally (optional)

```bash
cp .env.example .env   # paste your Lakebase URL into LAKEBASE_URL
pip install -r requirements.txt
python app.py
```

Visit `http://localhost:8000`. The NWS API needs no key, so `/sync` works locally exactly as it will when deployed.

### 6. Deploy as a Databricks App

Same as the day-2 template: create a Git folder pointing at this repo (or link the GitHub repo directly in the Databricks Apps "Create new app" wizard's Git step), create a Custom app, attach the Secret resource from step 4, and deploy.

### 7. Add sample data and verify

1. Open the deployed app, add a location (e.g. lat `38.5816`, lon `-121.4944` for Sacramento, CA).
2. Click **Sync now** - pulls current alerts + forecast periods into `weather_documents`.
3. Confirm rows appear in the Lakebase SQL editor: `SELECT * FROM weather_documents ORDER BY synced_at DESC;`

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
| POST | `/sync` | fetch alerts + forecasts for every watchlisted location, upsert into `weather_documents` |
| GET | `/documents?location=&source_type=&limit=` | read synced documents |

## Notes

- No API key needed for weather data - `weather_client.py` only sets a
  descriptive `User-Agent`, which NWS requires but doesn't gate behind auth.
- `weather_documents_embeddings` has no chunking table equivalent (unlike
  `ticker_news_chunk_embeddings` in the day-2 template) - alert/forecast
  text is already short and self-contained, with no external article URL to
  fetch and chunk.
