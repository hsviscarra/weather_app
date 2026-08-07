"""
Weather Watchlist - a small Databricks App backed by Lakebase.

Users track locations (lat/lon) they care about, or sync ad-hoc place names
directly. Syncing pulls active alerts and forecast periods from the
National Weather Service (NWS) API and normalizes them into the
weather_documents table - the raw document store later turned into
chunked vector embeddings (weather_embeddings) by
notebooks/ingest_weather_embeddings.py. POST /weather/search embeds a query
at request time and runs a pgvector similarity search over those chunks.

Run locally:
    python app.py
Deploy as a Databricks App using app.yaml.
"""

import hashlib
import json
import logging
import os
import time

import psycopg2

from dotenv import load_dotenv

load_dotenv()

from flask import Flask, jsonify, render_template, request  # noqa: E402

import lakebase  # noqa: E402
from weather_client import WeatherClient  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("weather-app")

app = Flask(__name__)

WATCHLIST_TABLE_NAME = os.environ.get("WEATHER_WATCHLIST_TABLE_NAME", "weather_watchlist")
DOCUMENTS_TABLE_NAME = os.environ.get("WEATHER_DOCUMENTS_TABLE_NAME", "weather_documents")
EMBEDDINGS_TABLE_NAME = os.environ.get("WEATHER_EMBEDDINGS_TABLE_NAME", "weather_embeddings")

# Must match the model used by notebooks/ingest_weather_embeddings.py to
# write weather_embeddings - otherwise query vectors and stored vectors
# wouldn't be comparable. MODEL_CACHE_PATH points at the same Unity Catalog
# Volume the notebook caches the model into (requires a Volume resource
# attached to the app - see README). Only set via app.yaml on the deployed
# app - left unset locally, where sentence-transformers falls back to its
# own default cache dir since there's no real Volume mount outside
# Databricks.
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
MODEL_CACHE_PATH = os.environ.get("MODEL_CACHE_PATH") or None

# When pointed at a real Volume, force fully offline loading - serverless
# compute has no reliable internet egress to download from Hugging Face at
# runtime, so this must find the model already cached in the Volume (from
# an earlier run on compute that DOES have internet access) rather than
# attempting any network call, even a version-check ping.
if MODEL_CACHE_PATH:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


def _load_embedding_model():
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model %s from %s ...", EMBEDDING_MODEL_NAME, MODEL_CACHE_PATH)
    return SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder=MODEL_CACHE_PATH)


# Loaded once here at module level - not per-request - so /weather/search
# never pays model-load cost on the request path. A load failure here
# (e.g. the Volume resource isn't attached yet on a fresh deploy) is logged
# but doesn't crash the whole app; _get_embedding_model() retries lazily on
# the next call so a later successful redeploy self-heals without a code
# change.
try:
    _embedding_model = _load_embedding_model()
except Exception:
    logger.exception("Failed to load embedding model at startup - /weather/search will retry lazily")
    _embedding_model = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = _load_embedding_model()
    return _embedding_model


def ensure_watchlist_table():
    """Create the watchlist table in Lakebase if it doesn't exist yet."""
    lakebase.run_write(
        f"""
        CREATE TABLE IF NOT EXISTS {WATCHLIST_TABLE_NAME} (
            label TEXT NOT NULL,
            lat NUMERIC NOT NULL,
            lon NUMERIC NOT NULL,
            email TEXT NOT NULL,
            added_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (lat, lon, email)
        )
        """
    )


def ensure_documents_table():
    """Create weather_documents if it doesn't exist yet.

    Mirrors sql/01_setup_weather_documents_table.sql so the app is
    self-healing even if the schema wasn't created manually first.
    """
    lakebase.run_write(
        f"""
        CREATE TABLE IF NOT EXISTS {DOCUMENTS_TABLE_NAME} (
            id TEXT PRIMARY KEY,
            location TEXT NOT NULL,
            source_type TEXT NOT NULL CHECK (source_type IN ('alert', 'forecast')),
            headline TEXT NOT NULL,
            narrative_text TEXT,
            issued_at TIMESTAMPTZ,
            effective_at TIMESTAMPTZ,
            payload JSONB NOT NULL,
            synced_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    lakebase.run_write(
        f"CREATE INDEX IF NOT EXISTS idx_{DOCUMENTS_TABLE_NAME}_location ON {DOCUMENTS_TABLE_NAME} (location)"
    )


def _current_user_email() -> str:
    """Databricks Apps injects the logged-in user's identity via the
    X-Forwarded-Email header. Fall back to a generic value locally."""
    return request.headers.get("X-Forwarded-Email", "local-dev@example.com")


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.errorhandler(Exception)
def handle_exception(err):
    """Ensure all unhandled errors return JSON (not an HTML error page),
    so the frontend's resp.json() call never chokes on HTML."""
    logger.exception("Unhandled exception while processing request")
    status_code = getattr(err, "code", 500)
    if not isinstance(status_code, int):
        status_code = 500
    return jsonify({"error": str(err)}), status_code


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/watchlist", methods=["GET"])
def get_watchlist():
    """Return the current user's watched locations."""
    ensure_watchlist_table()
    email = _current_user_email()
    rows = lakebase.run_query(
        f"SELECT label, lat, lon, email, added_at FROM {WATCHLIST_TABLE_NAME} "
        f"WHERE email = %s ORDER BY label ASC",
        (email,),
    )
    return jsonify(rows)


@app.route("/watchlist", methods=["POST"])
def add_to_watchlist():
    """Add (or relabel) a lat/lon location on the current user's watchlist."""
    ensure_watchlist_table()
    body = request.json if request.is_json else {}

    try:
        lat = float(body.get("lat"))
        lon = float(body.get("lon"))
    except (TypeError, ValueError):
        return jsonify({"error": "lat and lon are required and must be numbers"}), 400

    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return jsonify({"error": "lat must be between -90 and 90, lon between -180 and 180"}), 400

    label = (body.get("label") or "").strip() or f"{lat:.4f},{lon:.4f}"
    email = _current_user_email()

    lakebase.run_write(
        f"""
        INSERT INTO {WATCHLIST_TABLE_NAME} (label, lat, lon, email, added_at)
        VALUES (%s, %s, %s, %s, now())
        ON CONFLICT (lat, lon, email) DO UPDATE SET label = EXCLUDED.label
        """,
        (label, lat, lon, email),
    )
    return jsonify({"label": label, "lat": lat, "lon": lon, "email": email})


@app.route("/watchlist", methods=["DELETE"])
def delete_from_watchlist():
    """Remove a lat/lon location from the current user's watchlist."""
    ensure_watchlist_table()
    body = request.json if request.is_json else {}

    try:
        lat = float(body.get("lat"))
        lon = float(body.get("lon"))
    except (TypeError, ValueError):
        return jsonify({"error": "lat and lon are required and must be numbers"}), 400

    email = _current_user_email()
    deleted = lakebase.run_write(
        f"DELETE FROM {WATCHLIST_TABLE_NAME} WHERE lat = %s AND lon = %s AND email = %s",
        (lat, lon, email),
    )
    if not deleted:
        return jsonify({"error": "Location not found on your watchlist"}), 404
    return jsonify({"lat": lat, "lon": lon, "deleted": True})


@app.route("/documents", methods=["GET"])
def list_documents():
    """Read synced weather documents, optionally filtered by location/source_type."""
    ensure_documents_table()

    location = request.args.get("location")
    source_type = request.args.get("source_type")
    if source_type and source_type not in ("alert", "forecast"):
        return jsonify({"error": "source_type must be 'alert' or 'forecast'"}), 400
    limit = int(request.args.get("limit", 100))

    sql = (
        f"SELECT id, location, source_type, headline, narrative_text, "
        f"issued_at, effective_at, synced_at FROM {DOCUMENTS_TABLE_NAME}"
    )
    conditions, params = [], []
    if location:
        conditions.append("location = %s")
        params.append(location)
    if source_type:
        conditions.append("source_type = %s")
        params.append(source_type)
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += " ORDER BY synced_at DESC LIMIT %s"
    params.append(limit)

    rows = lakebase.run_query(sql, params)
    return jsonify(rows)


@app.route("/weather/sync", methods=["POST"])
def sync_weather():
    """Pull active alerts + forecast periods and upsert them into
    weather_documents.

    Body (optional JSON): {"locations": ["Chicago, IL", "Austin, TX"], "limit": 50}
    - locations: place names, geocoded via Nominatim then synced. If
      omitted (or an empty list), falls back to the current deployment's
      weather_watchlist entries (already lat/lon, no geocoding needed) -
      this is what the UI's "Sync now" button calls, unchanged.
    - limit: max alerts AND max forecast periods to persist per location
      (applied independently to each source_type). Defaults to 50, clamped
      to [1, 200].
    """
    ensure_documents_table()
    body = request.json if request.is_json else {}

    try:
        limit = int(body.get("limit", 50))
    except (TypeError, ValueError):
        return jsonify({"error": "'limit' must be an integer"}), 400
    limit = max(1, min(limit, 200))

    raw_locations = body.get("locations")
    client = WeatherClient()

    if raw_locations:
        if not isinstance(raw_locations, list) or not all(isinstance(x, str) and x.strip() for x in raw_locations):
            return jsonify({"error": "'locations' must be a non-empty list of place-name strings"}), 400

        targets = []
        for i, place in enumerate(raw_locations):
            if i > 0:
                time.sleep(1)  # respect Nominatim's 1 req/sec usage policy
            try:
                lat, lon = client.geocode(place.strip())
                targets.append((place.strip(), lat, lon))
            except Exception as exc:
                logger.warning("Skipping %s: could not geocode (%s)", place, exc)
    else:
        ensure_watchlist_table()
        watchlist_rows = lakebase.run_query(f"SELECT DISTINCT label, lat, lon FROM {WATCHLIST_TABLE_NAME}")
        if not watchlist_rows:
            return jsonify({"synced": 0, "locations": 0, "message": "No locations given and watchlist is empty"})
        targets = [(row["label"], float(row["lat"]), float(row["lon"])) for row in watchlist_rows]

    total = 0
    synced_locations = 0
    for label, lat, lon in targets:
        try:
            total += _sync_alerts(client, label, lat, lon, limit)
            total += _sync_forecast(client, label, lat, lon, limit)
            synced_locations += 1
        except Exception as exc:
            logger.warning("Skipping %s: %s", label, exc)
            continue

    return jsonify({"synced": total, "locations": synced_locations})


def _sync_alerts(client: WeatherClient, label: str, lat: float, lon: float, limit: int) -> int:
    """Upsert active alerts for one location (up to `limit`). Mirrors
    _upsert_batch in the day-2 template's app.py."""
    features = client.get_active_alerts(lat, lon)[:limit]
    count = 0
    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            for feature in features:
                props = feature.get("properties", {})
                cur.execute(
                    f"""
                    INSERT INTO {DOCUMENTS_TABLE_NAME}
                        (id, location, source_type, headline, narrative_text, issued_at, effective_at, payload, synced_at)
                    VALUES (%s, %s, 'alert', %s, %s, %s, %s, %s, now())
                    ON CONFLICT (id) DO UPDATE
                        SET headline = EXCLUDED.headline,
                            narrative_text = EXCLUDED.narrative_text,
                            issued_at = EXCLUDED.issued_at,
                            effective_at = EXCLUDED.effective_at,
                            payload = EXCLUDED.payload,
                            synced_at = EXCLUDED.synced_at
                    """,
                    (
                        props.get("id"),
                        label,
                        props.get("headline") or props.get("event", ""),
                        props.get("description") or props.get("instruction"),
                        props.get("sent"),
                        props.get("effective"),
                        json.dumps(feature),
                    ),
                )
                count += 1
            conn.commit()
    return count


def _sync_forecast(client: WeatherClient, label: str, lat: float, lon: float, limit: int) -> int:
    """Upsert forecast periods for one location (up to `limit`). Each period
    has no natural id from the API, so _forecast_doc_id() hashes location +
    start time into a stable dedup key."""
    periods = client.get_forecast_periods(lat, lon)[:limit]
    count = 0
    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            for period in periods:
                name = period.get("name", "")
                short = period.get("shortForecast", "")
                cur.execute(
                    f"""
                    INSERT INTO {DOCUMENTS_TABLE_NAME}
                        (id, location, source_type, headline, narrative_text, issued_at, effective_at, payload, synced_at)
                    VALUES (%s, %s, 'forecast', %s, %s, %s, %s, %s, now())
                    ON CONFLICT (id) DO UPDATE
                        SET headline = EXCLUDED.headline,
                            narrative_text = EXCLUDED.narrative_text,
                            issued_at = EXCLUDED.issued_at,
                            effective_at = EXCLUDED.effective_at,
                            payload = EXCLUDED.payload,
                            synced_at = EXCLUDED.synced_at
                    """,
                    (
                        _forecast_doc_id(lat, lon, period),
                        label,
                        f"{name}: {short}" if short else name,
                        period.get("detailedForecast"),
                        period.get("startTime"),
                        period.get("endTime"),
                        json.dumps(period),
                    ),
                )
                count += 1
            conn.commit()
    return count


def _forecast_doc_id(lat: float, lon: float, period: dict) -> str:
    """Stable dedup key for a forecast period: no natural id field exists in
    the NWS API response, so hash location + the period's start time and
    name (unique per location per forecast window)."""
    raw = f"{lat:.4f},{lon:.4f}:{period.get('startTime', '')}:{period.get('name', '')}"
    return "forecast:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


@app.route("/weather/search", methods=["POST"])
def search_weather():
    """Semantic search over weather_embeddings.

    Body: {"query": "risk of flooding near rivers", "top_k": 5}

    Embeds `query` with the same model notebooks/ingest_weather_embeddings.py
    used to write weather_embeddings, then runs a pgvector cosine-distance
    nearest-neighbor search, joined back to weather_documents for context.
    """
    body = request.json if request.is_json else {}

    query = body.get("query")
    if not isinstance(query, str) or not query.strip():
        return jsonify({"error": "'query' is required and must be a non-empty string"}), 400
    query = query.strip()

    try:
        top_k = int(body.get("top_k", 5))
    except (TypeError, ValueError):
        return jsonify({"error": "'top_k' must be an integer"}), 400
    top_k = max(1, min(top_k, 20))  # clamp, don't reject - see README edge cases

    try:
        query_embedding = _get_embedding_model().encode(query).tolist()
    except Exception:
        logger.exception("Failed to load embedding model or encode query")
        return jsonify({"error": "Search is temporarily unavailable (embedding model failed to load)"}), 503

    # pgvector has no native psycopg2 adapter - passing the vector as a
    # bracketed literal string and casting with ::vector is the standard
    # workaround. It's still a bound parameter (not string-interpolated
    # into the query), so this isn't a SQL injection risk.
    embedding_literal = "[" + ",".join(f"{x:.8f}" for x in query_embedding) + "]"

    try:
        rows = lakebase.run_query(
            f"""
            SELECT
                d.id, d.location, d.headline, d.narrative_text, d.source_type, d.issued_at,
                e.chunk_text, e.chunk_index, e.model_name,
                1 - (e.embedding <=> %s::vector) AS similarity
            FROM {EMBEDDINGS_TABLE_NAME} e
            JOIN {DOCUMENTS_TABLE_NAME} d ON d.id = e.document_id
            ORDER BY e.embedding <=> %s::vector
            LIMIT %s
            """,
            (embedding_literal, embedding_literal, top_k),
        )
    except psycopg2.errors.UndefinedTable:
        return jsonify(
            {
                "results": [],
                "query": query,
                "message": (
                    f"{EMBEDDINGS_TABLE_NAME} doesn't exist yet - run "
                    "sql/02_setup_weather_embeddings_table.sql, then the "
                    "ingest_weather_embeddings notebook."
                ),
            }
        )

    if not rows:
        return jsonify({"results": [], "query": query, "message": "No embeddings synced yet."})

    return jsonify({"results": rows, "query": query, "top_k": top_k})


if __name__ == "__main__":
    host = os.getenv("FLASK_RUN_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_RUN_PORT", 8000))
    app.run(debug=True, host=host, port=port)
