-- Setup script for weather_embeddings table
-- Run this manually in your Lakebase Postgres database before running the
-- notebook or using POST /weather/search.
--
-- One row per CHUNK of a document's narrative_text (usually just one chunk
-- for a short forecast period, possibly several for a long alert
-- description) - not one row per document. This mirrors the day-2
-- template's ticker_news_chunk_embeddings table shape.

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 384 matches sentence-transformers/all-MiniLM-L6-v2 (the default model in
-- notebooks/ingest_weather_embeddings.py and app.py). If you switch models,
-- update this dimension to match.
CREATE TABLE IF NOT EXISTS weather_embeddings (
    id           TEXT PRIMARY KEY,
    document_id  TEXT NOT NULL REFERENCES weather_documents(id) ON DELETE CASCADE,
    chunk_index  INT NOT NULL,
    chunk_text   TEXT NOT NULL,
    embedding    VECTOR(384) NOT NULL,
    model_name   TEXT NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Create HNSW index for fast cosine similarity search
CREATE INDEX IF NOT EXISTS idx_weather_embeddings_embedding
ON weather_embeddings
USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_weather_embeddings_document_id
ON weather_embeddings (document_id);

-- Verify the table was created
SELECT
    table_name,
    column_name,
    data_type,
    udt_name
FROM information_schema.columns
WHERE table_name = 'weather_embeddings'
ORDER BY ordinal_position;
