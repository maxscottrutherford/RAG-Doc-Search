-- init.sql — runs once when the Postgres data directory is first created
-- (Docker: files in /docker-entrypoint-initdb.d/).
--
-- pgvector
-- --------
-- pgvector extends PostgreSQL with the `vector` type and operators for
-- approximate / exact nearest-neighbor search. You store one embedding per row
-- (e.g. one chunk of a document) and query with another embedding to retrieve
-- semantically similar rows.
--
-- What a vector column stores
-- ----------------------------
-- `vector(1536)` holds exactly 1536 floats: the output of an embedding model
-- for a piece of text. Those numbers encode "meaning" in a high-dimensional
-- space; similar texts land near each other, so distance in this space approximates
-- semantic similarity.
--
-- Why 1536 dimensions?
-- ----------------------
-- The dimension count is fixed by the embedding model, not by Postgres. 1536 is
-- the size returned by OpenAI's `text-embedding-ada-002` and a common setting
-- for `text-embedding-3-small`. Your application must use the same model (or
-- same output width) as this column; otherwise inserts/queries will error or
-- silently mismatch.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1536) NOT NULL,
    chunk_index INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
