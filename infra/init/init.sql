-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- ── Documents table ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS documents (
    doc_id          UUID PRIMARY KEY,
    doc_name        TEXT NOT NULL,
    jurisdiction    TEXT NOT NULL,
    doc_type        TEXT NOT NULL,
    source_url      TEXT NOT NULL,
    effective_date  DATE,
    gazette_number  TEXT,
    total_chunks    INTEGER DEFAULT 0,
    ingested_at     TIMESTAMPTZ DEFAULT NOW(),
    is_active       BOOLEAN DEFAULT TRUE
);

-- ── Legal chunks table ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS legal_chunks (
    chunk_id            UUID PRIMARY KEY,
    doc_id              UUID REFERENCES documents(doc_id) ON DELETE CASCADE,
    jurisdiction        TEXT NOT NULL,
    doc_type            TEXT NOT NULL,
    doc_name            TEXT NOT NULL,
    section_id          TEXT NOT NULL,
    section_title       TEXT NOT NULL,
    section_hierarchy   TEXT[] NOT NULL DEFAULT '{}',
    text                TEXT NOT NULL,
    effective_date      DATE,
    superseded_by       UUID REFERENCES legal_chunks(chunk_id),
    status              TEXT NOT NULL DEFAULT 'active',
    source_url          TEXT NOT NULL,
    page_number         INTEGER NOT NULL,
    embedding           vector(768),          -- matches EMBEDDING_DIMENSION in .env
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ── Indexes ────────────────────────────────────────────────────────────────────
-- HNSW index for fast approximate nearest neighbor search
CREATE INDEX IF NOT EXISTS legal_chunks_embedding_idx
    ON legal_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Filter indexes for metadata queries
CREATE INDEX IF NOT EXISTS legal_chunks_jurisdiction_idx
    ON legal_chunks (jurisdiction);

CREATE INDEX IF NOT EXISTS legal_chunks_doc_type_idx
    ON legal_chunks (doc_type);

CREATE INDEX IF NOT EXISTS legal_chunks_section_id_idx
    ON legal_chunks (section_id);

CREATE INDEX IF NOT EXISTS legal_chunks_status_idx
    ON legal_chunks (status);
