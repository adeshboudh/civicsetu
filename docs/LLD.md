# CivicSetu — Low Level Design (LLD)

**Version:** 0.1.0 (Phase 0)
**Last Updated:** March 2026

---

## 1. Module Map

```

src/civicsetu/
├── config/
│   └── settings.py         Pydantic BaseSettings singleton (lru_cache)
├── models/
│   ├── enums.py             StrEnum: Jurisdiction, DocType, QueryType, etc.
│   └── schemas.py           Pydantic models: LegalChunk, Citation, CivicSetuResponse
├── ingestion/
│   ├── downloader.py        httpx PDF downloader with MD5 cache check
│   ├── parser.py            PyMuPDF text extractor, scanned PDF detection
│   ├── chunker.py           Section-boundary regex chunker + fallback
│   ├── metadata_extractor.py Date/reference/amendment regex extraction
│   ├── embedder.py          nomic-embed-text via Ollama (document + query prefixes)
│   └── pipeline.py          Orchestrates all ingestion steps end-to-end
├── stores/
│   ├── relational_store.py  Async SQLAlchemy — documents + legal_chunks tables
│   ├── vector_store.py      pgvector HNSW cosine search
│   └── graph_store.py       Neo4j Cypher interface (Phase 1)
├── retrieval/
│   ├── vector_retriever.py  Wraps VectorStore for agent use
│   ├── graph_retriever.py   Cypher query builder (Phase 1)
│   └── reranker.py          FlashRank cross-encoder wrapper
├── agent/
│   ├── state.py             CivicSetuState TypedDict (frozen contract)
│   ├── nodes.py             Pure functions: classifier, retrieval, reranker,
│   │                        generator, validator
│   ├── edges.py             Conditional routing: route_after_classifier,
│   │                        route_after_validator
│   └── graph.py             StateGraph assembly + get_compiled_graph()
├── prompts/
│   ├── classifier.py        Query type classification + rewriting prompt
│   ├── generator.py         Cited answer generation prompt
│   └── validator.py         Hallucination + confidence check prompt
├── guardrails/
│   ├── input_guard.py       PII detection + off-topic filter (Phase 1)
│   └── output_guard.py      Faithfulness check + disclaimer injection (Phase 1)
└── api/
├── main.py              FastAPI app factory + lifespan (graph pre-compiled)
├── routes/
│   ├── health.py        GET /health — DB ping
│   ├── query.py         POST /api/v1/query — main RAG endpoint
│   └── ingest.py        POST /api/v1/ingest — Phase 1 admin endpoint
└── middleware/
└── logging.py       Request/response structured logging

```

---

## 2. Database Schema

### PostgreSQL Tables

```sql
documents (
    doc_id          UUID PRIMARY KEY,
    doc_name        TEXT,
    jurisdiction    TEXT,   -- Jurisdiction enum value
    doc_type        TEXT,   -- DocType enum value
    source_url      TEXT,
    effective_date  DATE,
    gazette_number  TEXT,
    total_chunks    INTEGER,
    ingested_at     TIMESTAMPTZ,
    is_active       BOOLEAN
)

legal_chunks (
    chunk_id            UUID PRIMARY KEY,
    doc_id              UUID → documents.doc_id,
    jurisdiction        TEXT,
    doc_type            TEXT,
    doc_name            TEXT,
    section_id          TEXT,   -- "18", "Rule 12", "Para-3"
    section_title       TEXT,
    section_hierarchy   TEXT[], -- ["RERA Act 2016", "18"]
    text                TEXT,
    effective_date      DATE,
    superseded_by       UUID → legal_chunks.chunk_id,
    status              TEXT,   -- ChunkStatus enum value
    source_url          TEXT,
    page_number         INTEGER,
    embedding           vector(768)  -- HNSW indexed
)
```


### pgvector Index

```sql
CREATE INDEX legal_chunks_embedding_idx
    ON legal_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
```

`m=16` — 16 connections per node. `ef_construction=64` — 64 candidates during index build.
Tuned for recall/speed balance at <10K vectors. Revisit at 100K+.

### Neo4j Graph Schema (Phase 1)

```
Nodes:
  (:Document {doc_id, doc_name, jurisdiction, doc_type, effective_date})
  (:Section  {section_id, title, chunk_id, jurisdiction, is_active})
  (:Amendment{amendment_id, circular_number, issued_date, effective_date})
  (:Concept  {label})  -- "Promoter", "Allottee", "Carpet Area"
  (:Penalty  {section_ref, amount_or_formula, trigger_condition})

Edges:
  (:Document)-[:HAS_SECTION]->(:Section)
  (:Section) -[:REFERENCES]->(:Section)
  (:Section) -[:SUPERSEDES]->(:Section)
  (:Section) -[:AMENDED_BY]->(:Amendment)
  (:Amendment)-[:MODIFIES]->(:Section)
  (:Section) -[:DEFINES]->(:Concept)
  (:Section) -[:IMPOSES]->(:Penalty)
  (:Section) -[:CONFLICTS_WITH]->(:Section)   -- Phase 3
  (:Document)-[:DERIVED_FROM]->(:Document)    -- MahaRERA ← Central RERA
```


---

## 3. LangGraph State Machine

### State Contract (`agent/state.py`)

```python
class CivicSetuState(TypedDict):
    # Input
    query: str
    session_id: Optional[str]
    jurisdiction_filter: Optional[Jurisdiction]
    top_k: int

    # Classification
    query_type: Optional[QueryType]
    rewritten_query: Optional[str]

    # Retrieval — Annotated[list, operator.add] enables parallel node merging
    retrieved_chunks: Annotated[list[RetrievedChunk], operator.add]
    reranked_chunks: list[RetrievedChunk]

    # Generation
    raw_response: Optional[str]
    citations: list[Citation]
    confidence_score: float
    conflict_warnings: list[str]
    amendment_notice: Optional[str]

    # Control
    retry_count: int          # max 2 retries
    hallucination_flag: bool
    error: Optional[str]
```


### Node Responsibilities

| Node | Input Keys | Output Keys | LLM Call |
| :-- | :-- | :-- | :-- |
| classifier | query | query_type, rewritten_query | Yes |
| vector_retrieval | rewritten_query, top_k | retrieved_chunks | No |
| reranker | retrieved_chunks, query | reranked_chunks | No |
| generator | reranked_chunks, query | raw_response, citations, confidence_score | Yes |
| validator | raw_response, reranked_chunks | hallucination_flag, confidence_score | Yes |
| retry | retry_count | retry_count+1, cleared retrieval fields | No |

### Routing Logic

```
classifier → route_after_classifier:
    fact_lookup, penalty_lookup    → vector_retrieval
    cross_reference, temporal      → graph_retrieval (Phase 1) / vector_retrieval (Phase 0)
    conflict_detection             → hybrid_retrieval (Phase 3) / vector_retrieval (Phase 0)

validator → route_after_validator:
    confidence >= 0.5 AND not hallucinated → END
    (confidence < 0.5 OR hallucinated) AND retry_count < 2 → retry → classifier
    (confidence < 0.5 OR hallucinated) AND retry_count >= 2 → END (low confidence answer)
```


---

## 4. Chunking Strategy

### Section Boundary Detection

Indian legal acts follow a consistent numbering format:

```
```

^\s*(?P<id>\d+[A-Z]?)\.\s+(?P<title>[A-Za-z][^—\n]{3,80})\.?—

```
```

Matches: `1. Short title.—` / `18A. Special provisions.—` / ` 2. Definitions.—`

### Chunk Size Limits

```
MIN_CHARS = 100   — discard fragments (headers, page numbers)
MAX_CHARS = 2000  — split large sections at subsection markers (1), (2), (a), (b)
```


### Split Priority for Large Sections

```
1. Subsection markers: \n\s*\((?:\d+|[a-z])\)\s+
2. Sentence boundary near MAX_CHARS: rfind('. ')
3. Hard cut at MAX_CHARS (last resort)
```


### Fallback Strategy

If no section boundaries found (non-standard docs):
→ Paragraph chunking on `\n\n` separator
→ Logs `WARNING: fallback_paragraph_chunking`
→ section_id becomes `Para-1`, `Para-2`, etc.

---

## 5. Embedding Strategy

**Model:** `nomic-embed-text` (Ollama local)
**Dimension:** 768
**Asymmetric prefixes:**

```
Ingestion time:  "search_document: {text}"   → embed_document()
Query time:      "search_query: {query}"     → embed_query()
```

Using wrong prefix at query time causes ~10–15% recall degradation.
The `embed_document()` / `embed_query()` method split enforces this at the API level.

---

## 6. Response Contract

Every response from `POST /api/v1/query` conforms to:

```python
CivicSetuResponse:
    answer: str                    # plain English, cites section numbers
    citations: list[Citation]      # min_length=1 — NEVER empty
    confidence_score: float        # 0.0–1.0
    confidence_level: str          # computed: "high"/"medium"/"low"
    query_type_resolved: QueryType
    conflict_warnings: list[str]   # empty in Phase 0
    amendment_notice: Optional[str]
    disclaimer: str                # always present

Citation:
    section_id: str
    doc_name: str
    jurisdiction: Jurisdiction
    effective_date: Optional[date]
    source_url: str
    chunk_id: UUID                 # traceable to exact DB row
```

If `citations` would be empty → return `InsufficientInfoResponse` instead.

---

## 7. Error Handling

| Scenario | Behaviour |
| :-- | :-- |
| LLM provider rate limited | LiteLLM auto-rotates to next provider |
| All LLM providers fail | `RuntimeError` → FastAPI 500 |
| No chunks retrieved | `InsufficientInfoResponse` returned |
| Hallucination detected | retry (max 2x) → low confidence answer |
| DB unreachable | `/health` returns `degraded`, query returns 500 |
| Scanned PDF detected | Warning logged, text extracted as-is, Surya OCR Phase 2 |
| Section patterns not matched | Fallback paragraph chunking, warning logged |
