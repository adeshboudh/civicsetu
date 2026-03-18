# CivicSetu — Low Level Design (LLD)

**Version:** 0.3.0 — Phase 3 Complete
**Last Updated:** March 2026

---

## 1. Module Map

```

src/civicsetu/
├── config/
│   ├── settings.py           Pydantic BaseSettings singleton (lru_cache)
│   └── document_registry.py  All document URLs + metadata (single source of truth)
├── models/
│   ├── enums.py              StrEnum: Jurisdiction, DocType, QueryType, etc.
│   └── schemas.py            Pydantic models: LegalChunk, Citation, RetrievedChunk, CivicSetuResponse
├── ingestion/
│   ├── downloader.py         httpx PDF downloader with MD5 cache check
│   ├── parser.py             PyMuPDF text extractor, scanned PDF detection
│   ├── chunker.py            Section-boundary regex chunker + fallback
│   ├── metadata_extractor.py Date/Section/Rule reference/amendment regex extraction
│   ├── embedder.py           nomic-embed-text via Ollama (document + query prefixes)
│   ├── pipeline.py           Orchestrates all ingestion steps end-to-end
│   └── graph_seeder.py       Post-ingestion REFERENCES + DERIVED_FROM edge seeding
├── stores/
│   ├── relational_store.py   Async SQLAlchemy — documents + legal_chunks tables
│   ├── vector_store.py       pgvector HNSW cosine search
│   └── graph_store.py        Neo4j Cypher interface — fresh driver per call
├── retrieval/
│   ├── vector_retriever.py   Wraps VectorStore for agent use
│   ├── graph_retriever.py    REFERENCES + DERIVED_FROM traversal, Section/Rule ID extraction
│   └── reranker.py           FlashRank cross-encoder wrapper
├── agent/
│   ├── state.py              CivicSetuState TypedDict (frozen contract)
│   ├── nodes.py              Pure functions: classifier, retrieval, reranker,
│   │                         generator, validator
│   ├── edges.py              Conditional routing: route_after_classifier,
│   │                         route_after_validator
│   └── graph.py              StateGraph assembly + get_compiled_graph()
├── prompts/
│   ├── classifier.py         Query type classification + rewriting prompt
│   ├── generator.py          Cited answer generation prompt
│   └── validator.py          Hallucination + confidence check prompt
├── guardrails/
│   ├── input_guard.py        PII detection + off-topic filter
│   └── output_guard.py       Faithfulness check + disclaimer injection
└── api/
    ├── main.py               FastAPI app factory + lifespan (graph pre-compiled)
    ├── routes/
    │   ├── health.py         GET /health — DB ping
    │   ├── query.py          POST /api/v1/query — main RAG endpoint
    │   └── ingest.py         POST /api/v1/ingest — admin endpoint
    └── middleware/
        └── logging.py        Request/response structured logging

```

---

## 2. Database Schema

### PostgreSQL Tables

```sql
documents (
    doc_id          UUID PRIMARY KEY,
    doc_name        TEXT,
    jurisdiction    TEXT,   -- Jurisdiction enum value
    doc_type        TEXT,   -- DocType enum value  (stored uppercase: ACT, RULES)
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
    section_id          TEXT,   -- "18", "3(2)", "Para-3"
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
  (:Section  {section_id, title, chunk_id, jurisdiction, doc_name, is_active})

Edges:
  (:Document)-[:HAS_SECTION]->(:Section)
  (:Section) -[:REFERENCES]->(:Section)       -- intra + cross-jurisdiction citations
  (:Section) -[:DERIVED_FROM]->(:Section)     -- MahaRERA Rule N → RERA Act Sec M
  (:Document)-[:DERIVED_FROM]->(:Document)    -- MahaRERA Rules → RERA Act 2016

Planned (Phase 4+):
  (:Section) -[:SUPERSEDES]->(:Section)
  (:Section) -[:AMENDED_BY]->(:Amendment)
  (:Section) -[:CONFLICTS_WITH]->(:Section)
```

**Live graph stats (Phase 3):**

| Metric        | Count |
|---------------|-------|
| Documents     | 2     |
| Sections      | 438   |
| HAS_SECTION   | 438   |
| REFERENCES    | 171   |
| DERIVED_FROM  | 18    |

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

### RetrievedChunk Schema (`models/schemas.py`)

```python
class RetrievedChunk(BaseModel):
    chunk: LegalChunk
    vector_score: float | None = None
    rerank_score: float | None = None
    retrieval_source: str = "vector"   # "vector" | "graph"
    graph_path: Optional[str] = None   # e.g. "source:18@CENTRAL"
    is_pinned: bool = False            # True = exact source section, bypasses reranker sort
```


### Node Responsibilities

| Node | Input Keys | Output Keys | LLM Call |
| :-- | :-- | :-- | :-- |
| classifier | query | query_type, rewritten_query | Yes |
| vector_retrieval | rewritten_query, top_k | retrieved_chunks | No |
| graph_retrieval | rewritten_query, top_k | retrieved_chunks | No |
| reranker | retrieved_chunks, query | reranked_chunks | No |
| generator | reranked_chunks, query | raw_response, citations, confidence_score | Yes |
| validator | raw_response, reranked_chunks | hallucination_flag, confidence_score | Yes |
| retry | retry_count | retry_count+1, cleared retrieval fields | No |

### Routing Logic

| classifier → route_after_classifier   |                                          |
|---------------------------------------|------------------------------------------|
| fact_lookup                           | vector_retrieval                         |
| cross_reference                       | graph_retrieval (→ vector fallback)      |
| penalty_lookup                        | graph_retrieval (→ vector fallback)      |
| temporal                              | graph_retrieval (→ vector fallback)      |
| conflict_detection                    | hybrid_retrieval (Phase 4)               |

```
validator → route_after_validator:
    confidence >= 0.5 AND not hallucinated → END
    (confidence < 0.5 OR hallucinated) AND retry_count < 2 → retry → classifier
    (confidence < 0.5 OR hallucinated) AND retry_count >= 2 → END (low confidence answer)
```


---

## 4. Chunking Strategy

### Section Boundary Detection

Two regex patterns to cover both document formats ingested:

**Act format** (RERA Act 2016):

```
^\s*(?P<id>\d+[A-Z]?)\.?\s*(?P<title>[A-Z][^\n—]{3,80})\.?—
```

Matches: `18. Return of amount and compensation.—`

**Rule format** (MahaRERA Rules 2017):

```
\n(?P<id>\d+)\.\s*\n(?P<title>[A-Z][^\n]{3,80})\n
```

Matches: `\n3.\nInformation to be furnished...\n`

Chunker tries Act pattern first; falls back to Rule pattern; falls back to paragraph
split if neither matches. Logs `chunking_fallback_used` on paragraph path.

### Chunk Size Limits

```
MIN_CHARS = 100   — discard fragments (headers, page numbers)
MAX_CHARS = 1500  — split large sections at subsection markers (1), (2), (a), (b)
```

### Split Priority for Large Sections

```
1. Subsection markers: \n\s*\((?:\d+|[a-z])\)\s+
2. Sentence boundary near MAX_CHARS: rfind('. ')
3. Hard cut at MAX_CHARS (last resort)
```

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

### Truncation Guard

```python
MAX_EMBED_CHARS = 6000   # ~1500 tokens for nomic-embed-text
if len(text) > MAX_EMBED_CHARS:
    log.warning("embedding_truncated", original_len=len(text), truncated_to=MAX_EMBED_CHARS)
    text = text[:MAX_EMBED_CHARS]
```

Prevents silent API errors on oversized chunks. Expected to fire on 0–2 chunks per
document where subsection splitting fails (complex tables, long definition lists).

---

## 6. Graph Retriever — Phase 3 Logic

`graph_retriever.py` — called on `cross_reference`, `penalty_lookup`, `temporal` query types.

### Section ID Extraction

```python
# Matches both "Section 18" / "Sec 18" and "Rule 3" / "Rule 12A"
section_pattern = re.compile(r'\b(?:section|sec\.?|s\.)\s*(\d+[A-Z]?)\b', re.IGNORECASE)
rule_pattern    = re.compile(r'\bRule\s+(\d+[A-Z]?)\b', re.IGNORECASE)
```

Section pattern takes priority; Rule pattern is fallback.

### Traversal Strategy (per jurisdiction)

For each jurisdiction (`CENTRAL`, `MAHARASHTRA`):

```
1. Source section chunks    — exact section_id match → is_pinned=True
2. REFERENCES outgoing      — sections source cites (depth=2)
3. REFERENCES incoming      — sections that cite source
4. DERIVED_FROM outgoing    — Act sections this Rule derives from
5. DERIVED_FROM incoming    — Rule sections implementing this Act section
```

### Pinning Rule

Only the exact `section_id` match gets `is_pinned=True`. Sub-sections like `3(2)`, `3(5)`
are NOT pinned — they compete normally in the reranker.
Max pinned chunks: 2 (one per jurisdiction). Remaining 3 slots filled by reranker.

---

## 7. Response Contract

Every response from `POST /api/v1/query` conforms to:

```python
CivicSetuResponse:
    answer: str                    # plain English, cites section numbers
    citations: list[Citation]      # min_length=1 — NEVER empty
    confidence_score: float        # 0.0–1.0
    confidence_level: str          # computed: "high"/"medium"/"low"
    query_type_resolved: QueryType
    conflict_warnings: list[str]   # empty until Phase 4
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

## 8. Error Handling

| Scenario | Behaviour |
| :-- | :-- |
| LLM provider rate limited | LiteLLM auto-rotates to next provider |
| All LLM providers fail | `RuntimeError` → FastAPI 500 |
| No chunks retrieved | `InsufficientInfoResponse` returned |
| Hallucination detected | retry (max 2x) → low confidence answer |
| DB unreachable | `/health` returns `degraded`, query returns 500 |
| Scanned PDF detected | Warning logged, text extracted as-is |
| Section patterns not matched | Fallback paragraph chunking, warning logged |
| Neo4j event loop mismatch | Prevented — `_get_driver()` creates fresh driver per call |

## 9. Neo4j Graph — Phase 3 State

**Nodes:** 2 Documents, 438 Sections
**Edges:** 438 HAS_SECTION, 171 REFERENCES, 18 DERIVED_FROM

### Documents in graph

| Document | Jurisdiction | Sections | REFERENCES | DERIVED_FROM |
|---|---|---|---|---|
| RERA Act 2016 | CENTRAL | 224 | 110 outgoing | — |
| MahaRERA Rules 2017 | MAHARASHTRA | 214 | 61 intra + 0 cross-outgoing | 17 section + 1 doc |

### DERIVED_FROM section map (`_SECTION_DERIVED_FROM_MAP`)

17 hand-mapped MahaRERA Rule → RERA Act section pairs:

| MahaRERA Rule | RERA Act Section | Subject |
|---|---|---|
| 3 | 4 | Promoter registration info |
| 4 | 4 | Promoter declaration |
| 5 | 4 | Separate account |
| 6 | 5 | Grant/rejection of project registration |
| 7 | 6 | Extension of registration |
| 8 | 7 | Revocation of registration |
| 10 | 11 | Agreement for Sale |
| 11 | 9 | Agent registration application |
| 12 | 9 | Agent grant/rejection |
| 14 | 10 | Obligations of real estate agents |
| 15 | 9 | Revocation of agent registration |
| 17 | 10 | Other functions of agent |
| 18 | 19 | Rate of interest |
| 19 | 18 | Timelines for refund |
| 20 | 11 | Website disclosures (projects) |
| 21 | 11 | Website disclosures (agents) |
| 31 | 31 | Governing law / complaints |

### MetadataExtractor — reference patterns

```python
# Section references: "Section 18", "section 4A", "s. 9", "sec 18"
extract_section_references(text) → list[str]

# Rule references (Phase 3): "Rule 3", "Rule 12A", "under Rule 9"
extract_rule_references(text) → list[str]
```

Cross-act scrub removes false positives from IPC, CPC, Companies Act, etc.
RERA section bounds filter: sections 1–92 only.

### GraphStore — driver lifecycle

`_get_driver()` creates a **fresh** `AsyncDriver` on every call — no singleton, no
`@lru_cache`. Every method wraps its session in `try/finally: await driver.close()`.
This prevents `RuntimeError: Future attached to a different loop` when `asyncio.run()`
is called multiple times in the same process (LangGraph sync `.invoke()` pattern).
