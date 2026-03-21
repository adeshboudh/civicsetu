# CivicSetu — Low Level Design (LLD)

**Version:** 0.5.0 — Phase 6 Complete (4-State Ingestion)
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
│   ├── parser.py             PyMuPDF text extractor — max_pages cap, scanned PDF detection
│   ├── chunker.py            Section-boundary regex chunker — 6 format patterns + fallback
│   ├── metadata_extractor.py Date/Section/Rule reference/amendment regex extraction
│   ├── embedder.py           nomic-embed-text via Ollama — truncate at 4000 chars pre-prefix
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
    doc_type        TEXT,   -- DocType enum value  (stored uppercase: ACT, RULES, CIRCULAR)
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

### Neo4j Graph Schema

```
Nodes:
  (:Document {doc_id, doc_name, jurisdiction, doc_type, effective_date})
  (:Section  {section_id, title, chunk_id, jurisdiction, doc_name, is_active})

Edges:
  (:Document)-[:HAS_SECTION]->(:Section)
  (:Section) -[:REFERENCES]->(:Section)       -- intra + cross-jurisdiction citations
  (:Section) -[:DERIVED_FROM]->(:Section)     -- State Rule N → RERA Act Sec M
  (:Document)-[:DERIVED_FROM]->(:Document)    -- State Rules → RERA Act 2016

Planned (Phase 7+):
  (:Section) -[:SUPERSEDES]->(:Section)
  (:Section) -[:AMENDED_BY]->(:Amendment)
  (:Section) -[:CONFLICTS_WITH]->(:Section)
```

**Live graph stats (Phase 6):**

| Metric       | Count |
|--------------|-------|
| Documents    | 9     |
| Sections     | 2090  |
| HAS_SECTION  | 1297  |
| REFERENCES   | 933   |
| DERIVED_FROM | 91    |

---

## 3. Document Registry

`document_registry.py` — single source of truth for all ingested documents.

```python
@dataclass(frozen=True)
class DocumentSpec:
    name: str
    url: str
    jurisdiction: Jurisdiction
    doc_type: DocType
    effective_date: date | None
    filename: str
    dest_subdir: str
    max_pages: int | None = None  # None = all pages; cap excludes forms/schedules appendices
```

### Ingested Documents (Phase 6)

| Key | Document | Jurisdiction | DocType | Chunks | max_pages |
|---|---|---|---|---|---|
| `rera_act_2016` | RERA Act 2016 | CENTRAL | ACT | ~224 | None |
| `mahrera_rules_2017` | MahaRERA Rules 2017 | MAHARASHTRA | RULES | ~214 | None |
| `up_rera_rules_2016` | UP RERA Rules 2016 | UTTAR_PRADESH | RULES | 170 | 24 |
| `up_rera_general_regulations_2019` | UP RERA General Regulations 2019 | UTTAR_PRADESH | CIRCULAR | 85 | None |
| `karnataka_rera_rules_2017` | Karnataka RERA Rules 2017 | KARNATAKA | RULES | 235 | 37 |
| `tn_rera_rules_2017` | Tamil Nadu RERA Rules 2017 | TAMIL_NADU | RULES | 157 | 15 |

**PDF source notes:**
- Karnataka official PDF (`rera.karnataka.gov.in`) is fully scanned (19MB image) — NAREDCO mirror used
- TN PDF bundles rules + forms (101 pages); `max_pages=15` excludes Forms A–O
- UP Rules PDF bundles rules + forms (52 pages); `max_pages=24` excludes prescribed forms

---

## 4. LangGraph State Machine

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
| conflict_detection                    | hybrid_retrieval (Phase 7)               |

```
validator → route_after_validator:
    confidence >= 0.5 AND not hallucinated → END
    (confidence < 0.5 OR hallucinated) AND retry_count < 2 → retry → classifier
    (confidence < 0.5 OR hallucinated) AND retry_count >= 2 → END (low confidence answer)
```

---

## 5. Chunking Strategy

### Section Boundary Detection

Six regex patterns across `DocType.RULES`, tried in order (first match wins per line):

| # | Pattern | Format | Jurisdiction |
|---|---|---|---|
| 1 | `\n(?P<id>\d{1,2}[A-Z]?)\.\s*\n(?P<title>...)` | Newline-dot-newline | MahaRERA |
| 2 | `^\s*(?P<id>\d{1,2}[A-Z]?)\.\s+(?P<title>...)\.?—` | Same-line em-dash | MahaRERA |
| 3 | `^Rule\s+(?P<id>\d{1,2}[A-Z]?)\s*[.\-–]\s*(?P<title>...)` | Explicit Rule prefix | Generic |
| 4 | `^\s*(?P<id>\d{1,2}[A-Z]?)\.\s+(?P<title>...?)\.–` | ASCII hyphen `.-` | Karnataka, Tamil Nadu |
| 5 | `(?P<id>\d{1,2}[A-Z]?)-\(1\)\s*\n(?P<title>...)` | `N-(1)\nTitle` | UP RERA multi-clause |
| 6 | `(?P<id>\d{1,2}[A-Z]?)-(?!\()\s*\n(?P<title>...)` | `N-\nTitle` | UP RERA single-clause |

`DocType.ACT` uses a separate pattern set. Fallback: paragraph split on double newlines.
Rule IDs capped at `\d{1,2}` (max 2 digits) — prevents year strings like `2016` matching as rule IDs.
Logs `no_section_boundaries_found` + `fallback_paragraph_chunking` when falling back.

### Chunk Size Limits

```
MIN_CHARS = 100   — discard fragments (headers, page numbers)
MAX_CHARS = 1500  — split large sections at subsection markers (1), (2), (a), (b)
```

### Split Priority for Large Sections

```
1. Subsection markers: \n\s*\((?:\d+|[a-z]{1,3})\)\s+
2. Sentence boundary near MAX_CHARS: rfind('. ')
3. Hard cut at MAX_CHARS (last resort)
```

### parser.py — max_pages cap

```python
@staticmethod
def parse(source: str | Path, max_pages: int | None = None) -> ParsedDocument:
    all_pages = list(doc)
    if max_pages is not None:
        all_pages = all_pages[:max_pages]   # slice before fulltext build
```

---

## 6. Embedding Strategy

**Model:** `nomic-embed-text` (Ollama local)
**Dimension:** 768
**Asymmetric prefixes:**

```
Ingestion time:  "search_document: {text}"   → embed_document()
Query time:      "search_query: {query}"     → embed_query()
```

Using wrong prefix at query time causes ~10–15% recall degradation.

### Truncation Guard

```python
MAX_EMBED_CHARS = 4000   # ~1000 tokens — safe ceiling before prefix added

def embed_document(self, text: str) -> list[float]:
    if len(text) > MAX_EMBED_CHARS:
        log.warning("embedding_truncated", original_len=len(text), truncated_to=MAX_EMBED_CHARS)
        text = text[:MAX_EMBED_CHARS]
    prefixed = f"search_document: {text.strip()}"  # prefix AFTER truncation
    return self.embed_one(prefixed)
```

Truncation happens **before** prefix is added — prevents Ollama 500 errors on Tamil Nadu
and other gazette PDFs where sub-sections exceed 10K chars.

---

## 7. Graph Retriever

`graph_retriever.py` — called on `cross_reference`, `penalty_lookup`, `temporal` query types.

### Section ID Extraction

```python
section_pattern = re.compile(r'\b(?:section|sec\.?|s\.)\s*(\d+[A-Z]?)\b', re.IGNORECASE)
rule_pattern    = re.compile(r'\bRule\s+(\d+[A-Z]?)\b', re.IGNORECASE)
```

### Traversal Strategy (per jurisdiction)

For each jurisdiction (`CENTRAL`, `MAHARASHTRA`, `UTTAR_PRADESH`, `KARNATAKA`, `TAMIL_NADU`):

```
1. Source section chunks    — exact section_id match → is_pinned=True
2. REFERENCES outgoing      — sections source cites (depth=2)
3. REFERENCES incoming      — sections that cite source
4. DERIVED_FROM outgoing    — Act sections this Rule derives from
5. DERIVED_FROM incoming    — Rule sections implementing this Act section
```

### Pinning Rule

Only the exact `section_id` match gets `is_pinned=True`. Sub-sections are NOT pinned.
Max pinned chunks: 2 (one per jurisdiction). Remaining 3 slots filled by reranker.

---

## 8. Response Contract

```python
CivicSetuResponse:
    answer: str                    # plain English, cites section numbers
    citations: list[Citation]      # min_length=1 — NEVER empty
    confidence_score: float        # 0.0–1.0
    confidence_level: str          # "high"/"medium"/"low"
    query_type_resolved: QueryType
    conflict_warnings: list[str]   # empty until Phase 7
    amendment_notice: Optional[str]
    disclaimer: str                # always present

Citation:
    section_id: str
    doc_name: str
    jurisdiction: Jurisdiction
    effective_date: Optional[date]
    source_url: str
    chunk_id: UUID
```

---

## 9. Error Handling

| Scenario | Behaviour |
| :-- | :-- |
| LLM provider rate limited | LiteLLM auto-rotates to next provider |
| All LLM providers fail | `RuntimeError` → FastAPI 500 |
| No chunks retrieved | `InsufficientInfoResponse` returned |
| Hallucination detected | retry (max 2x) → low confidence answer |
| DB unreachable | `/health` returns `degraded`, query returns 500 |
| Scanned PDF detected | Warning logged, fallback URL used (Karnataka) |
| Section patterns not matched | Fallback paragraph chunking, warning logged |
| Neo4j event loop mismatch | Prevented — `_get_driver()` creates fresh driver per call |
| Embedding input too long | Truncated at 4000 chars before prefix; warning logged |
| max_pages exceeded | Parser silently caps pages; total_pages reflects capped count |

---

## 10. Neo4j Graph — Phase 6 State

**Nodes:** 9 Documents, 2090 Sections
**Edges:** 1297 HAS_SECTION, 933 REFERENCES, 91 DERIVED_FROM

### Documents in Graph

| Document | Jurisdiction | DocType | Chunks | Sections | DERIVED_FROM edges |
|---|---|---|---|---|---|
| RERA Act 2016 | CENTRAL | ACT | ~224 | ~224 | — |
| MahaRERA Rules 2017 | MAHARASHTRA | RULES | ~214 | ~214 | 17 sec + 1 doc |
| UP RERA Rules 2016 | UTTAR_PRADESH | RULES | 170 | 33 | 11 sec + 1 doc |
| UP RERA General Regs 2019 | UTTAR_PRADESH | CIRCULAR | 85 | 53 | — |
| Karnataka RERA Rules 2017 | KARNATAKA | RULES | 235 | 45 | 15 sec + 1 doc |
| Tamil Nadu RERA Rules 2017 | TAMIL_NADU | RULES | 157 | 36 | 15 sec + 1 doc |

### Known Open Issues (non-blocking)

| Issue | Affected | Root Cause |
|---|---|---|
| Act §13 missing from graph | UP rule 14, KA rule 11, TN rule 11 | RERA Act ingestion — §13 chunked under different ID |
| Act §66 missing from graph | KA rule 19, TN rule 19 | RERA Act ingestion — §66 not ingested |

### DERIVED_FROM Map Summary

| Jurisdiction | Mapped pairs | Resolved | Unresolved |
|---|---|---|---|
| MAHARASHTRA | 17 | 17 | 0 |
| UTTAR_PRADESH | 15 | 11 | 4 |
| KARNATAKA | 17 | 15 | 2 |
| TAMIL_NADU | 17 | 15 | 2 |

### PDF Source Decisions

| Jurisdiction | Primary URL | Issue | Resolution |
|---|---|---|---|
| CENTRAL | indiacode.nic.in | — | — |
| MAHARASHTRA | naredco.in | — | — |
| UTTAR_PRADESH | up-rera.in/pdf/rera.pdf | pages 25–52 are forms | max_pages=24 |
| KARNATAKA | naredco.in (mirror) | Official PDF fully scanned (19MB) | NAREDCO born-digital |
| TAMIL_NADU | cms.tn.gov.in | pages 16–101 are Forms A–O | max_pages=15 |
