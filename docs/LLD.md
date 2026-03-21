# CivicSetu — Low Level Design (LLD)

**Version:** 0.4.0 — Phase 4 Complete
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
│   ├── chunker.py            Section-boundary regex chunker — 5 format patterns + fallback
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
  (:Section) -[:DERIVED_FROM]->(:Section)     -- State Rule N → RERA Act Sec M
  (:Document)-[:DERIVED_FROM]->(:Document)    -- State Rules → RERA Act 2016

Planned (Phase 5+):
  (:Section) -[:SUPERSEDES]->(:Section)
  (:Section) -[:AMENDED_BY]->(:Amendment)
  (:Section) -[:CONFLICTS_WITH]->(:Section)
```

**Live graph stats (Phase 3):**

| Metric       | Count |
|--------------|-------|
| Documents    | 7     |
| Sections     | 1184  |
| HAS_SECTION  | 905   |
| REFERENCES   | 665   |
| DERIVED_FROM | 51    |

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

### Ingested Documents (Phase 4)

| Key | Document | Jurisdiction | DocType | Chunks | max_pages |
|---|---|---|---|---|---|
| `rera_act_2016` | RERA Act 2016 | CENTRAL | ACT | ~224 | None |
| `mahrera_rules_2017` | MahaRERA Rules 2017 | MAHARASHTRA | RULES | ~214 | None |
| `up_rera_rules_2016` | UP RERA Rules 2016 | UTTAR_PRADESH | RULES | 170 | 24 |
| `up_rera_general_regulations_2019` | UP RERA General Regulations 2019 | UTTAR_PRADESH | CIRCULAR | 85 | None |

> `up_rera_rules_2016` uses `max_pages=24` — pages 1–24 are rule text; pages 25–52 are
> prescribed forms/schedules that pollute section IDs if ingested.

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
| conflict_detection                    | hybrid_retrieval (Phase 5)               |

```
validator → route_after_validator:
    confidence >= 0.5 AND not hallucinated → END
    (confidence < 0.5 OR hallucinated) AND retry_count < 2 → retry → classifier
    (confidence < 0.5 OR hallucinated) AND retry_count >= 2 → END (low confidence answer)
```

---

## 5. Chunking Strategy

### Section Boundary Detection

Five regex patterns across DocType.RULES, tried in order (first match wins per line):

| # | Pattern | Format | Example |
|---|---|---|---|
| 1 | `\n(?P<id>\d+[A-Z]?)\.\s*\n(?P<title>...)` | MahaRERA newline-dot | `\n3.\nInformation to be furnished` |
| 2 | `^\s*(?P<id>\d+[A-Z]?)\.\s+(?P<title>...)\.?—` | Same-line dash | `3. Application for registration.—` |
| 3 | `^Rule\s+(?P<id>\d+[A-Z]?)\s*[.\-–]\s*(?P<title>...)` | Explicit Rule prefix | `Rule 3 - Application` |
| 4 | `(?P<id>\d{1,2}[A-Z]?)-\(1\)\s*\n(?P<title>...)` | UP RERA multi-clause | `7-(1)\nThe registration granted...` |
| 5 | `(?P<id>\d{1,2}[A-Z]?)-(?!\()\s*\n(?P<title>...)` | UP RERA single-clause | `15-\nThe authority shall maintain...` |

DocType.ACT uses a separate pattern set. Fallback: paragraph split on double newlines.
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

Used for UP RERA Rules 2016 (`max_pages=24`) to exclude prescribed forms in pages 25–52.

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
The `embed_document()` / `embed_query()` method split enforces this at the API level.

### Truncation Guard

```python
MAX_EMBED_CHARS = 6000   # ~1500 tokens for nomic-embed-text
if len(text) > MAX_EMBED_CHARS:
    log.warning("embedding_truncated", original_len=len(text), truncated_to=MAX_EMBED_CHARS)
    text = text[:MAX_EMBED_CHARS]
```

---

## 7. Graph Retriever — Phase 3 Logic

`graph_retriever.py` — called on `cross_reference`, `penalty_lookup`, `temporal` query types.

### Section ID Extraction

```python
# Matches both "Section 18" / "Sec 18" and "Rule 3" / "Rule 12A"
section_pattern = re.compile(r'\b(?:section|sec\.?|s\.)\s*(\d+[A-Z]?)\b', re.IGNORECASE)
rule_pattern    = re.compile(r'\bRule\s+(\d+[A-Z]?)\b', re.IGNORECASE)
```

Section pattern takes priority; Rule pattern is fallback.

### Traversal Strategy (per jurisdiction)

For each jurisdiction (`CENTRAL`, `MAHARASHTRA`, `UTTAR_PRADESH`):

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

## 8. Response Contract

Every response from `POST /api/v1/query` conforms to:

```python
CivicSetuResponse:
    answer: str                    # plain English, cites section numbers
    citations: list[Citation]      # min_length=1 — NEVER empty
    confidence_score: float        # 0.0–1.0
    confidence_level: str          # computed: "high"/"medium"/"low"
    query_type_resolved: QueryType
    conflict_warnings: list[str]   # empty until Phase 5
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

## 9. Error Handling

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
| max_pages exceeded | Parser silently caps pages; logged in `pdf_parsed` as total_pages |

---

## 10. Neo4j Graph — Phase 4 State

**Nodes:** 7 Documents, 1184 Sections
**Edges:** 905 HAS_SECTION, 665 REFERENCES, 51 DERIVED_FROM

### Documents in Graph

| Document | Jurisdiction | DocType | Chunks | Sections | REF edges | DERIVED_FROM |
|---|---|---|---|---|---|---|
| RERA Act 2016 | CENTRAL | ACT | ~224 | ~224 | ~350 out | — |
| MahaRERA Rules 2017 | MAHARASHTRA | RULES | ~214 | ~214 | ~61 intra | 17 sec + 1 doc |
| UP RERA Rules 2016 | UTTAR_PRADESH | RULES | 170 | 33 | 102 | 11 sec + 1 doc |
| UP RERA General Regs 2019 | UTTAR_PRADESH | CIRCULAR | 85 | 53 | 18 | — |

> 3 additional documents exist in the DB from earlier ingestion runs (cleaned from graph).
> Graph reflects only clean, re-seeded documents.

### DERIVED_FROM Maps

**MahaRERA Rules 2017 → RERA Act 2016** (17 pairs):

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

**UP RERA Rules 2016 → RERA Act 2016** (15 pairs, 11 resolved, 4 unresolved):

| UP Rule | RERA Act Section | Status |
|---|---|---|
| 3 | 4 | ✓ |
| 4 | 4 | ✓ |
| 5 | 4 | ✓ |
| 6 | 5 | ✓ |
| 7 | 6 | ✓ |
| 8 | 7 | ✗ rule_found=False (chunk ID mismatch) |
| 9 | 9 | ✓ |
| 10 | 9 | ✓ |
| 11 | 9 | ✓ |
| 12 | 10 | ✗ rule_found=False |
| 13 | 11 | ✗ rule_found=False |
| 14 | 13 | ✗ act_found=False (RERA Act §13 missing) |
| 15 | 18 | ✓ |
| 16 | 19 | ✓ |
| 18 | 31 | ✓ |

### MetadataExtractor — reference patterns

```python
# Section references: "Section 18", "section 4A", "s. 9", "sec 18"
extract_section_references(text) → list[str]

# Rule references: "Rule 3", "Rule 12A", "under Rule 9"
extract_rule_references(text) → list[str]
```

Cross-act scrub removes false positives from IPC, CPC, Companies Act, etc.
RERA section bounds filter: sections 1–92 only.

### GraphStore — driver lifecycle

`_get_driver()` creates a **fresh** `AsyncDriver` on every call — no singleton, no
`@lru_cache`. Every method wraps its session in `try/finally: await driver.close()`.
This prevents `RuntimeError: Future attached to a different loop` when `asyncio.run()`
is called multiple times in the same process (LangGraph sync `.invoke()` pattern).
