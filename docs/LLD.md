# CivicSetu — Low Level Design (LLD)

**Version:** 2.0.0 — Phase 8 Complete (RAGAS Evaluation + Retrieval Improvements)
**Live:** https://civicsetu-two.vercel.app
**Last Updated:** April 2026

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
│   ├── embedder.py           nomic-embed-text-v1.5 via sentence-transformers — truncate at 4000 chars pre-prefix
│   ├── pipeline.py           Orchestrates ingestion; prepends section_title to embeddings
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
│   ├── nodes.py              Pure functions: classifier, _rrf_retrieve (shared hybrid),
│   │                         vector_retrieval, graph_retrieval, hybrid_retrieval,
│   │                         reranker, generator, validator
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

eval/
├── golden_dataset.jsonl      31-row RAGAS evaluation dataset across 5 jurisdictions
scripts/
├── run_eval.py               Two-phase RAGAS evaluation: Phase 1 (graph invoke) + Phase 2 (RAGAS scoring)

frontend/                     Next.js 15 App Router — deployed on Vercel
├── src/app/
│   ├── layout.tsx            Root layout: ThemeProvider + dark mode
│   ├── page.tsx              Main page: wires all components together
│   └── globals.css           Tailwind directives + gradient utilities
├── src/components/
│   ├── Header.tsx            Logo, new chat, theme toggle, GitHub link
│   ├── ChatThread.tsx        Scrollable message list + empty state examples
│   ├── MessageBubble.tsx     User/assistant/error bubbles with badges + citations
│   ├── ConfidenceBadge.tsx   HIGH/MEDIUM/LOW pill
│   ├── CitationsPanel.tsx    Collapsible citation cards
│   └── InputBar.tsx          Auto-resize textarea, jurisdiction select, send
├── src/hooks/
│   └── useChat.ts            Chat state, session_id localStorage, sendMessage
└── src/lib/
    ├── types.ts              TypeScript interfaces (mirrors backend Pydantic models)
    └── api.ts                queryRera() fetch wrapper → /api/v1/query
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
| fact_lookup                           | vector_retrieval (RRF hybrid)            |
| cross_reference                       | graph_retrieval (→ RRF fallback)         |
| penalty_lookup                        | graph_retrieval (→ RRF fallback)         |
| temporal                              | graph_retrieval (→ RRF fallback)         |
| conflict_detection                    | hybrid_retrieval (RRF across jur.)       |

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

**Model:** `nomic-embed-text-v1.5` (via `sentence-transformers`, local — no Ollama required)
**Dimension:** 768
**Asymmetric prefixes** (MTEB/nomic-embed requirement):

```
Ingestion time:  "search_document: {section_title}\n{text}"  → pipeline.py
Query time:      "search_query: {rewritten_query}"            → retrieval/__init__.py
```

**Section title prepend (Phase 8 change):** `pipeline.py` prepends `section_title` to the
embedded text so sub-chunks (e.g. `S.11(2)`) retain their section context.
Without this, sub-chunks embed without "Obligations of promoter" — cosine similarity misses them.
The reranker still receives raw `chunk.text` (no title prefix).

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

## 7. Hybrid Retrieval — `_rrf_retrieve()`

All retrieval nodes share a single async helper `_rrf_retrieve()` in `agent/nodes.py`.

### Reciprocal Rank Fusion (RRF)

```python
RRF_K = 60   # standard constant

rrf_score(chunk) = 1/(K + rank_in_vector) + 1/(K + rank_in_fts)
```

Fetches `top_k × 3` vector results and `top_k × 2` FTS results, deduplicates by `chunk_id`,
merges via RRF, returns top `top_k × 2`.

### Full-Text Search

`VectorStore.full_text_search()` uses `websearch_to_tsquery` in OR mode:

```sql
WHERE to_tsvector('english', text) @@ websearch_to_tsquery('english', :query)
ORDER BY ts_rank(to_tsvector('english', text), websearch_to_tsquery('english', :query)) DESC
```

Changed from `plainto_tsquery` (AND-mode) — AND required all query words to match,
excluding relevant sections that matched most but not all words.

### Section Family Expansion

After RRF merge, top-3 results trigger family expansion:

```python
for rc in merged[:3]:
    base_sid = re.sub(r'\([^)]*\)$', '', section_id).strip()  # "5(4)" → "5"
    family = await VectorStore.get_section_family(section_id=base_sid, jurisdiction=jur)
    # returns all chunks where section_id = '5' OR section_id LIKE '5(%'
```

`get_section_family` guard: skips if `section_id` already contains `(` (base_sid computation
strips this before calling). Hard cap: `_MAX_VECTOR_EXPANDED = 40` chunks before reranker.

**Why top-3 not top-1:** If top-1 RRF result is a sub-section (`S.5(4)`), its parent
family is expanded. But if the truly relevant parent section (`S.11`) appears at RRF rank 2,
only expanding top-1 misses it. Expanding top-3 covers more cases at the cost of a slightly
larger pool.

---

## 7b. Reranker Detail

`reranker_score_threshold = 0.1` — minimum cross-encoder score to enter candidate pool.
`reranker_score_gap = 0.6` — gap filter cliff threshold.

**Gap filter:**

```python
def _apply_score_gap(chunks, gap=0.6):
    for i in range(1, len(chunks)):
        if chunks[i-1].rerank_score - chunks[i].rerank_score >= gap:
            return chunks[:i]
    return chunks
```

**Threshold history:** Originally `threshold=0.3, gap=0.35`. Gap=0.35 was too aggressive —
cut chunks with 0.36 score drop, leaving only 1 context for generator. Raised to 0.6 (Phase 8).

Final context: `pinned_chunks + gap_filtered[:max(0, 5 - len(pinned))]` → max 5 chunks.

---

## 8. Graph Retriever

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

## 9. Response Contract

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

## 10. Neo4j Graph — Phase 6 State (Current)

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

## 11. Agent Pipeline — Bug Fixes (2026-03-22)

Three production bugs fixed after 12-case E2E suite. All verified: 0 retries, 0
hallucinations, avg latency 7.6s.

### Fix 1 — `vector_store.py::get_section_family` — Pydantic crash on SELECT *

`SELECT *` returned `embedding` as a raw string; Pydantic `list[float]` validation
failed. Fix: explicit column projection, `embedding=None` on all returned chunks.
Matches every other `VectorStore` method.

### Fix 2 — `nodes.py::vector_retrieval_node` — Reranker blowup on section expansion

Section family expansion ran on all 5 similarity hits → up to 121 chunks → FlashRank
cross-encoder serial scoring → 65s reranker time. Fix (Phase 5): expand top-1 hit only; hard
cap at 25 chunks before reranker.

```python
for rc in results[:1]:
    ...family expansion...
expanded = expanded[:25]  # hard safety cap
```

**Phase 8 update:** Expanded to top-3 after RAGAS eval revealed that when a sub-section
(e.g. `S.5(4)`) ranks #1, its parent `S.5` (with the 30-day rule) was never expanded.
Cap raised to 40 to accommodate larger families.

```python
for rc in merged[:3]:    # top-3 RRF results (was: top-1)
    ...family expansion...
expanded = expanded[:40]  # was: 25
```


### Fix 3 — `nodes.py::validator_node` — False hallucination flag

Validator built context as raw `chunk.text` joined string. Generator answer cites
`"Section 11(1)"` but raw text has no section number → validator scores 0.2 →
`hallucinated=True` → spurious retry loops (7 retries across 12 tests).

Fix: mirror generator's numbered context block `[i] doc — section_id: title\ntext`.
Validator can now match cited section numbers to source context.

### E2E Regression Results (post-fix)

| Metric | Pre-fix | Post-fix |
| :-- | :-- | :-- |
| Avg latency | 19.6s | **7.6s** |
| Max latency | 87.1s | **13.3s** |
| Avg confidence | 0.908 | **0.958** |
| Total retries | 7 | **0** |
| Slow (>20s) | 3 | **0** |
| Low conf (<0.7) | 2 | **0** |
| Pass rate | 12/12 | **12/12** |

---

## 12. Agent Pipeline — RAGAS Eval Fixes (Phase 8, April 2026)

Five changes from RAGAS evaluation revealing retrieval and faithfulness failures.

### Fix 4 — Reranker thresholds too aggressive (`settings.py`)

Old `score_gap=0.35` cut after any 0.36 point drop → only 1 chunk reached generator.
New: `score_threshold=0.1`, `score_gap=0.6`. Keeps secondary relevant chunks while still
filtering genuine noise (0.98 → 0.20 drop would still cut at 0.78 gap).

### Fix 5 — Generator analogy instruction caused hallucination (`generator.py`)

"Use an analogy or real-world example" produced analogies ("Think of it like selling a
used car") not present in retrieved context → faithfulness judge scored as hallucination.
Fix: removed analogy instruction; replaced with "using only information from the provided context".

### Fix 6 — Generator weak grounding for sparse contexts (`generator.py`)

Generator constructed legal conclusions from reasoning even when context lacked evidence.
Added explicit rules:
- For sparse context: say "Based on the available context: [X]" and note missing elements
- For conflict detection: only assert conflict if BOTH provisions present in context

### Fix 7 — CONFLICT_DETECTION tone hint implied precedence reasoning (`nodes.py`)

Tone hint said "state which jurisdiction takes precedence when context supports it" —
LLM interpreted "when context supports it" loosely and applied legal reasoning.
Rewritten to: "Never infer precedence from legal reasoning — only state precedence if
the context explicitly says so."

### Fix 8 — Temporal query rewrite too generic (`classifier.py`)

Query "What is the timeline for project registration?" produced rewrite "registration
timeline period" — FTS missed Section 5 which uses "within thirty days" and "deemed registered".
Added rewriting guidance to expand temporal queries with specific legal time-period keywords.

### RAGAS Results (Phase 8 baseline, 5-row smoke, gemma-4-31b-it judge)

| Row | Faith (before) | Faith (after) | Prec (before) | Prec (after) |
|---|---|---|---|---|
| CENTRAL-FACT-001 | 1.00 | 0.50 | 0.00 | 0.00 |
| CENTRAL-FACT-002 | 0.80 | 0.62 | 0.00 | 0.33 |
| CENTRAL-XREF-001 | 0.63 | 0.50 | 1.00 | 1.00 |
| CENTRAL-CONF-001 | 0.00 | 0.62 | 0.00 | 0.00 |
| CENTRAL-TEMP-001 | 0.67 | 1.00 | 1.00 | 0.00 |
| **Overall** | 0.618 | **0.650** | 0.400* | 0.267 |

\* Before baseline had inflated precision from duplicate chunks (non-deterministic doc_id).
After Phase 8: deterministic UUID5 chunk IDs prevent duplicates on re-ingest.