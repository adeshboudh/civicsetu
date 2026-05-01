# CivicSetu - RAG Techniques Reference

**Version:** 2.3 - Mobile Ledger + Quality Hardening  
**Last Updated:** 2026-05-01

This document describes the retrieval-augmented generation stack currently used in CivicSetu, what is live in the app today, and where the weak spots still are.

---

## 1. Current Status Snapshot

As of **2026-05-01**, CivicSetu's RAG app is at production-grade stability (v1.0.0-level), with mobile responsiveness and retrieval quality fixes live.

- **Phase 9 Complete (Mobile Responsive)**
  - Dual-pane layout for desktop; tabbed "Digital Ledger" UI for mobile.
  - Interactive Graph Explorer with section drill-down.
- **Cloud Infrastructure Live**
  - Relational & Vector: **Neon (Postgres + pgvector)**
  - Graph: **Neo4j AuraDB**
  - Frontend: **Vercel**
  - Backend API: **Hugging Face Spaces**
- **Live app routes**
  - `POST /api/v1/query` - buffered response
  - `POST /api/v1/query/stream` - SSE token streaming
  - `POST /api/v1/query/section-context` - section-focused chat
  - `/api/v1/graph/*` - graph explorer and section drill-down
- **Session-aware graph**
  - LangGraph uses `session_id` as thread key.
  - Each turn clears retrieval/generation fields but preserves conversation history.
- **Active retrieval routing**
  - `fact_lookup -> vector_retrieval`
  - `cross_reference|penalty_lookup|temporal -> graph_retrieval`
  - `conflict_detection -> hybrid_retrieval`
- **Streaming is now first-class**
  - streaming path reuses classifier, retrieval, and reranker
  - answer text streams first
  - citations and metadata are extracted in a second fast pass
- **Latest eval artifact (0.90 Faithfulness)**
  - `eval_results.json` dated **2026-04-28**
  - `faithfulness=0.900`
  - `answer_relevancy=0.858`
  - `context_precision=0.696`
  - `pass_rate=0.581`
- **Knowledge Graph Scale (as of 2026-05-01)**
  - Documents: `6`
  - Sections: `2,090`
  - Edges: `2,321` (REFERENCES, DERIVED_FROM, HAS_SECTION)
- **Main remaining weakness**
  - multi-jurisdiction retrieval still weak (`MULTI` rows pass only `20%`)
  - context precision for broad fact lookups needs further HNSW tuning

---

## 2. System Overview

CivicSetu is a legal-domain RAG system over five Indian RERA jurisdictions plus cross-jurisdiction queries.

Core problem:

- legal text is structured around sections, rules, sub-clauses, and cross-references
- users ask imprecise natural-language questions
- answers must stay grounded and cite the right legal section

Why plain semantic RAG fails here:

- embeddings blur important legal entities
- user queries often omit exact statute wording
- conflict questions need more than one legal source
- generation models tend to fill gaps unless grounding is strict

---

## 3. Ingestion Pipeline

### 3.1 PDF Parsing

`ingestion/parser.py` uses **PyMuPDF**.

Important guards:

- document-level `max_pages` trims form-heavy tails
- scanned PDF detection avoids unusable OCR-free sources
- metadata stores capped page count, not necessarily total PDF pages

### 3.2 Section Boundary Chunking

`ingestion/chunker.py` applies multiple regex families in priority order to detect section and rule boundaries.

Current purpose:

- preserve citation boundaries
- keep section hierarchy intact
- split oversized sections without destroying legal structure

Fallback mode is paragraph chunking on double newlines, logged as `fallback_paragraph_chunking`.

### 3.3 Deterministic Chunk IDs

`chunk_id` is a UUID5 over stable section identity data.

Effect:

- re-ingestion is idempotent
- `ON CONFLICT DO UPDATE` replaces old chunk content
- same legal section does not duplicate across re-runs

### 3.4 Section Title Prepended to Embeddings

During embedding, section title is prepended to chunk text.

Reason:

- split sub-sections often lose the title phrase that users actually search for
- title prefix restores semantic recall for questions like "obligations of promoter"

Reranker still reads raw chunk text, not the prefixed text.

### 3.5 Embedding Model

Current defaults from `config/settings.py`:

- `embedding_model = nomic-embed-text`
- `embedding_dimension = 768`

Query and document embeddings use asymmetric prefixes (`search_query: ` vs `search_document: `) compatible with Nomic-style retrieval.

### 3.6 Graph Seeding

`ingestion/graph_seeder.py` populates the Neo4j knowledge graph using data already persisted in PostgreSQL.

Key steps:
- **Idempotent Upsert:** Documents and Sections are merged into Neo4j using UUID5 `chunk_id`.
- **Relationship Extraction:** 
  - `REFERENCES`: `MetadataExtractor` identifies section numbers in text (e.g., "under section 18"). Handles internal and cross-jurisdiction links.
  - `DERIVED_FROM`: Static mapping identifies which State Rule sections derive from which Central Act sections (both at Document and Section level).
- **Execution:** Automatically triggered at the end of `scripts/ingest.py` or manually via `scripts/seed_phase3.py`.

---

## 4. Query Pipeline

### 4.1 Query Classification and Rewriting

`agent/nodes.py::classifier_node` classifies query and rewrites it for retrieval.

Output shape:

```json
{
  "query_type": "fact_lookup | cross_reference | temporal | penalty_lookup | conflict_detection",
  "rewritten_query": "expanded retrieval-friendly query"
}
```

Current route mapping:

| Query Type | Route |
|---|---|
| `fact_lookup` | `vector_retrieval` |
| `cross_reference" | `graph_retrieval` |
| `penalty_lookup` | `graph_retrieval` |
| `temporal` | `graph_retrieval` |
| `conflict_detection" | `hybrid_retrieval` |

Classifier fallback: if JSON parse fails, default to `fact_lookup` with original query.

### 4.2 LLM Routing and Fallback Chain

All non-streaming LLM calls use `_llm_call()`. Streaming uses `_llm_stream()`.

Current model chain:

```text
THINKING tier (Generator)
1. gemini/gemini-1.5-flash
2. groq/llama-3.3-70b-versatile
3. NVIDIA NIM: z-ai/glm4.7 | minimaxai/minimax-m2.7

FAST tier (Classifier/Validator)
1. gemini/gemini-1.5-flash
```

Provider notes:

- NVIDIA-hosted models (Minimax, GLM) use `https://integrate.api.nvidia.com/v1`
- `temperature=0.0` for all grounding tasks
- Gemini models use a temperature of `1.0` if specified as such by provider requirements for certain tiers.

---

## 5. Hybrid Retrieval

Hybrid retrieval combines vector similarity and PostgreSQL full-text search, then expands section families.

### 5.1 Vector Similarity Search

Used to catch semantic matches when wording differs from statute text.

### 5.2 Full-Text Search

Used for exact legal wording, section numbers, and important terms via `websearch_to_tsquery`.

### 5.3 Reciprocal Rank Fusion

Vector and FTS results are merged with RRF so chunks that rank well in both signals rise to the top.

### 5.4 Section-ID-Aware Direct Lookup

If a query contains explicit section/rule numbers (e.g., "Section 18 refund"), the retriever performs a direct indexed lookup for those sections and **pins** them to the top of the retrieval list. This acts as a safety net when semantic search fails to rank the exact section high enough.

### 5.5 Central Act Supplementation

For queries filtered by a specific State Jurisdiction (e.g., Maharashtra), the retriever automatically supplements results with chunks from the **Central RERA Act 2016**. This is critical because state rules often omit core definitions or penalties that are defined once in the Central Act.

---

## 6. Graph-Based Retrieval

Used for section-centric questions and legal relationships.

Current behavior:

- extract section or rule IDs from query
- traverse Neo4j relationships (`REFERENCES` and `DERIVED_FROM`)
- hydrate matching sections back from Postgres

Graph retrieval is especially important for:

- explicit section lookups
- penalty questions
- central vs state derivation paths

Pinned chunks (from direct lookup or graph traversal) stay ahead of reranked chunks.

---

## 7. Reranking

### 7.1 Cross-Encoder

`retrieval/reranker.py` uses FlashRank (`ms-marco-MiniLM-L-12-v2`).

Pipeline:

1. deduplicate by `(section_id, doc_name)`
2. split pinned vs rankable chunks
3. rerank rankable chunks with cross-encoder
4. filter by minimum score (0.05)
5. apply score-gap cutoff (0.95)
6. prepend pinned chunks

### 7.2 Context Assembly

Max context size is **7 chunks**. Pinned chunks (exact matches) are never discarded by the reranker unless the context is fully saturated.

---

## 8. Generation

### 8.1 Buffered Generation

`generator_node()` builds a numbered context block and asks for JSON output.

### 8.2 Streaming Generation

`stream_generator_node()` now drives SSE output.
1. Run classification/retrieval/reranking.
2. Stream answer tokens immediately.
3. Run a second fast metadata extraction prompt
4. Push metadata/citations as the final SSE event.

### 8.3 Tone Hints by Query Type

| Type | Tone Guidance |
|---|---|
| `fact_lookup` | Direct, no metaphors, cite per bullet. |
| `penalty_lookup` | Lead with consequence/penalty. |
| `cross_reference` | Explain primary section, then connections. |
| `conflict_detection` | Flag contradiction ONLY if both sides are in context. |
| `temporal` | Lead with exact numeric deadline/time. |

---

## 9. Validation

### 9.1 Validator Design

`validator_node()` treats `confidence_score < 0.2` as a hallucination risk.
- Returns `hallucination_flag: True` if score is below floor.
- Graph triggers a **retry** (up to 2 times) with different retrieval parameters if flagged.

### 9.2 Output Guardrails

`guardrails/output_guard.py`:
- Intercepts low-confidence or safe-guard failures.
- Returns `InsufficientInfoResponse` when grounding is weak.
- Appends legal disclaimer.

---

## 10. RAGAS Evaluation Pipeline

### 10.1 Two-Phase Architecture

- **Phase 1:** Graph invocation -> `eval_phase1_results.json`.
- **Phase 2:** RAGAS scoring -> `eval_results.json`.

### 10.2 Dataset & Metrics

- **Rows:** 31 (Central, 4 States, Multi-Jurisdiction).
- **Primary Metrics:** Faithfulness, Answer Relevancy, Context Precision.
- **Goal:** Faithfulness > 0.85; Answer Relevancy > 0.80.

---

## 11. Known Failure Modes

- **Multi-Jurisdiction Retrieval:** Reranker often prefers one jurisdiction's terminology, leading to unbalanced context for comparison queries.
- **Large Context Noise:** 7 chunks sometimes include irrelevant sub-clauses that distract the generator.

---

## 12. Implementation Checklist

- [x] Add `DocumentSpec` to registry.
- [x] Verify PDF text extraction.
- [x] Run `make ingest`.
- [x] Seed Neo4j graph.
- [x] Run `make eval-smoke` to verify precision.
