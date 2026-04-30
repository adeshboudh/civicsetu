# CivicSetu - RAG Techniques Reference

**Version:** 2.2 - Cloud Sync + Ingestion Refresh  
**Last Updated:** 2026-04-30

This document describes the retrieval-augmented generation stack currently used in CivicSetu, what is live in the app today, and where the weak spots still are.

---

## 1. Current Status Snapshot

As of **2026-04-30**, CivicSetu's RAG app is usable end-to-end, with a fresh ingestion cycle completed.

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
- **Latest eval artifact improved a lot over old smoke baseline**
  - `eval_results.json` dated **2026-04-28**
  - `faithfulness=0.900`
  - `answer_relevancy=0.858`
  - `context_precision=0.696`
  - `pass_rate=0.581`
- **Knowledge Graph Scale (as of 2026-04-30)**
  - Documents: `6`
  - Sections: `1,160`
  - `REFERENCES` edges: `314`
  - `DERIVED_FROM` edges: `62`
- **Main remaining weakness**
  - multi-jurisdiction retrieval still weak
  - `MULTI` rows pass only `20%`
  - common `fact_lookup` traffic is still less reliable than penalty or graph-heavy queries

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

Query and document embeddings use asymmetric prefixes compatible with Nomic-style retrieval.

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
| `cross_reference` | `graph_retrieval` |
| `penalty_lookup` | `graph_retrieval` |
| `temporal` | `graph_retrieval` |
| `conflict_detection` | `hybrid_retrieval` |

Classifier fallback: if JSON parse fails, default to `fact_lookup` with original query.

### 4.2 LLM Routing and Fallback Chain

All non-streaming LLM calls use `_llm_call()`. Streaming uses `_llm_stream()`.

Current model chain:

```text
THINKING tier
1. gemini/gemini-3.1-flash-lite-preview
2. groq/llama-3.3-70b-versatile
3. openrouter/meta-llama/llama-3.3-70b-instruct:free
4. openrouter/qwen/qwen3.6-plus:free

FAST tier
1. gemini/gemini-3.1-flash-lite-preview
```

Provider notes:

- non-NVIDIA models go through LiteLLM
- NVIDIA-backed models use `ChatNVIDIA` directly
- generator and metadata extraction use `temperature=0.0`
- fast-tier tasks use the lighter chain

---

## 5. Hybrid Retrieval

Hybrid retrieval combines vector similarity and PostgreSQL full-text search, then expands section families.

### 5.1 Vector Similarity Search

Used to catch semantic matches when wording differs from statute text.

Strength:

- good for paraphrase and plain-English phrasing

Weakness:

- can still over-focus on one jurisdiction or sub-clause family

### 5.2 Full-Text Search

Used for exact legal wording, section numbers, and important terms.

Strength:

- precise keyword and section hits

Weakness:

- misses paraphrases and concept-only questions

### 5.3 Reciprocal Rank Fusion

Vector and FTS results are merged with RRF so chunks that rank well in both signals rise to the top.

### 5.4 Section Family Expansion

Top merged sections expand to include parent and sibling sub-sections.

Purpose:

- restore surrounding legal context for split sections
- prevent generator from seeing one isolated sub-clause only

---

## 6. Graph-Based Retrieval

Used for section-centric questions and legal relationships.

Current behavior:

- extract section or rule IDs from query
- traverse Neo4j relationships (`REFERENCES` and `DERIVED_FROM`) populated during [Graph Seeding](#36-graph-seeding)
- hydrate matching sections back from Postgres

Graph retrieval is especially important for:

- explicit section lookups
- penalty questions
- temporal questions
- central vs state derivation paths

Pinned chunks stay ahead of reranked chunks so exact requested sections do not get buried.

---

## 7. Reranking

### 7.1 Cross-Encoder

`retrieval/reranker.py` uses FlashRank with current default:

- `reranker_model = ms-marco-MiniLM-L-12-v2`

Pipeline:

1. deduplicate by `(section_id, doc_name)`
2. split pinned vs rankable chunks
3. rerank rankable chunks with cross-encoder
4. filter by minimum score
5. apply score-gap cutoff
6. prepend pinned chunks

### 7.2 Current Thresholds

Current defaults:

- `reranker_score_threshold = 0.05`
- `reranker_score_gap = 0.95`

These are intentionally recall-friendly compared to older stricter settings.

### 7.3 Final Context Assembly

Current max context size is **7 chunks**.

Assembly rule:

- pinned chunks first
- then top reranked chunks until 7 total

---

## 8. Generation

### 8.1 Buffered Generation

`generator_node()` builds a numbered context block and asks the model for JSON output:

```json
{
  "answer": "<markdown>",
  "confidence_score": 0.0,
  "cited_chunks": [1, 3],
  "amendment_notice": null,
  "conflict_warnings": []
}
```

If parsing fails but raw text exists, answer text is salvaged and citations fall back to all visible chunks.

### 8.2 Streaming Generation

`stream_generator_node()` now drives SSE output.

Flow:

1. run classifier, retrieval, reranker
2. stream plain-text answer tokens to client
3. run a second fast metadata extraction prompt
4. map `cited_chunks` indices back to real citations

Why split answer and metadata:

- keeps first-token latency lower
- avoids forcing model to stream valid JSON
- still preserves grounded citations in final event

### 8.3 Tone Hints by Query Type

Current tone shaping:

| Type | Current Hint |
|---|---|
| `fact_lookup` | direct answer, no analogies or metaphors |
| `penalty_lookup` | lead with consequence |
| `cross_reference` | explain cited section first, then linked sections present in context |
| `conflict_detection` | only claim conflict if both sides are present |
| `temporal` | lead with exact time value if present, otherwise say it is missing |

### 8.4 Citation Extraction

Both buffered and streaming generators anchor citations from 1-based `cited_chunks` indices.

Effect:

- citations reflect chunks actually used by generator
- not every retrieved chunk becomes a citation

---

## 9. Validation

### 9.1 Current Validator Design

Current validator is lightweight and **not** an LLM verifier.

`validator_node()` currently:

- returns early if answer or chunks are missing
- treats `confidence_score < 0.2` as a hallucination risk signal
- otherwise preserves generator confidence

This is cheaper and faster than second-pass validation, but weaker as a grounding guarantee.

### 9.2 Retry Logic

Current retry rule:

```text
no reranked chunks -> end
confidence < 0.2 and retry_count < 2 -> retry
otherwise -> end
```

Max retries: **2**

Consequence:

- avoids excessive cost
- surfaces more low-confidence answers instead of repeatedly re-running graph

### 9.3 Output Guardrails

`guardrails/output_guard.py` still does final shaping:

- below confidence floor, return `InsufficientInfoResponse`
- always append disclaimer
- input guard handles safety/PII before graph execution

---

## 10. RAGAS Evaluation Pipeline

### 10.1 Two-Phase Architecture

Evaluation is checkpointed into two phases:

- **Phase 1:** invoke graph -> `eval_phase1_results.json`
- **Phase 2:** score with RAGAS -> `eval_results.json`

Important current behavior:

- phase 1 resumes from valid cached rows
- phase 2 resumes from already scored rows
- failures do not require restarting from row 1

### 10.2 Dataset

`eval/golden_dataset.jsonl` currently has **31 rows** across:

- `CENTRAL`
- `MAHARASHTRA`
- `UTTAR_PRADESH`
- `KARNATAKA`
- `TAMIL_NADU`
- `MULTI`

Each row includes:

- `query`
- `query_type`
- `ground_truth`
- `expected_section_ids`

### 10.3 Judge Providers

Current supported judge providers:

```bash
JUDGE_PROVIDER=groq       JUDGE_MODEL=llama-3.3-70b-versatile
JUDGE_PROVIDER=gemini     JUDGE_MODEL=gemini/gemma-4-31b-it
JUDGE_PROVIDER=openrouter JUDGE_MODEL=meta-llama/llama-3.3-70b-instruct
JUDGE_PROVIDER=osmapi     JUDGE_MODEL=qwen3.5-397b-a17b
JUDGE_PROVIDER=nvidia     JUDGE_MODEL=z-ai/glm4.7
```

Current rate-limit controls:

- `PHASE2_DELAY_SEC=20`
- `PHASE2_MAX_RETRIES=4`
- retry delay parsed from provider errors when available

### 10.4 Judge Input Trimming

Current defaults:

```text
RAGAS_MAX_CONTEXTS = 7
RAGAS_CONTEXT_CHAR_LIMIT = 1200
RAGAS_ANSWER_CHAR_LIMIT = 1500
RAGAS_REFERENCE_CHAR_LIMIT = 600
```

### 10.5 Latest Full Eval Results

Source: `eval_results.json` generated on **2026-04-28**

Run config captured in artifact:

- graph model: `openai/z-ai/glm4.7`
- judge model: `qwen3.5-397b-a17b`
- pass threshold: `0.7`

Overall:

| Metric | Value |
|---|---|
| Faithfulness | `0.900` |
| Answer Relevancy | `0.858` |
| Context Precision | `0.696` |
| Pass Rate | `0.581` |
| P50 Latency | `101,853.7 ms` |
| P90 Latency | `161,085.2 ms` |

By query type:

| Query Type | Faithfulness | Relevancy | Precision | Pass Rate |
|---|---|---|---|---|
| `penalty_lookup` | `0.866` | `0.873` | `1.000` | `1.000` |
| `temporal` | `0.965` | `0.841` | `0.713` | `0.500` |
| `cross_reference` | `0.894` | `0.796` | `0.736` | `0.500` |
| `conflict_detection` | `0.899` | `0.911` | `0.551` | `0.500` |
| `fact_lookup` | `0.880` | `0.865` | `0.513` | `0.429` |

By jurisdiction:

| Jurisdiction | Faithfulness | Relevancy | Precision | Pass Rate |
|---|---|---|---|---|
| `TAMIL_NADU` | `0.978` | `0.900` | `0.747` | `0.800` |
| `KARNATAKA` | `0.907` | `0.847` | `0.851` | `0.800` |
| `UTTAR_PRADESH` | `0.978` | `0.904` | `0.600` | `0.600` |
| `MAHARASHTRA` | `0.834` | `0.879` | `0.702` | `0.600` |
| `CENTRAL` | `0.933` | `0.792` | `0.912` | `0.500` |
| `MULTI` | `0.763` | `0.838` | `0.322` | `0.200` |

Interpretation:

- grounding is now strong enough that hallucination is no longer the primary failure mode
- ranking quality improved materially
- multi-jurisdiction comparison remains the biggest retrieval gap
- eval latency is high enough that streaming is important for user-facing UX

---

## 11. Known Failure Modes

### 11.1 Multi-Jurisdiction Retrieval

Cross-jurisdiction questions still underperform.

Observed symptom:

- `MULTI` pass rate only `0.200`

Likely causes:

- one jurisdiction dominates semantic retrieval
- reranker still sees mixed chunks without explicit two-sided decomposition

### 11.2 Fact Lookup Precision

`fact_lookup` is still weakest among common traffic:

- pass rate `0.429`
- context precision `0.513`

Likely causes:

- broad questions invite many semantically related sub-clauses
- exact top-level section sometimes loses to more specific descendants

### 11.3 Latency

Eval-mode latency remains large:

- p50 ~102 seconds
- p90 ~161 seconds

Implication:

- buffered endpoint is expensive for user experience
- streaming endpoint is necessary, not optional

---

## 12. Configuration Reference

Important RAG settings from `config/settings.py`:

| Parameter | Default | Effect |
|---|---|---|
| `primary_model` | `gemini/gemini-3.1-flash-lite-preview` | main thinking-tier model |
| `fast_model` | `gemini/gemini-3.1-flash-lite-preview` | fast-tier tasks |
| `fallback_model_1` | `groq/llama-3.3-70b-versatile` | first fallback |
| `fallback_model_2` | `openrouter/meta-llama/llama-3.3-70b-instruct:free` | second fallback |
| `fallback_model_3` | `openrouter/qwen/qwen3.6-plus:free` | third fallback |
| `embedding_model` | `nomic-embed-text` | embedding model |
| `embedding_dimension` | `768` | pgvector dimension |
| `reranker_model` | `ms-marco-MiniLM-L-12-v2` | FlashRank cross-encoder |
| `reranker_score_threshold` | `0.05` | minimum score to keep |
| `reranker_score_gap` | `0.95` | score-cliff cutoff |

Important eval env vars:

| Env Var | Effect |
|---|---|
| `EVAL_PHASE` | run phase 1 only, phase 2 only, or both |
| `EVAL_LIMIT` | limit number of rows |
| `EVAL_IDS` | evaluate only specific row IDs |
| `EVAL_JURISDICTION` | evaluate one jurisdiction only |
| `JUDGE_PROVIDER` | judge backend |
| `JUDGE_MODEL` | judge model |
| `NO_REASONING` | disable thinking where supported |
| `PASS_THRESHOLD` | pass threshold |
| `PHASE2_DELAY_SEC` | inter-row delay for judge calls |
| `RAGAS_MAX_CONTEXTS` | contexts sent to judge |
| `RAGAS_CONTEXT_CHAR_LIMIT` | trim limit per context |

---

## 13. Implementation Checklist

When adding a new jurisdiction or corpus:

- add `DocumentSpec` with correct page cap
- verify PDF is text-extractable, not image-only
- ingest and inspect fallback chunking logs
- verify chunk counts in Postgres
- run graph seeding (manual via `scripts/seed_phase3.py` or via `scripts/ingest.py`)
- add eval rows for all major query types
- rerun full eval and inspect jurisdiction-specific precision
