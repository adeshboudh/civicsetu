# CivicSetu — RAG Techniques Reference

**Version:** 2.0 — Phase 8 (RAGAS Evaluation)
**Last Updated:** April 2026

This document covers every retrieval-augmented generation technique used in CivicSetu,
why each decision was made, and what it costs when it goes wrong.

---

## 1. System Overview

CivicSetu is a **legal domain RAG system** over five Indian RERA jurisdictions.
The core challenge: legal text is highly structured (section numbers, cross-references,
state vs central hierarchy) and users ask imprecise plain-English questions that need
precise legal citations. Standard RAG fails here because:

- Dense embeddings miss entity distinctions ("promoter" vs "agent" look similar at 768 dims)
- Query strings rarely contain the exact legal keywords in the source text
- Single-source retrieval misses the two-law comparison needed for conflict queries
- Generator LLMs "fill in the gaps" with legal reasoning not present in context

Every technique below addresses one of these failure modes.

---

## 2. Ingestion Pipeline

### 2.1 PDF Parsing

`ingestion/parser.py` uses **PyMuPDF** for text extraction. Key guards:
- `max_pages` per document — UP Rules truncate at page 24 (pages 25–52 are government forms),
  Tamil Nadu at page 15 (pages 16–101 are Forms A–O)
- Scanned page detection — Karnataka's official PDF is a 19MB image scan; NAREDCO mirror used instead
- `total_pages` in metadata reflects the capped count, not the PDF total

### 2.2 Section Boundary Chunking

`ingestion/chunker.py` applies **six regex patterns** in priority order. First match per line wins.

| # | Pattern | Example | Jurisdiction |
|---|---|---|---|
| 1 | `\n{id}.\n{title}` | `\n3.\nRegistration` | MahaRERA |
| 2 | `{id}. {title}.—` | `3. Registration.—` | MahaRERA |
| 3 | `Rule {id} - {title}` | `Rule 3 - Application` | Generic |
| 4 | `{id}. {title}.–` | `3. Application.–` | Karnataka, Tamil Nadu |
| 5 | `{id}-(1)\n{title}` | `3-(1)\nApplication` | UP RERA (multi-clause) |
| 6 | `{id}-\n{title}` | `3-\nApplication` | UP RERA (single-clause) |

`DocType.ACT` uses a separate pattern set. Fallback: paragraph split on double newlines,
logged as `fallback_paragraph_chunking`.

**Chunk size limits:**

```
MIN_CHARS = 100    — discard page headers, footnotes, empty sections
MAX_CHARS = 1500   — split large sections at subsection markers
```

**Large section splitting priority:**
1. Subsection markers: `\n\s*\((?:\d+|[a-z]{1,3})\)\s+`
2. Sentence boundary near MAX_CHARS: `rfind('. ')`
3. Hard cut at MAX_CHARS (last resort, logs a warning)

Sub-chunks get IDs like `"11(1)"`, `"11(2)"` — the base section becomes `"11"`.

### 2.3 Deterministic Chunk IDs

`chunk_id` is a **UUID5 hash** of `(doc_id, section_id, chunk_index)`. This makes
re-ingestion idempotent — the same section always gets the same UUID, so `ON CONFLICT DO UPDATE`
replaces rather than duplicates. Earlier versions used random UUIDs; re-ingest doubled the corpus.

### 2.4 Section Title Prepended to Embeddings

`ingestion/pipeline.py` prepends `section_title` to the text before embedding:

```python
texts = [
    f"{c['section_title']}\n{c['text']}" if c.get('section_title') else c['text']
    for c in raw_chunks
]
```

**Why:** Sub-chunks (e.g. `S.11(2)`, `S.11(3)`) split off from a parent section carry
the body text but lose the section header. Without prepending, a query for
"obligations of promoter" cannot match sub-chunks of Section 11 via cosine similarity
because the phrase "obligations of promoter" never appears in their raw text.
After prepending, every sub-chunk embeds: `"Obligations of promoter\n[sub-clause text]"`.

Note: the **reranker** still receives raw `chunk.text` (without the title prefix),
so it scores on the actual legal text content.

### 2.5 Embedding Model

**Model:** `nomic-embed-text-v1.5` via `sentence-transformers` (local, no Ollama required)
**Dimension:** 768
**Asymmetric prefixes** (MTEB/nomic-embed requirement):

```
Ingestion:  "search_document: {section_title}\n{text}"
Query:      "search_query: {rewritten_query}"
```

Using the wrong prefix at query time causes ~10–15% recall degradation.

**Truncation guard:**
```python
MAX_EMBED_CHARS = 4000   # ~1000 tokens
text = text[:MAX_EMBED_CHARS]   # BEFORE adding prefix
prefixed = f"search_document: {text.strip()}"
```

---

## 3. Query Pipeline

### 3.1 Query Classification and Rewriting

`agent/nodes.py::classifier_node` calls an LLM with `CLASSIFIER_PROMPT` to produce:

```json
{
  "query_type": "fact_lookup | cross_reference | temporal | penalty_lookup | conflict_detection",
  "rewritten_query": "expanded query for better retrieval"
}
```

**Classification rules (first match wins):**

| Type | Trigger | Retrieval Route |
|---|---|---|
| `conflict_detection` | Keywords: conflict, contradict, inconsistent, override, vs | `hybrid_retrieval` |
| `penalty_lookup` | Fine, jail, imprisonment, consequences | `graph_retrieval` |
| `temporal` | Timeline, deadline, days, months, period, within, stage-wise | `graph_retrieval` |
| `cross_reference` | Explicit section number ("Section 18", "Rule 3") | `graph_retrieval` |
| `fact_lookup` | Everything else | `vector_retrieval` |

**Query rewriting** expands the query to include legal keywords that appear in source text.
Key use case: temporal queries. "What is the timeline for project registration?" becomes
`"grant or reject registration within thirty days deemed registered period"` — now FTS
can match Section 5 which uses exactly those words.

**Fallback:** if classifier fails to parse JSON, defaults to `fact_lookup` with original query.

### 3.2 LLM Routing and Fallback Chain

All LLM calls go through `_llm_call()` which tries models in order:

```
1. gemini/gemini-2.5-flash-lite   (Gemini API, GEMINI_API_KEY)
2. openrouter/meta-llama/llama-3.3-70b-instruct:free  (OpenRouter, OPENROUTER_API_KEY)
3. groq/llama-3.3-70b-versatile   (Groq, GROQ_API_KEY)
```

**Gemini temperature quirk:** Gemini 3.x models degrade below `temperature=1.0`. The
fallback chain auto-sets `effective_temp = 1.0 if "gemini-3" in model else 0.0`.

All legal reasoning uses `temperature=0.0` — determinism over creativity.

---

## 4. Hybrid Retrieval — `_rrf_retrieve()`

The core retrieval function is `_rrf_retrieve()` (shared across all retrieval nodes).
It combines **three retrieval signals** into one ranked list:

### 4.1 Vector Similarity Search

```python
vector_results = await VectorStore.similarity_search(
    query_embedding=embed_query(rewritten_query),
    top_k=top_k * 3,    # fetch 3× to give RRF enough candidates
    jurisdiction=filter
)
```

**Index:** HNSW with cosine similarity on 768-dim vectors.
**Strengths:** Catches semantic matches — "penalties for building without approval" matches
"consequence of non-registration" even without keyword overlap.
**Weakness:** Embeddings for sub-clauses of large sections lose their section context.
Fixed by section title prepending during ingestion (§2.4).

### 4.2 Full-Text Search (PostgreSQL `tsvector`)

```python
fts_results = await VectorStore.full_text_search(
    query=rewritten_query,
    top_k=top_k * 2,
    jurisdiction=filter
)
```

**Query operator:** `websearch_to_tsquery` in OR mode — each word becomes an independent
FTS term. Changed from `plainto_tsquery` (AND-mode) because legal queries contain
both relevant and irrelevant words; AND-mode required all words to match, excluding
relevant sections that only matched most words.

**Strengths:** Exact entity matches — "Section 59" or "promoter" finds the right chunks
even when semantic similarity is ambiguous.
**Weakness:** Misses synonyms, paraphrased text, and queries whose words don't appear
verbatim in the legal text.

### 4.3 Reciprocal Rank Fusion (RRF)

`_rrf_merge()` merges vector and FTS results:

```python
RRF_K = 60   # standard constant — higher = smoother blending

rrf_score(chunk) = 1/(K + rank_in_vector) + 1/(K + rank_in_fts)
```

Chunks appearing in **both** result sets score highest (both terms add). A chunk at
rank 1 in vector but absent from FTS scores `1/(60+1) ≈ 0.016`. A chunk at rank 3
in both scores `1/63 + 1/63 ≈ 0.032` — higher than vector-only rank 1.

**Effect:** Sections that both semantically match AND contain the right legal keywords
float to the top. Pure vector or pure FTS results that lack overlap with the other
signal drop naturally.

### 4.4 Section Family Expansion

After RRF merge, the top-3 results trigger **family expansion**:

```python
for rc in merged[:3]:
    sid = rc.chunk.section_id         # e.g. "5(4)"
    base_sid = re.sub(r'\([^)]*\)$', '', sid).strip()  # → "5"
    for expand_sid in {sid, base_sid}:
        family = await VectorStore.get_section_family(
            section_id=expand_sid, jurisdiction=jur
        )
        # adds parent section + all sibling sub-sections
```

`get_section_family` queries: `section_id = '5'` OR `section_id LIKE '5(%'` — returns
the base section plus all sub-sections (`5(1)`, `5(2)`, `5(3)`, `5(4)`, `5(5)`).

**Guard:** if `section_id` already contains `(`, expansion is skipped to avoid double-expanding
sub-sections. The code strips the parenthetical suffix first (`"5(4)"` → `"5"`) before calling
expansion, so sub-sections found by RRF still trigger their parent section's family.

**Why this matters:** Large sections (S.11 "Obligations of promoter") are split into 10+
sub-chunks. If only `S.11(3)` appears in the RRF top-10, family expansion pulls in
`S.11` (main), `S.11(1)`, `S.11(2)` etc. — the full context the generator needs.

**Cap:** `_MAX_VECTOR_EXPANDED = 40` — prevents excessive reranker load.

---

## 5. Graph-Based Retrieval

Used for `cross_reference`, `penalty_lookup`, `temporal` query types.

### 5.1 Section ID Extraction

```python
section_pattern = re.compile(r'\b(?:section|sec\.?|s\.)\s*(\d+[A-Z]?)\b', re.IGNORECASE)
rule_pattern    = re.compile(r'\bRule\s+(\d+[A-Z]?)\b', re.IGNORECASE)
```

If no section ID found in query, falls back to `_rrf_retrieve()` (hybrid retrieval).

### 5.2 Neo4j Traversal

For each detected section ID, across all relevant jurisdictions:

```
1. Source section chunks        — exact section_id match → is_pinned=True
2. REFERENCES outgoing          — sections the source cites (depth=2)
3. REFERENCES incoming          — sections that cite the source
4. DERIVED_FROM outgoing        — Act sections this State Rule derives from
5. DERIVED_FROM incoming        — State Rule sections implementing this Act section
```

**DERIVED_FROM** is the cross-jurisdiction link. When user asks about Maharashtra Rule 3,
traversal follows `DERIVED_FROM` to RERA Act Section 4 — so both jurisdictions appear
in context without needing to explicitly mention both.

### 5.3 Pinning Rule

Exact `section_id` matches get `is_pinned=True`. Pinned chunks are prepended to
reranker output — they bypass score-based ordering. This prevents a highly relevant
source section from being demoted if the cross-encoder scores some related section higher.

---

## 6. Reranking

### 6.1 FlashRank Cross-Encoder

`retrieval/reranker.py` uses **FlashRank** with `rank-T5-flan`:

```python
from flashrank import Ranker, RerankRequest
ranker = Ranker(model_name="rank-T5-flan", cache_dir=".cache/flashrank")

passages = [{"text": c.chunk.text} for c in non_pinned]
request = RerankRequest(query=state["query"], passages=passages)
results = ranker.rerank(request)
```

The cross-encoder reads the **query** and **chunk text** together in a single forward pass,
producing a relevance score (0.0–1.0). This is far more accurate than cosine similarity
but ~10× slower — hence the `_MAX_VECTOR_EXPANDED=40` cap upstream.

**Inputs:** raw `chunk.text` — NOT the section-title-prefixed version used at embedding time.
The title prefix is for embedding quality; the cross-encoder scores on the actual legal text.

### 6.2 Score Gap Filter — `_apply_score_gap()`

```python
def _apply_score_gap(chunks, gap=0.6):
    """Drop chunks after the first score cliff ≥ gap."""
    for i in range(1, len(chunks)):
        if chunks[i - 1].rerank_score - chunks[i].rerank_score >= gap:
            return chunks[:i]
    return chunks
```

**Purpose:** Stop including chunks once there is a large quality drop. A top chunk at
score 0.98 followed by a chunk at 0.30 represents a genuine relevance cliff — the 0.30
chunk is noise, not context.

**Threshold values:**
- `reranker_score_threshold = 0.1` — minimum score to enter candidate pool (filters near-zero)
- `reranker_score_gap = 0.6` — cliff threshold

**Tuning history:** Original values were `threshold=0.3, gap=0.35`. The gap of 0.35
was too aggressive — a chunk scoring 0.52 after one scoring 0.88 was cut (gap=0.36 ≥ 0.35),
leaving only 1 context chunk for the generator. Increasing to 0.6 keeps reasonable
secondary chunks while still cutting genuine noise.

### 6.3 Final Context Assembly

```python
slots_for_ranked = max(0, 5 - len(pinned))
reranked = pinned + gap_filtered[:slots_for_ranked]
```

Maximum 5 context chunks: pinned first, then reranker-scored chunks. This is the input
to the generator prompt as numbered blocks `[1]`, `[2]`, ..., `[5]`.

---

## 7. Generation

### 7.1 Generator Prompt Structure

`prompts/generator.py` — the generator is instructed to:

1. Open with plain-English summary (1–3 sentences, no jargon)
2. Bulleted key points using only information from provided context
3. Note connections, contradictions, exceptions
4. Close with section references

**Output schema:**
```json
{
  "answer": "<markdown>",
  "confidence_score": 0.0-1.0,
  "cited_chunks": [1, 3],
  "amendment_notice": null,
  "conflict_warnings": []
}
```

### 7.2 Grounding Rules

Critical rules that prevent hallucination on sparse contexts:

- **Only use context** — never external legal knowledge or training data
- **Sparse context handling** — if context lacks the answer, say "Based on the available context: [X]" and explicitly note "The context does not contain sufficient information to determine [Y]"
- **Conflict detection grounding** — only assert a conflict exists if BOTH conflicting provisions appear in the context. If only one side is present: "The context contains [jurisdiction X's position] but does not include [jurisdiction Y's position] to confirm or deny a conflict"
- **No invented citations** — never invent section numbers, legal provisions, or figures not in context
- **Confidence calibration** — if `cited_chunks` is empty, `confidence_score` must be < 0.3

### 7.3 Tone Hints per Query Type

Each query type receives a tone instruction injected into the generator system prompt:

| Type | Tone hint |
|---|---|
| `fact_lookup` | Give a direct answer and include one helpful analogy |
| `penalty_lookup` | Lead with the consequence, then explain why it applies |
| `cross_reference` | Explain the connection between sections as a narrative |
| `conflict_detection` | Only flag contradiction if BOTH sides appear in context; never infer precedence |
| `temporal` | Explain what changed, when, and why it matters |

### 7.4 Citation Extraction

The generator returns 1-based indices (`cited_chunks: [1, 3]`) into the numbered context
blocks. The agent extracts `CivicSetuResponse.citations` from only those positions —
not all retrieved chunks. This ensures citations are grounded in what the LLM actually read.

---

## 8. Validation

### 8.1 Validator Node

`nodes.py::validator_node` sends the answer back to an LLM with the same numbered context
blocks the generator used, asking:

- Is each claim in the answer supported by the provided context?
- Confidence score: 0.0–1.0
- `hallucination_flag`: True if any claim is not grounded

**Context format matters:** The validator must receive context in the same `[N] doc — section: title\ntext`
format the generator used. Early versions passed raw `chunk.text` — the validator couldn't
match "Section 11(1)" claims to source chunks that never contained "Section 11(1)" in their text,
causing spurious `hallucination=True` flags and retry loops.

### 8.2 Retry Logic

```
confidence >= 0.5 AND not hallucinated → END
(confidence < 0.5 OR hallucinated) AND retry_count < 2 → retry → classifier
retry_count >= 2 → END (low confidence answer)
```

Max 2 retries. On retry, the classifier runs again with the original query — a different
model in the fallback chain may produce a better rewrite.

### 8.3 Output Guardrails

`guardrails/output_guard.py`:
- Confidence floor: 0.30 — below this threshold, returns `InsufficientInfoResponse`
- Disclaimer injection: always appended to every response
- PII check not repeated (done at input)

---

## 9. RAGAS Evaluation Pipeline

### 9.1 Architecture — Two-Phase with Caching

Phase 1 (slow, ~2–3 min): invoke the RAG graph for every query → save to `eval_phase1_results.json`
Phase 2 (fast, ~1 min): score cached results with RAGAS → save to `eval_results.json`

This separation allows iterating on RAGAS scoring (prompt changes, judge model changes)
without re-invoking the full RAG pipeline. Phase 1 cache is invalidated manually via `make eval-reset`.

```bash
make eval-smoke-p1   # Phase 1: run graph for 5-row smoke dataset
make eval-smoke-p2   # Phase 2: score the 5 cached rows
make eval-reset      # Clear both caches (re-runs everything)
make eval-p1         # Phase 1: full 31-row golden dataset
make eval-p2         # Phase 2: score all 31 rows
```

### 9.2 Golden Dataset

`eval/golden_dataset.jsonl` — 31 rows across 5 jurisdictions and all 5 query types.

Each row:
```json
{
  "id": "CENTRAL-FACT-001",
  "jurisdiction": "CENTRAL",
  "query_type": "fact_lookup",
  "query": "What are the obligations of a promoter under RERA?",
  "ground_truth": "Under Section 11...",
  "expected_section_ids": ["Section 11", "Section 4"],
  "tags": ["promoter", "obligations"]
}
```

**Coverage:**

| Jurisdiction | fact | xref | temporal | penalty | conflict | Total |
|---|---|---|---|---|---|---|
| CENTRAL | 2 | 1 | 1 | 1 | 1 | 6 |
| MAHARASHTRA | 1 | 1 | 1 | 1 | 1 | 5 |
| UTTAR_PRADESH | 1 | 1 | 1 | 1 | 1 | 5 |
| KARNATAKA | 1 | 1 | 1 | 1 | 1 | 5 |
| TAMIL_NADU | 1 | 1 | 1 | 1 | 1 | 5 |
| MULTI (null jur) | 1 | 1 | 1 | 1 | 1 | 5 |
| **Total** | | | | | | **31** |

### 9.3 RAGAS Metrics

Three metrics computed per row:

**Faithfulness** — are all claims in the answer grounded in the retrieved contexts?
```
faithfulness = (claims supported by context) / (total claims in answer)
```
Score 1.0 = fully grounded. Score 0.0 = complete hallucination.

**Answer Relevancy** — does the answer actually address the question?
```
answer_relevancy = similarity(answer, question) averaged over generated question variants
```
RAGAS generates N paraphrased questions from the answer, embeds them, and measures cosine
similarity to the original question. High score = answer stays on topic.

**Context Precision** — are the retrieved contexts ranked in order of usefulness?
```
context_precision = precision@k averaged over ranks
```
Each context is scored by a judge LLM: "Is this context useful for answering the question
given the ground truth?" Contexts marked useful at rank 1 weight higher than rank 3.
A precision of 0.0 means none of the retrieved contexts were useful — retrieval failure.

### 9.4 Judge LLM

RAGAS uses an LLM judge for faithfulness and context precision verdicts. Supported providers:

```bash
JUDGE_PROVIDER=groq   JUDGE_MODEL=llama-3.3-70b-versatile  # default
JUDGE_PROVIDER=gemini JUDGE_MODEL=gemini/gemma-4-31b-it
JUDGE_PROVIDER=osmapi JUDGE_MODEL=qwen3.5-397b-a17b
```

Judge API key can be rotated independently of the graph LLM:
```bash
JUDGE_GEMINI_API_KEY=<alternate_key> make eval-smoke-p2
```

**Rate limiting:** RAGAS calls the judge LLM once per context per row. With 3 contexts × 31 rows,
that is ~93 judge calls. Free-tier APIs (Gemini 2.0 Flash: 10 RPM, 200 RPD) may be exhausted.
`PHASE2_DELAY_SEC=5` adds a delay between rows; `PHASE2_MAX_RETRIES=4` retries on 429.

### 9.5 Context Trimming for Judge

Raw legal chunks can be 1500 chars. RAGAS judge prompt with 5 chunks × 1500 chars exceeds
context limits. Trimming applied before Phase 2:

```
RAGAS_MAX_CONTEXTS = 3      — only score top-3 contexts (not all 5)
RAGAS_CONTEXT_CHAR_LIMIT = 800    — trim each context to 800 chars
RAGAS_ANSWER_CHAR_LIMIT = 800     — trim answer to 800 chars
RAGAS_REFERENCE_CHAR_LIMIT = 600  — trim ground truth to 600 chars
```

### 9.6 Current Eval Results (5-row smoke, April 2026)

| Row | Query Type | Faithfulness | Relevancy | Prec | Root Issue |
|---|---|---|---|---|---|
| CENTRAL-FACT-001 | fact_lookup | 0.50 | ~0.80 | 0.00 | S.11 main demoted by reranker; sub-sections rank higher |
| CENTRAL-FACT-002 | fact_lookup | 0.62 | ~0.82 | 0.33 | S.19 at rank 3; judge awards partial credit |
| CENTRAL-XREF-001 | cross_reference | 0.50 | ~0.80 | 1.00 | S.18 at rank 1; garbage financial doc removed |
| CENTRAL-CONF-001 | conflict_detection | 0.62 | ~0.85 | 0.00 | Only central RERA retrieved; state rule missing |
| CENTRAL-TEMP-001 | temporal | 1.00 | ~0.72 | 0.00 | Generator says "insufficient info" — S.5 not in top-5 |

Overall: faithfulness=0.650, context_precision=0.267, pass_rate=0% (threshold=0.7)

---

## 10. Known Failure Modes

### 10.1 Sub-Section Demotion (FACT-001)

**Symptom:** S.11 main ("Obligations of promoter") gets demoted by the cross-encoder even
though it's the target section. The cross-encoder scores S.11(2) and S.11(3) higher because
their body text contains specific duty clauses more directly relevant to the query.

**Why:** S.11 main's first sentence is procedural ("log in to the website with credentials..."),
not about obligations. Cross-encoder scores it lower.

**Mitigation options:**
- Pin parent section when any sub-section is retrieved (aggressive pinning)
- Rewrite the section text during ingestion to deduplicate procedural preambles
- Ask the generator to synthesize from all S.11 sub-sections even without S.11 main

### 10.2 Single-Jurisdiction Retrieval for Conflict Detection (CONF-001)

**Symptom:** "How do state RERA rules differ from central RERA on registration?" retrieves
central RERA chunks but not the relevant state rule chunk. Faithfulness=0.62 (generator
uses hedging language), context_precision=0.00 (state rule not in context at all).

**Root cause:** `conflict_detection` routes to `hybrid_retrieval`, which does a single
RRF search. A query about "state vs central" is ambiguous — the embedding similarity
pulls toward whichever jurisdiction dominates the corpus.

**Planned fix:** Multi-query decomposition — run two separate RRF queries (one biased
toward central, one toward each state) and merge before reranking.

### 10.3 Temporal Query FTS Miss (TEMP-001)

**Symptom:** Query "What is the timeline for project registration?" should retrieve S.5
(30-day rule). S.5 IS found by RRF and scores 0.99 on the cross-encoder, but sometimes
doesn't appear in the final top-5 due to S.4 family expansion filling pool slots.

**Root cause:** S.4(14) appears in the RRF top-10, which triggers expansion of the entire
S.4 family (14 sub-sections). These fill the `_MAX_VECTOR_EXPANDED=40` pool before
reranking, and after gap filtering S.4 sub-sections at low scores may still occupy slots
that S.5 would have used.

**Partial fix applied:** Classifier now rewrites temporal queries to include legal time
keywords ("grant or reject registration within thirty days deemed registered") — this
improves FTS recall of S.5.

---

## 11. Configuration Reference

All RAG parameters in `config/settings.py`:

| Parameter | Default | Effect |
|---|---|---|
| `reranker_model` | `rank-T5-flan` | Cross-encoder model |
| `reranker_score_threshold` | `0.1` | Min score to enter candidate pool |
| `reranker_score_gap` | `0.6` | Score cliff threshold for gap filter |
| `embedding_model` | `nomic-embed-text` | Sentence-transformers model name |
| `embedding_dimension` | `768` | pgvector index dimension |

Environment overrides for eval:

| Env Var | Effect |
|---|---|
| `EVAL_LIMIT` | Limit eval to first N rows (default: all) |
| `EVAL_PHASE` | "1" = phase 1 only, "2" = phase 2 only |
| `JUDGE_PROVIDER` | `groq` / `gemini` / `osmapi` |
| `JUDGE_MODEL` | Judge LLM model name |
| `JUDGE_GEMINI_API_KEY` | Alternate Gemini key for judge only |
| `PASS_THRESHOLD` | RAGAS pass threshold (default: 0.7) |
| `PHASE2_DELAY_SEC` | Delay between judge calls (default: 5) |
| `RAGAS_MAX_CONTEXTS` | Max contexts passed to judge (default: 3) |
| `RAGAS_CONTEXT_CHAR_LIMIT` | Char trim per context (default: 800) |

---

## 12. Implementation Checklist

When adding a new jurisdiction or document corpus:

- [ ] Add `DocumentSpec` to `document_registry.py` with correct `max_pages` to exclude forms
- [ ] Verify PDF is born-digital (not scanned) — use `pdffonts` or PyMuPDF `get_text()` check
- [ ] Run `make ingest --jurisdiction <JUR>` — check `fallback_paragraph_chunking` logs
- [ ] Verify chunk count in PostgreSQL (`SELECT COUNT(*), jurisdiction FROM legal_chunks GROUP BY 2`)
- [ ] Run `make eval-smoke-p1` and inspect retrieved contexts for new jurisdiction queries
- [ ] Add rows to `eval/golden_dataset.jsonl` covering all 5 query types for new jurisdiction
- [ ] Run `make eval-p1 && make eval-p2` — check context_precision for new rows ≥ 0.40
