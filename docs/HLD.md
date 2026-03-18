# CivicSetu — High Level Design (HLD)

**Version:** 0.3.0 — Phase 2 Complete
**Last Updated:** March 2026
**Status:** Phase 2 Complete — Multi-jurisdiction ingestion live

---

## 1. System Overview

CivicSetu is an open-source RAG (Retrieval-Augmented Generation) system that answers
plain-English questions about Indian civic and legal documents with accurate citations,
amendment tracking, and conflict detection between laws.

**Target Users:** Indian citizens, lawyers, homebuyers, activists navigating RERA, RTI,
labor law, GST compliance, and other civic frameworks.

**Current Scope:** RERA Act 2016 (Central) + Maharashtra Real Estate Rules 2017.

---

## 2. Architecture Overview

```

┌──────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                              │
│              HTTP REST (FastAPI) — /api/v1/query                 │
└────────────────────────────┬─────────────────────────────────────┘
							 │
┌────────────────────────────▼─────────────────────────────────────┐
│                     LANGGRAPH AGENT                              │
│                                                                  │
│  [Classifier] → [Vector Retrieval] → [Reranker]                  │
│       ↑         [Graph Retrieval]  ↗                             │
│  [Retry]  ←  [Validator] ← [Generator]                           │
└────────────────────────────┬─────────────────────────────────────┘
							 │
		  ┌──────────────────┼──────────────────────┐
		  │                  │                      │
  ┌───────▼──────┐   ┌───────▼─────────┐    ┌───────▼────────┐
  │  pgvector    │   │   Neo4j         │    │   PostgreSQL   │
  │  (vectors)   │   │   (graph)       │    │   (metadata)   │
  │  Phase 0     │   │   Phase 1       │    │   Phase 0      │
  └───────┬──────┘   └───────┬─────────┘    └───────┬────────┘
          │                  │                      │
          └──────────────────┴──────────────────────┘
							 │
┌────────────────────────────▼─────────────────────────────────────┐
│                    INGESTION PIPELINE                            │
│  Download → Parse → Chunk → Enrich → Embed → Store               │
│  document_registry.py — single source of truth for all doc URLs  │
└──────────────────────────────────────────────────────────────────┘

```

---

## 3. Two Pipelines

### 3.1 Ingestion Pipeline (Offline)

Runs once per document. Triggered via `make ingest` or `POST /api/v1/ingest`.

```

PDF URL (from document_registry.py)
→ Downloader        (httpx, cached locally with MD5 check)
→ PDFParser         (PyMuPDF, text extraction, scanned page detection)
→ LegalChunker      (multi-format regex: Act + Rule boundary detection)
→ MetadataExtractor (dates, cross-references, amendment signals)
→ Embedder          (nomic-embed-text via Ollama, MAX_EMBED_CHARS=6000 guard)
→ RelationalStore   (PostgreSQL — documents + legal_chunks tables)
→ VectorStore       (pgvector — HNSW index, cosine similarity)
→ GraphStore        (Neo4j — Document + Section nodes + edges)

```

### 3.2 Query Pipeline (Online, per-request)

Triggered on every `POST /api/v1/query`.

```

User Query
→ Input Guardrails  (PII + off-topic filter)
→ Classifier Node   (LLM — query_type + rewritten_query)
→ Vector Retrieval  (pgvector cosine search, top_k chunks)  ← fact_lookup
→ Graph Retrieval   (Neo4j, REFERENCES traversal, depth=2)  ← cross_reference / penalty / temporal
Fallback: vector retrieval when no section ID in query
→ Reranker          (FlashRank ms-marco-MiniLM-L-12-v2, cross-encoder)
→ Generator Node    (LLM — structured JSON answer with citations)
→ Validator Node    (LLM — hallucination + confidence check)
→ Output Guardrails (faithfulness check + disclaimer injection)
→ CivicSetuResponse (answer + citations + confidence + disclaimer)

```

---

## 4. Component Responsibilities

| Component          | Responsibility                              | Technology                      |
|--------------------|---------------------------------------------|---------------------------------|
| DocumentRegistry   | Centralised doc URL + metadata management   | Python dataclass                |
| PDFParser          | Text extraction from PDFs                   | PyMuPDF                         |
| LegalChunker       | Multi-format section-boundary splitting     | Regex (Act + Rule patterns)     |
| MetadataExtractor  | Date, reference, amendment extraction       | Regex                           |
| Embedder           | Dense vector generation + truncation guard  | nomic-embed-text (Ollama)       |
| VectorStore        | Semantic similarity search                  | pgvector + HNSW                 |
| GraphStore         | Section relationship traversal              | Neo4j Community                 |
| RelationalStore    | Metadata persistence + chunk storage        | PostgreSQL + SQLAlchemy         |
| LangGraph Agent    | Query orchestration state machine           | LangGraph                       |
| LiteLLM Gateway    | LLM provider fallback routing               | LiteLLM                         |
| FastAPI            | HTTP API layer                              | FastAPI + Uvicorn               |
| FlashRank          | Cross-encoder reranking                     | ONNX local model                |

---

## 5. LLM Fallback Chain

```

Primary  → gemini/gemini-3.1-flash-lite-preview  (Gemini API)
Backup 1 → groq/llama-3.3-70b-versatile          (Groq API)
Backup 2 → openrouter/:free models               (OpenRouter)
Local    → ollama/mistral                         (local, offline)

```

All routing handled by LiteLLM. Model swap = config change only.

---

## 6. Data Flow: Query to Response

```

Input:  {"query": "What are builder obligations under Section 18?"}

Step 1  Classify    → query_type=cross_reference (explicit section number detected)
Step 2  Graph       → traverse Section 18 node, incoming + outgoing REFERENCES edges
Step 2b Fallback    → vector retrieval if graph returns 0 results
Step 3  Rerank      → cross-encoder scores, top 5 ordered
Step 4  Generate    → LLM produces JSON with answer + citations
Step 5  Validate    → hallucination check, confidence score
Step 6  Respond     → CivicSetuResponse with citations + disclaimer

Output: {
"answer": "Under Section 18(1)...",
"citations": [{"section_id": "18", "doc_name": "RERA Act 2016", ...}],
"confidence_score": 0.95,
"confidence_level": "high",
"disclaimer": "This is AI-generated information..."
}

```

---

## 7. Phase Roadmap

| Phase | Scope                                          | Status          |
|-------|------------------------------------------------|-----------------|
| 0     | RERA Act 2016, vector RAG, FastAPI             | ✅ Complete     |
| 1     | Neo4j graph, cross-reference queries           | ✅ Complete     |
| 2     | MahaRERA Rules 2017, multi-jurisdiction        | ✅ Complete     |
| 3     | DERIVED_FROM edges, conflict detection         | Next            |
| 4     | Multi-state expansion (UP, TN, Karnataka RERA) | Planned         |
| 5     | Open-source SaaS, UI, public API               | Planned         |

---

## 8. Non-Functional Requirements

| Requirement        | Target                               | Current Status                  |
|--------------------|--------------------------------------|---------------------------------|
| Response latency   | < 10s per query                      | ~5–8s (local embedding)         |
| Citation accuracy  | 100% — never answer without citation | Enforced by schema              |
| Hallucination rate | < 5%                                 | Validator node + confidence gate|
| Cost               | $0 for dev/staging                   | All free tier                   |
| Portability        | Runs on any machine with Docker      | Docker Compose                  |