# CivicSetu

Open-source RAG system for querying Indian civic and legal documents — with accurate
citations, cross-reference traversal, and conflict detection between laws.

**Current status:** Phase 2 complete — RERA Act 2016 (Central) + Maharashtra Rules 2017.

---

## What it does

Ask a plain-English question about RERA or Maharashtra real estate rules. Get a cited,
structured answer with section references, confidence score, and a legal disclaimer.

```

Query: "What must a promoter disclose before selling a flat?"

Answer: "Under Section 11(3) of RERA Act 2016, a promoter must disclose...
Rule 3(2) of Maharashtra Rules further requires..."

Citations: [Section 11, RERA Act 2016], [Rule 3(2), Maharashtra Rules 2017]
Confidence: 0.95 (high)

```

---

## Architecture

```

FastAPI → LangGraph Agent → pgvector + Neo4j + PostgreSQL
↑
Ingestion Pipeline (PDF → chunks → embeddings → graph)

```

Three stores per query:
- **pgvector** — semantic similarity (fact lookups)
- **Neo4j** — section graph traversal (cross-references, penalties)
- **PostgreSQL** — full chunk text + metadata

Full design: [HLD.md](docs/HLD.md) | [LLD.md](docs/LLD.md)

---

## Quickstart

### Prerequisites

- Docker + Docker Compose
- [Ollama](https://ollama.ai) running locally
- `uv` package manager

### 1. Start infrastructure

```bash
docker compose up -d          # PostgreSQL + pgvector + Neo4j
ollama pull nomic-embed-text  # embedding model
```


### 2. Configure environment

```bash
cp .env.example .env
# Set GEMINI_API_KEY (or GROQ_API_KEY for backup)
# Neo4j and Postgres defaults work out of the box with Docker Compose
```


### 3. Install dependencies

```bash
uv sync
```


### 4. Ingest documents

```bash
uv run python scripts/ingest_phase0.py   # RERA Act 2016
uv run python scripts/ingest_phase2.py   # Maharashtra Rules 2017
```


### 5. Run the API

```bash
uv run uvicorn civicsetu.api.main:app --reload
```


### 6. Query

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the penalties for a promoter who delays possession?"}'
```


---

## Documents ingested

| Document | Jurisdiction | Chunks | Sections |
| :-- | :-- | :-- | :-- |
| RERA Act 2016 | Central | 224 | 92 |
| Maharashtra Real Estate Rules 2017 | Maharashtra | 214 | 44 |


---

## Tech stack

| Layer | Technology |
| :-- | :-- |
| API | FastAPI + Uvicorn |
| Orchestration | LangGraph StateGraph |
| LLM routing | LiteLLM (Gemini → Groq → OpenRouter) |
| Embeddings | nomic-embed-text via Ollama (local) |
| Vector DB | pgvector + HNSW index |
| Graph DB | Neo4j Community |
| Relational | PostgreSQL + SQLAlchemy |
| Reranker | FlashRank (ms-marco-MiniLM-L-12-v2) |
| PDF parsing | PyMuPDF |


---

## Phase roadmap

| Phase | Scope | Status |
| :-- | :-- | :-- |
| 0 | RERA Act 2016, vector RAG, FastAPI | Complete |
| 1 | Neo4j graph, cross-reference queries | Complete |
| 2 | MahaRERA Rules 2017, multi-jurisdiction | Complete |
| 3 | DERIVED_FROM edges, conflict detection | Next |
| 4 | Multi-state expansion (UP, TN, Karnataka) | Planned |
| 5 | Open-source SaaS, UI, public API | Planned |


---

## ADRs

- [ADR 001 — three store architecture](docs/adr/001-three-store-architecture.md)
- [ADR 002 — section boundary chunking](docs/adr/002-section-boundary-chunking.md)
- [ADR 003 — LangGraph over LangChain chains](docs/adr/003-langgraph-over-langchain.md)
- [ADR 004 — Multi-format chunker](docs/adr/004-multi-format-chunker.md)
- [ADR 005 — Document registry](docs/adr/005-document-registry.md)


## Disclaimer

CivicSetu provides AI-generated legal information, not legal advice.
Always verify with a qualified lawyer or the official gazette.
