---
title: CivicSetu
emoji: 🏛️
colorFrom: blue
colorTo: purple
sdk: docker
sdk_version: "default"
app_file: app.py
pinned: false
---

# CivicSetu

**Live:** [https://civicsetu-two.vercel.app](https://civicsetu-two.vercel.app)

Open-source RAG system for querying Indian civic and legal documents — with accurate
citations, cross-reference traversal, and conflict detection between laws.

**Current status:** Phase 6 complete — 5-jurisdiction RERA coverage (Central + MH + UP + KA + TN),
cross-jurisdiction graph edges live, 12/12 E2E passing, Next.js frontend deployed on Vercel.

---

## What it does

Ask a plain-English question about RERA. Get a cited, structured answer with section
references, confidence score, and a legal disclaimer — grounded in real legal text.

```

Query:  "Which state rules implement section 9 of RERA on agent registration?"

Answer: "Section 9 of the RERA Act 2016 governs agent registration at the central level.
Rule 11 of Maharashtra Rules 2017 and Rule 8 of Karnataka RERA Rules derive
from Section 9, specifying application procedures and timelines..."

Citations: [Section 9, RERA Act 2016], [Rule 11, Maharashtra Rules 2017],
[Rule 8, Karnataka RERA Rules]
Confidence: 0.96 (high)

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
- **Neo4j** — section graph traversal (cross-references, DERIVED_FROM edges)
- **PostgreSQL** — full chunk text + metadata

Full design: [HLD.md](docs/HLD.md) | [LLD.md](docs/LLD.md)

---

## Quickstart

### Prerequisites

- Docker + Docker Compose
- `uv` package manager
- One of: Gemini API key (free tier) or Groq API key (free tier)

> **No Ollama required.** Embeddings run locally via `sentence-transformers`.
> First run downloads `nomic-embed-text-v1.5` (~550MB) from HuggingFace and caches it.

### Setup

```bash
# 1. Clone and install
git clone https://github.com/adeshboudh/civicsetu.git && cd civicsetu
make install

# 2. Configure secrets
cp .env.example .env
# Set GEMINI_API_KEY and/or GROQ_API_KEY — everything else has working defaults

# 3. Start infrastructure
make docker-up

# 4. Ingest all 5 jurisdictions
make ingest

# 5. Start the API
make serve
```

**Full docs:** [HLD](docs/HLD.md) | [LLD](docs/LLD.md)

## Production

- **Frontend:** [Vercel](https://civicsetu-two.vercel.app) — Next.js 15 App Router
- **API:** [Hugging Face Spaces](https://huggingface.co/spaces/adesh01/civicsetu) — FastAPI + Docker + 550MB model baked in
- **PostgreSQL + pgvector:** [Neon](https://neon.tech) — 1203 chunks
- **Neo4j:** [AuraDB Free](https://neo4j.com/cloud/aura) — 2090 sections, 2321 edges
- **LLM:** LiteLLM (Gemini → Groq → OpenRouter)

### 6. Query

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the penalties for a promoter who delays possession?"}'
```

> First request will be slow (~30–45s) while the embedding model loads into memory.
> Subsequent requests run at 5–15s.

### Other useful commands

```bash
make e2e        # Run 12-case E2E benchmark across all 5 jurisdictions
make test       # Run unit tests
make lint       # Ruff linter
make typecheck  # mypy

make ingest --jurisdiction MAHARASHTRA  # Re-ingest a single jurisdiction
make docker-down                        # Tear down containers
```

---

## Documents ingested

| Document                           | Jurisdiction  | Sections |
| ---------------------------------- | ------------- | -------- |
| RERA Act 2016                      | Central       | 224      |
| Maharashtra Real Estate Rules 2017 | Maharashtra   | 214      |
| UP RERA Rules 2016                 | Uttar Pradesh | 170      |
| UP RERA General Regulations 2019   | Uttar Pradesh | 85       |
| Karnataka RERA Rules 2017          | Karnataka     | 235      |
| Tamil Nadu RERA Rules 2017         | Tamil Nadu    | 157      |

Total chunks: 1203.
Graph: 2090 Section nodes, 1297 HAS_SECTION edges, 933 REFERENCES edges, 91 DERIVED_FROM edges.


---

## Tech stack

| Layer | Technology |
| :-- | :-- |
| API | FastAPI + Uvicorn |
| Orchestration | LangGraph StateGraph |
| LLM routing | LiteLLM (Gemini → Groq → OpenRouter) |
| Embeddings | nomic-embed-text-v1.5 via sentence-transformers (local) |
| Vector DB | pgvector + HNSW index |
| Graph DB | Neo4j Community |
| Relational | PostgreSQL + SQLAlchemy |
| Reranker | FlashRank (ms-marco-MiniLM-L-12-v2) |
| PDF parsing | PyMuPDF |


---

## Phase roadmap

| Phase | Scope | Status |
| :-- | :-- | :-- |
| 0 | RERA Act 2016, vector RAG, FastAPI | ✅ Complete |
| 1 | Neo4j graph, cross-reference queries | ✅ Complete |
| 2 | MahaRERA Rules 2017, multi-jurisdiction | ✅ Complete |
| 3 | DERIVED_FROM edges, cross-jurisdiction graph | ✅ Complete |
| 4 | Multi-state expansion (UP, TN, Karnataka) | ✅ Complete |
| 5 | Agent pipeline hardening, E2E test suite | ✅ Complete |
| 6 | Next.js frontend, Vercel deployment, public URL | ✅ Complete |
| 7 | Graph explorer, section content drawer, D3 visualization | ✅ Complete |


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
