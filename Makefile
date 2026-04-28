.PHONY: help install dev serve ingest lint format typecheck test test-cov e2e \
        eval eval-p1 eval-p2 eval-smoke eval-smoke-p1 eval-smoke-p2 \
        eval-collect eval-score eval-score-smoke eval-large eval-reset \
        docker-up docker-down clean \
        frontend-install frontend-dev frontend-build frontend-start frontend-lint frontend-typecheck

help:
	@echo "CivicSetu — available commands:"
	@echo ""
	@echo "  Setup:"
	@echo "    make install      Install all dependencies"
	@echo "    make dev          Install with dev + eval extras"
	@echo ""
	@echo "  Run:"
	@echo "    make docker-up    Start PostgreSQL + pgvector + Neo4j"
	@echo "    make docker-down  Stop all containers"
	@echo "    make serve        Start FastAPI server (hot reload)"
	@echo ""
	@echo "  Data:"
	@echo "    make ingest       Ingest all 5 jurisdictions (Central, MH, UP, KA, TN)"
	@echo ""
	@echo "  Quality:"
	@echo "    make lint         Run ruff linter"
	@echo "    make format       Run ruff formatter"
	@echo "    make typecheck    Run mypy"
	@echo "    make test         Run unit test suite"
	@echo "    make test-cov     Run tests with coverage report"
	@echo "    make e2e          Run 12-case E2E query benchmark"
	@echo "    make eval              Full eval — both phases, default RAG LLMs + Groq judge"
	@echo "    make eval-p1           Phase 1 only: graph invocation → eval_phase1_results.json"
	@echo "    make eval-p2           Phase 2 only: RAGAS scoring from cached phase 1"
	@echo "    make eval-smoke        5-row smoke test — both phases"
	@echo "    make eval-smoke-p1     5-row smoke test — phase 1 only"
	@echo "    make eval-smoke-p2     5-row smoke test — phase 2 only (needs eval-smoke-p1 first)"
	@echo "    make eval-large        Full eval with osmapi qwen3.5-397b graph model"
	@echo "    make eval-reset        Delete phase 1 cache to force fresh graph invocation"
	@echo ""
	@echo "    Judge override:        make eval-p2 JUDGE_PROVIDER=gemini JUDGE_MODEL=gemini/gemini-2.5-flash-lite"
	@echo "    Judge override:        make eval-p2 JUDGE_PROVIDER=osmapi JUDGE_MODEL=qwen3.5-397b-a17b"
	@echo "    Graph model override:  make eval-p1 EVAL_PRIMARY_MODEL=neysa/qwen3.5-122b-a10b"
	@echo ""
	@echo "    make clean        Remove __pycache__ and .pyc files"

install:
	uv sync

dev:
	uv sync --extra dev --extra eval

serve:
	uv run uvicorn civicsetu.api.main:app \
		--host 0.0.0.0 \
		--port 8000 \
		--reload \
		--log-level info

ingest:
	uv run python scripts/ingest.py

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

typecheck:
	uv run mypy src/

test:
	uv run pytest tests/ -v --tb=short

test-cov:
	uv run pytest tests/ --cov=civicsetu --cov-report=term-missing -q

e2e:
	PYTHONUTF8=1 uv run python scripts/test_e2e_queries.py

# ── Eval ──────────────────────────────────────────────────────────────────────
# Phase 1 uses default RAG app LLMs from .env (gemini-2.5-flash-lite chain).
# Override graph model:  make eval-p1 EVAL_PRIMARY_MODEL=qwen3.5-122b-a10b
# Default judge comes from scripts/run_eval.py / .env.
# Override judge provider/model:
#   make eval-p2 JUDGE_PROVIDER=gemini JUDGE_MODEL=gemini/gemini-2.5-flash-lite
#   make eval-p2 JUDGE_PROVIDER=osmapi JUDGE_MODEL=qwen3.5-397b-a17b
#   make eval-p2 JUDGE_PROVIDER=groq JUDGE_MODEL=llama-3.3-70b-versatile
# Gemini judge uses GEMINI_API_KEY_2 for both LLM and embeddings.
# Groq judge uses GROQ_API_KEY_2 (or GROQ_API_KEY / JUDGE_GROQ_API_KEY) for LLM, GEMINI_API_KEY_2 for embeddings.
# osmapi judge uses OSMAPI_API_KEY for LLM, GEMINI_API_KEY_2 for embeddings.
MAKE_JUDGE_ENV = $(if $(JUDGE_PROVIDER),JUDGE_PROVIDER=$(JUDGE_PROVIDER) )$(if $(JUDGE_MODEL),JUDGE_MODEL=$(JUDGE_MODEL) )

# Full eval — both phases
eval:
	PYTHONUTF8=1 $(MAKE_JUDGE_ENV)uv run python scripts/run_eval.py

# Phase 1 only — graph invocation, saves to eval_phase1_results.json
eval-p1:
	PYTHONUTF8=1 EVAL_PHASE=1 uv run python scripts/run_eval.py

# Phase 2 only — RAGAS scoring from cached phase 1
eval-p2:
	@test -f eval_phase1_results.json || (echo "ERROR: eval_phase1_results.json not found — run 'make eval-p1' first" && exit 1)
	PYTHONUTF8=1 EVAL_PHASE=2 $(MAKE_JUDGE_ENV)uv run python scripts/run_eval.py

# 5-row smoke tests
eval-smoke:
	PYTHONUTF8=1 EVAL_LIMIT=5 $(MAKE_JUDGE_ENV)uv run python scripts/run_eval.py

eval-smoke-p1:
	PYTHONUTF8=1 EVAL_PHASE=1 EVAL_LIMIT=5 uv run python scripts/run_eval.py

eval-smoke-p2:
	@test -f eval_phase1_results.json || (echo "ERROR: eval_phase1_results.json not found — run 'make eval-smoke-p1' first" && exit 1)
	PYTHONUTF8=1 EVAL_PHASE=2 EVAL_LIMIT=5 $(MAKE_JUDGE_ENV)uv run python scripts/run_eval.py

# Large graph model via osmapi
eval-large:
	PYTHONUTF8=1 EVAL_PRIMARY_MODEL=qwen3.5-397b-a17b $(MAKE_JUDGE_ENV)uv run python scripts/run_eval.py

eval-reset:
	rm -f eval_phase1_results.json
	@echo "Phase 1 cache cleared — next eval-p1 will re-invoke the graph"

eval-reset-all:
	rm -f eval_phase1_results.json eval_results.json
	@echo "Both caches cleared — next eval will run fully fresh"

# Aliases for backward compat
eval-collect: eval-p1
eval-score: eval-p2
eval-score-smoke: eval-smoke-p2

docker-up:
	cd infra && docker compose up -d
	@echo "Waiting for services to be healthy..."
	@sleep 5
	@docker compose -f infra/docker-compose.yml ps

docker-down:
	cd infra && docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

# Frontend
frontend-install:
	cd frontend && npm install

frontend-dev:
	cd frontend && npm run dev

frontend-build:
	cd frontend && npm run build

frontend-start:
	cd frontend && npm run start

frontend-lint:
	cd frontend && npm run lint

frontend-typecheck:
	cd frontend && npm run typecheck
