.PHONY: help install dev serve ingest lint format typecheck test test-cov e2e eval docker-up docker-down clean frontend-install frontend-dev frontend-build frontend-start frontend-lint frontend-typecheck

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
	@echo "    make eval         Run RAGAS quality benchmark (offline, batched)"
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

eval:
	PYTHONUTF8=1 uv run python scripts/run_eval.py

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
