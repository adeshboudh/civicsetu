.PHONY: help install dev serve ingest lint format typecheck test clean docker-up docker-down

help:
	@echo "CivicSetu — available commands:"
	@echo "  make install      Install all dependencies"
	@echo "  make dev          Install with dev + eval extras"
	@echo "  make serve        Start FastAPI server (hot reload)"
	@echo "  make ingest       Run Phase 0 ingestion (RERA Act 2016)"
	@echo "  make lint         Run ruff linter"
	@echo "  make format       Run ruff formatter"
	@echo "  make typecheck    Run mypy"
	@echo "  make test         Run test suite"
	@echo "  make docker-up    Start Postgres + pgvector"
	@echo "  make docker-down  Stop all containers"
	@echo "  make clean        Remove __pycache__ and .pyc files"

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
	uv run python scripts/ingest_phase0.py

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

docker-up:
	cd infra && docker compose up -d
	@echo "Waiting for Postgres healthcheck..."
	@sleep 5
	@docker compose -f infra/docker-compose.yml ps

docker-down:
	cd infra && docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
