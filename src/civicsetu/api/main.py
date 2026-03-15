from __future__ import annotations

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from civicsetu.api.middleware.logging import LoggingMiddleware
from civicsetu.api.routes import health, query
from civicsetu.config.settings import get_settings

log = structlog.get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    log.info("civicsetu_starting", env=settings.api_env)

    # Pre-compile the graph once at startup — not on first request
    from civicsetu.agent.graph import get_compiled_graph
    app.state.graph = get_compiled_graph()
    log.info("langgraph_compiled")

    yield

    log.info("civicsetu_shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="CivicSetu API",
        description="RAG system for Indian civic and legal documents",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if not settings.is_production else [],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.add_middleware(LoggingMiddleware)

    app.include_router(health.router, tags=["health"])
    app.include_router(query.router, prefix="/api/v1", tags=["query"])

    return app


app = create_app()
