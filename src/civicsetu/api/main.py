from __future__ import annotations

import asyncio
import sys
import time
from contextlib import asynccontextmanager

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from civicsetu.api.middleware.logging import LoggingMiddleware
from civicsetu.api.routes import health, query, graph
from civicsetu.config.settings import get_settings
from civicsetu.stores.graph_store import close_driver, get_driver

log = structlog.get_logger(__name__)
settings = get_settings()


def _mask_secret(value: str, visible: int = 3) -> str:
    if not value:
        return "NOT_SET"
    if len(value) <= visible * 2:
        return "*" * len(value)
    return f"{value[:visible]}...{value[-visible:]}"


def create_checkpointer():
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("civicsetu.models.schemas", "ChatMessage"),
            ("civicsetu.models.schemas", "Citation"),
            ("civicsetu.models.schemas", "LegalChunk"),
            ("civicsetu.models.schemas", "RetrievedChunk"),
            ("civicsetu.models.enums", "ChunkStatus"),
            ("civicsetu.models.enums", "DocType"),
            ("civicsetu.models.enums", "Jurisdiction"),
            ("civicsetu.models.enums", "QueryType"),
        ]
    )
    return AsyncPostgresSaver.from_conn_string(settings.postgres_conninfo, serde=serde)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    log.info("civicsetu_starting", env=settings.api_env)

    # Determine and log the primary model's masked API key
    model_name = settings.primary_model
    api_key_to_log = "UNKNOWN"
    provider = "Unknown"
    
    if model_name.startswith("gemini/"):
        api_key_to_log = settings.gemini_api_key
        provider = "Gemini"
    elif model_name.startswith("groq/"):
        api_key_to_log = settings.groq_api_key
        provider = "Groq"
    elif model_name.startswith("openrouter/"):
        api_key_to_log = settings.openrouter_api_key
        provider = "OpenRouter"
    elif model_name.startswith("openai/"):
        api_key_to_log = settings.nvidia_api_key
        provider = "NVIDIA"
        
    masked_key = "NOT_SET_OR_TOO_SHORT"
    if api_key_to_log and len(api_key_to_log) > 8:
        masked_key = f"{api_key_to_log[:4]}...{api_key_to_log[-4:]}"

    log.info("primary_model_api_key", provider=provider, model=model_name, api_key_masked=masked_key)
    log.info(
        "neo4j_config_loaded",
        uri=settings.neo4j_uri,
        user_masked=_mask_secret(settings.neo4j_user, visible=2),
        password_present=bool(settings.neo4j_password),
        password_len=len(settings.neo4j_password),
    )

    from civicsetu.agent.graph import get_compiled_graph

    async with create_checkpointer() as checkpointer:
        await checkpointer.setup()
        app.state.checkpointer = checkpointer
        app.state.graph = get_compiled_graph(checkpointer=checkpointer)
        log.info("langgraph_compiled", checkpointing=True)
        await get_driver()
        from civicsetu.retrieval import warm_embedding_model

        warm_start = time.perf_counter()
        await asyncio.to_thread(warm_embedding_model)
        log.info(
            "embedding_model_warmed",
            duration_ms=round((time.perf_counter() - warm_start) * 1000, 2),
        )
        from civicsetu.retrieval.reranker import _get_ranker

        ranker_start = time.perf_counter()
        await asyncio.to_thread(_get_ranker)
        log.info(
            "reranker_warmed",
            duration_ms=round((time.perf_counter() - ranker_start) * 1000, 2),
        )
        yield
        await close_driver()
        log.info("civicsetu_shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="CivicSetu API",
        description="RAG system for Indian civic and legal documents",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.add_middleware(LoggingMiddleware)

    app.include_router(health.router, tags=["health"])
    app.include_router(query.router, prefix="/api/v1", tags=["query"])
    app.include_router(graph.router, prefix="/api/v1", tags=["graph"])

    @app.get("/")
    async def root():
        return {
            "message": "CivicSetu API is running. Frontend is served separately in this deployment.",
            "status": "ok",
        }

    return app

app = create_app()
