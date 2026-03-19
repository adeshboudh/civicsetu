from __future__ import annotations

import asyncio

import structlog
from fastapi import APIRouter, HTTPException, Request

from civicsetu.guardrails.input_guard import InputGuard
from civicsetu.guardrails.output_guard import OutputGuard
from civicsetu.models.enums import QueryType
from civicsetu.models.schemas import (
    CivicSetuResponse,
    InsufficientInfoResponse,
    QueryRequest,
)

log = structlog.get_logger(__name__)
router = APIRouter()


@router.post("/query", response_model=CivicSetuResponse | InsufficientInfoResponse)
async def query_endpoint(request: Request, body: QueryRequest):
    # ── 1. Input guard ────────────────────────────────────────────────────────
    guard_result = InputGuard.check(body.query)
    if not guard_result.is_safe:
        log.info("query_rejected_by_input_guard", reason=guard_result.reason)
        raise HTTPException(status_code=400, detail=guard_result.reason)

    # Use sanitized query downstream (strips whitespace, no PII)
    safe_query = guard_result.sanitized_query

    # ── 2. Build initial state ────────────────────────────────────────────────
    graph = request.app.state.graph

    initial_state = {
        "query": safe_query,
        "session_id": body.session_id,
        "jurisdiction_filter": body.jurisdiction_filter,
        "top_k": body.top_k,
        "query_type": None,
        "rewritten_query": None,
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "raw_response": None,
        "citations": [],
        "confidence_score": 0.0,
        "conflict_warnings": [],
        "amendment_notice": None,
        "retry_count": 0,
        "hallucination_flag": False,
        "error": None,
    }

    # ── 3. Invoke graph ───────────────────────────────────────────────────────
    try:
        result = await asyncio.to_thread(graph.invoke, initial_state)
    except Exception as e:
        log.error("graph_invoke_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

    # ── 4. Output guard ───────────────────────────────────────────────────────
    return OutputGuard.process(result, original_query=body.query)
