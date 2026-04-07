from __future__ import annotations

import asyncio
import time
import uuid

import structlog
from fastapi import APIRouter, HTTPException, Request

from civicsetu.guardrails.input_guard import InputGuard
from civicsetu.guardrails.output_guard import OutputGuard
from civicsetu.models.schemas import CivicSetuResponse, ChatMessage, InsufficientInfoResponse, QueryRequest

log = structlog.get_logger(__name__)
router = APIRouter()


@router.post("/query", response_model=CivicSetuResponse | InsufficientInfoResponse)
async def query_endpoint(request: Request, body: QueryRequest):
    guard_result = InputGuard.check(body.query)
    if not guard_result.is_safe:
        log.info("query_rejected_by_input_guard", reason=guard_result.reason)
        raise HTTPException(status_code=400, detail=guard_result.reason)

    safe_query = guard_result.sanitized_query
    session_id = body.session_id or str(uuid.uuid4())
    graph = request.app.state.graph
    config = {"configurable": {"thread_id": session_id}}

    initial_state = {
        "query": safe_query,
        "session_id": session_id,
        "jurisdiction_filter": body.jurisdiction_filter,
        "top_k": body.top_k,
        "messages": [ChatMessage(role="user", content=safe_query)],
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

    try:
        invoke_start = time.perf_counter()
        result = await asyncio.to_thread(graph.invoke, initial_state, config)
        log.info(
            "graph_invoke_complete",
            route="query",
            duration_ms=round((time.perf_counter() - invoke_start) * 1000, 2),
        )
    except Exception as e:
        log.error("graph_invoke_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

    raw_response = result.get("raw_response")
    if raw_response:
        try:
            update_start = time.perf_counter()
            await asyncio.to_thread(
                graph.update_state,
                config,
                {"messages": [ChatMessage(role="assistant", content=raw_response)]},
            )
            log.info(
                "graph_update_state_complete",
                route="query",
                duration_ms=round((time.perf_counter() - update_start) * 1000, 2),
            )
        except Exception as e:
            log.warning("graph_update_state_failed", error=str(e))

    output_start = time.perf_counter()
    result["session_id"] = session_id
    response = OutputGuard.process(result, original_query=body.query)
    log.info(
        "output_guard_complete",
        route="query",
        duration_ms=round((time.perf_counter() - output_start) * 1000, 2),
    )
    return response
