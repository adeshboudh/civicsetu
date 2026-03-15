from __future__ import annotations

import asyncio

import structlog
from fastapi import APIRouter, HTTPException, Request

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
    graph = request.app.state.graph

    initial_state = {
        "query": body.query,
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

    try:
        result = await asyncio.to_thread(graph.invoke, initial_state)
    except Exception as e:
        log.error("graph_invoke_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

    if not result.get("citations"):
        return InsufficientInfoResponse(searched_query=body.query)

    return CivicSetuResponse(
        answer=result["raw_response"],
        citations=result["citations"],
        confidence_score=result["confidence_score"],
        query_type_resolved=result["query_type"] or QueryType.FACT_LOOKUP,
        conflict_warnings=result.get("conflict_warnings", []),
        amendment_notice=result.get("amendment_notice"),
    )
