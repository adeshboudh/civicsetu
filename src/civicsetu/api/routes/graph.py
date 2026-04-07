from __future__ import annotations

import asyncio
import time
import uuid

import structlog
from fastapi import APIRouter, HTTPException, Path, Query, Request
from sqlalchemy import text

from civicsetu.guardrails.input_guard import InputGuard
from civicsetu.guardrails.output_guard import OutputGuard
from civicsetu.models.enums import Jurisdiction, QueryType
from civicsetu.models.schemas import (
    ChatMessage,
    CivicSetuResponse,
    ConnectedSectionOut,
    GraphEdge,
    GraphNode,
    GraphTopologyResponse,
    InsufficientInfoResponse,
    SectionChunkOut,
    SectionContentResponse,
    SectionContextQueryRequest,
)
from civicsetu.stores.graph_store import GraphStore
from civicsetu.stores.relational_store import AsyncSessionLocal

log = structlog.get_logger(__name__)
router = APIRouter()

_topo_cache: dict = {"data": None, "ts": 0.0}
_TOPO_TTL = 300


@router.get("/graph/topology", response_model=GraphTopologyResponse)
async def get_graph_topology() -> GraphTopologyResponse:
    """All connected Section nodes and REFERENCES/DERIVED_FROM edges."""
    now = time.monotonic()
    if _topo_cache["data"] is not None and (now - _topo_cache["ts"]) < _TOPO_TTL:
        log.info("topology_cache_hit")
        return _topo_cache["data"]

    try:
        nodes_raw, edges_raw = await GraphStore.get_topology()
        stats = await GraphStore.graph_stats()
    except Exception as e:
        log.error("topology_fetch_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Graph topology fetch failed: {e}")

    response = GraphTopologyResponse(
        nodes=[GraphNode(**n) for n in nodes_raw],
        edges=[GraphEdge(**e) for e in edges_raw],
        stats=stats,
    )
    _topo_cache["data"] = response
    _topo_cache["ts"] = now
    log.info("topology_cache_updated", nodes=len(response.nodes), edges=len(response.edges))
    return response


@router.get("/graph/section/{section_id:path}", response_model=SectionContentResponse)
async def get_section_content(
    section_id: str = Path(..., description="Section ID, e.g. 'Section 18'"),
    jurisdiction: str = Query(..., description="Jurisdiction enum value, e.g. 'CENTRAL'"),
    chunk_id: str | None = Query(default=None, description="Optional chunk id from the graph node"),
) -> SectionContentResponse:
    """Stitched section text from Postgres plus connected sections from Neo4j."""
    async with AsyncSessionLocal() as db:
        resolved_section_id = section_id
        resolved_jurisdiction = jurisdiction

        if chunk_id:
            node_result = await db.execute(
                text(
                    """
                    SELECT
                        section_id,
                        jurisdiction
                    FROM legal_chunks
                    WHERE chunk_id::text = :chunk_id
                      AND lower(status) = 'active'
                    LIMIT 1
                    """
                ),
                {"chunk_id": chunk_id},
            )
            node_row = node_result.mappings().first()
            if node_row:
                resolved_section_id = node_row["section_id"]
                resolved_jurisdiction = node_row["jurisdiction"]

        result = await db.execute(
            text(
                """
                SELECT
                    chunk_id::text AS chunk_id,
                    section_id,
                    section_title,
                    text,
                    page_number,
                    doc_name,
                    jurisdiction,
                    effective_date,
                    source_url
                FROM legal_chunks
                WHERE section_id = :section_id
                  AND jurisdiction = :jurisdiction
                  AND lower(status) = 'active'
                ORDER BY page_number ASC
                """
            ),
            {"section_id": resolved_section_id, "jurisdiction": resolved_jurisdiction},
        )
        rows = result.mappings().all()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No chunks found for section_id='{section_id}' jurisdiction='{jurisdiction}'"
                + (f" chunk_id='{chunk_id}'" if chunk_id else "")
            ),
        )

    first = rows[0]
    chunks = [
        SectionChunkOut(chunk_id=row["chunk_id"], text=row["text"], page_number=row["page_number"])
        for row in rows
    ]

    refs_out, refs_in, derived_out, derived_in = await asyncio.gather(
        GraphStore.get_referenced_sections(resolved_section_id, resolved_jurisdiction),
        GraphStore.get_sections_referencing(resolved_section_id, resolved_jurisdiction),
        GraphStore.get_derived_act_sections(resolved_section_id, resolved_jurisdiction),
        GraphStore.get_deriving_rule_sections(resolved_section_id, resolved_jurisdiction),
        return_exceptions=True,
    )

    def _safe_connections(result: object, edge_type: str) -> list[ConnectedSectionOut]:
        if isinstance(result, Exception):
            log.warning("connected_sections_partial_failure", edge_type=edge_type, error=str(result))
            return []
        return [
            ConnectedSectionOut(
                section_id=row["section_id"],
                title=row.get("title", ""),
                jurisdiction=row.get("jurisdiction", resolved_jurisdiction),
                edge_type=edge_type,
            )
            for row in result  # type: ignore[union-attr]
        ]

    connected = (
        _safe_connections(refs_out, "REFERENCES_OUT")
        + _safe_connections(refs_in, "REFERENCES_IN")
        + _safe_connections(derived_out, "DERIVED_FROM_OUT")
        + _safe_connections(derived_in, "DERIVED_FROM_IN")
    )

    log.info(
        "section_content_fetched",
        section_id=resolved_section_id,
        jurisdiction=resolved_jurisdiction,
        chunks=len(chunks),
        connected=len(connected),
    )

    return SectionContentResponse(
        section_id=first.get("section_id", resolved_section_id),
        title=first["section_title"],
        doc_name=first["doc_name"],
        jurisdiction=resolved_jurisdiction,
        effective_date=str(first["effective_date"]) if first["effective_date"] else None,
        source_url=first["source_url"],
        chunks=chunks,
        connected_sections=connected,
    )


@router.post("/query/section-context", response_model=CivicSetuResponse | InsufficientInfoResponse)
async def section_context_query(
    request: Request,
    body: SectionContextQueryRequest,
) -> CivicSetuResponse | InsufficientInfoResponse:
    """Bypass the classifier and route directly into graph retrieval for a chosen section."""
    guard_result = InputGuard.check(body.query)
    if not guard_result.is_safe:
        log.info("section_context_rejected", reason=guard_result.reason)
        raise HTTPException(status_code=400, detail=guard_result.reason)

    try:
        jurisdiction = Jurisdiction(body.jurisdiction)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid jurisdiction: '{body.jurisdiction}'")

    safe_query = guard_result.sanitized_query
    session_id = body.session_id or str(uuid.uuid4())
    graph = request.app.state.graph
    config = {"configurable": {"thread_id": session_id}}

    initial_state = {
        "query": safe_query,
        "session_id": session_id,
        "jurisdiction_filter": jurisdiction,
        "top_k": 5,
        "messages": [ChatMessage(role="user", content=safe_query)],
        "source_section_id": body.section_id,
        "query_type": QueryType.CROSS_REFERENCE,
        "rewritten_query": safe_query,
        "skip_classifier": True,
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
            route="section_context",
            duration_ms=round((time.perf_counter() - invoke_start) * 1000, 2),
        )
    except Exception as e:
        log.error("section_context_invoke_failed", error=str(e))
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
                route="section_context",
                duration_ms=round((time.perf_counter() - update_start) * 1000, 2),
            )
        except Exception as e:
            log.warning("section_context_update_state_failed", error=str(e))

    output_start = time.perf_counter()
    result["session_id"] = session_id
    response = OutputGuard.process(result, original_query=body.query)
    log.info(
        "output_guard_complete",
        route="section_context",
        duration_ms=round((time.perf_counter() - output_start) * 1000, 2),
    )
    return response
