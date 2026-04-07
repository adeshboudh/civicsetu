from __future__ import annotations

import operator
from typing import Annotated, NotRequired

from typing_extensions import TypedDict

from civicsetu.models.enums import Jurisdiction, QueryType
from civicsetu.models.schemas import ChatMessage, Citation, RetrievedChunk


class CivicSetuState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────────
    query: str
    session_id: str | None
    jurisdiction_filter: Jurisdiction | None
    top_k: int
    messages: Annotated[list[ChatMessage], operator.add]

    # ── Classification ─────────────────────────────────────────────────────────
    query_type: QueryType | None
    rewritten_query: str | None           # expanded/clarified query

    # ── Retrieval ──────────────────────────────────────────────────────────────
    retrieved_chunks: list[RetrievedChunk]
    reranked_chunks: list[RetrievedChunk]

    # ── Generation ─────────────────────────────────────────────────────────────
    raw_response: str | None
    citations: list[Citation]
    confidence_score: float
    conflict_warnings: list[str]
    amendment_notice: str | None

    # ── Control Flow ───────────────────────────────────────────────────────────
    retry_count: int
    hallucination_flag: bool
    error: str | None
    skip_classifier: NotRequired[bool]
    source_section_id: NotRequired[str]
