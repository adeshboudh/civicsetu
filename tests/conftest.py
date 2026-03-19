from __future__ import annotations

import uuid
from datetime import date

import pytest

from civicsetu.models.enums import ChunkStatus, DocType, Jurisdiction, QueryType
from civicsetu.models.schemas import Citation, LegalChunk, RetrievedChunk


def _make_chunk(
    section_id: str = "18",
    doc_name: str = "RERA Act 2016",
    jurisdiction: Jurisdiction = Jurisdiction.CENTRAL,
    text: str = "Section text placeholder.",
    doc_id: uuid.UUID | None = None,
) -> LegalChunk:
    return LegalChunk(
        chunk_id=uuid.uuid4(),
        doc_id=doc_id or uuid.uuid4(),
        jurisdiction=jurisdiction,
        doc_type=DocType.ACT,
        doc_name=doc_name,
        section_id=section_id,
        section_title=f"Section {section_id} Title",
        section_hierarchy=[doc_name, section_id],
        text=text,
        effective_date=date(2016, 5, 1),
        status=ChunkStatus.ACTIVE,
        source_url="https://example.com/rera.pdf",
        page_number=1,
    )


def _make_rc(
    section_id: str = "18",
    doc_name: str = "RERA Act 2016",
    jurisdiction: Jurisdiction = Jurisdiction.CENTRAL,
    is_pinned: bool = False,
    retrieval_source: str = "vector",
    rerank_score: float | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=_make_chunk(section_id=section_id, doc_name=doc_name, jurisdiction=jurisdiction),
        retrieval_source=retrieval_source,
        is_pinned=is_pinned,
        rerank_score=rerank_score,
    )


def _base_state(**overrides) -> dict:
    state = {
        "query": "What are promoter obligations?",
        "session_id": "test-session",
        "jurisdiction_filter": None,
        "top_k": 5,
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
    state.update(overrides)
    return state


@pytest.fixture
def make_chunk():
    return _make_chunk


@pytest.fixture
def make_rc():
    return _make_rc


@pytest.fixture
def base_state():
    return _base_state
