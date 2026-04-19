from __future__ import annotations

from unittest.mock import AsyncMock, patch
import pytest

from tests.conftest import _make_rc


def test_rrf_merge_empty_inputs():
    from civicsetu.retrieval.vector_retriever import VectorRetriever
    assert VectorRetriever.rrf_merge([], [], top_n=5) == []


def test_rrf_merge_deduplicates_by_chunk_id():
    from civicsetu.retrieval.vector_retriever import VectorRetriever
    rc = _make_rc(section_id="18")
    result = VectorRetriever.rrf_merge([rc], [rc], top_n=5)
    assert len(result) == 1


def test_rrf_merge_ranks_overlap_highest():
    from civicsetu.retrieval.vector_retriever import VectorRetriever
    shared = _make_rc(section_id="18")
    vector_only = _make_rc(section_id="3")
    fts_only = _make_rc(section_id="7")
    result = VectorRetriever.rrf_merge([shared, vector_only], [shared, fts_only], top_n=3)
    assert result[0].chunk.section_id == "18"


def test_rrf_merge_respects_top_n():
    from civicsetu.retrieval.vector_retriever import VectorRetriever
    chunks = [_make_rc(section_id=str(i)) for i in range(5)]
    assert len(VectorRetriever.rrf_merge(chunks, [], top_n=2)) == 2


def test_rrf_merge_vector_chunk_wins_on_id_collision():
    from civicsetu.retrieval.vector_retriever import VectorRetriever
    rc_vector = _make_rc(section_id="18")
    rc_fts = _make_rc(section_id="18")
    rc_fts.chunk.chunk_id = rc_vector.chunk.chunk_id
    result = VectorRetriever.rrf_merge([rc_vector], [rc_fts], top_n=5)
    assert len(result) == 1
    assert result[0] is rc_vector


@pytest.mark.asyncio
async def test_retrieve_returns_list_of_retrieved_chunks():
    from civicsetu.retrieval.vector_retriever import VectorRetriever
    from civicsetu.models.schemas import RetrievedChunk
    rc = _make_rc(section_id="18")
    with patch("civicsetu.retrieval.vector_retriever.AsyncSessionLocal") as mock_scls, \
         patch("civicsetu.retrieval.vector_retriever.VectorStore") as mock_vs:
        mock_session = AsyncMock()
        mock_scls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_scls.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_vs.similarity_search = AsyncMock(return_value=[rc])
        mock_vs.full_text_search = AsyncMock(return_value=[rc])
        mock_vs.get_section_family = AsyncMock(return_value=[])
        result = await VectorRetriever.retrieve(
            query="test query", query_embedding=[0.1] * 768, top_k=5, jurisdiction=None
        )
    assert isinstance(result, list)
    assert all(isinstance(r, RetrievedChunk) for r in result)


@pytest.mark.asyncio
async def test_retrieve_caps_at_max_expanded():
    from civicsetu.retrieval.vector_retriever import VectorRetriever, _MAX_VECTOR_EXPANDED
    many = [_make_rc(section_id=str(i)) for i in range(50)]
    with patch("civicsetu.retrieval.vector_retriever.AsyncSessionLocal") as mock_scls, \
         patch("civicsetu.retrieval.vector_retriever.VectorStore") as mock_vs:
        mock_session = AsyncMock()
        mock_scls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_scls.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_vs.similarity_search = AsyncMock(return_value=many)
        mock_vs.full_text_search = AsyncMock(return_value=[])
        mock_vs.get_section_family = AsyncMock(return_value=[])
        result = await VectorRetriever.retrieve(
            query="test", query_embedding=[0.0] * 768, top_k=5, jurisdiction=None
        )
    assert len(result) <= _MAX_VECTOR_EXPANDED


@pytest.mark.asyncio
async def test_retrieve_expands_section_family():
    from civicsetu.retrieval.vector_retriever import VectorRetriever
    base_rc = _make_rc(section_id="18")
    family_rc = _make_rc(section_id="18(1)")
    with patch("civicsetu.retrieval.vector_retriever.AsyncSessionLocal") as mock_scls, \
         patch("civicsetu.retrieval.vector_retriever.VectorStore") as mock_vs:
        mock_session = AsyncMock()
        mock_scls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_scls.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_vs.similarity_search = AsyncMock(return_value=[base_rc])
        mock_vs.full_text_search = AsyncMock(return_value=[])
        mock_vs.get_section_family = AsyncMock(return_value=[family_rc])
        result = await VectorRetriever.retrieve(
            query="test", query_embedding=[0.0] * 768, top_k=5, jurisdiction=None
        )
    chunk_ids = [str(r.chunk.chunk_id) for r in result]
    assert str(family_rc.chunk.chunk_id) in chunk_ids
