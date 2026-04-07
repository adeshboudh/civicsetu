from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from civicsetu.retrieval.cache import make_key, retrieval_cache


@pytest.mark.asyncio
async def test_similarity_search_cache_hit():
    retrieval_cache.clear()

    fake_embedding = [0.1] * 768
    fake_chunks = [MagicMock()]
    retrieval_cache[make_key(str(fake_embedding), None, None, 5, True)] = fake_chunks

    session = AsyncMock()

    from civicsetu.stores.vector_store import VectorStore

    result = await VectorStore.similarity_search(
        session=session,
        query_embedding=fake_embedding,
        top_k=5,
        jurisdiction=None,
    )

    assert result == fake_chunks
    session.execute.assert_not_called()


@pytest.mark.asyncio
async def test_similarity_search_cache_miss_populates_cache():
    retrieval_cache.clear()

    fake_embedding = [0.2] * 768
    session = AsyncMock()
    result_proxy = MagicMock()
    result_proxy.fetchall.return_value = []
    session.execute = AsyncMock(return_value=result_proxy)

    from civicsetu.stores.vector_store import VectorStore

    result = await VectorStore.similarity_search(
        session=session,
        query_embedding=fake_embedding,
        top_k=5,
        jurisdiction=None,
    )

    assert result == []
    assert make_key(str(fake_embedding), None, None, 5, True) in retrieval_cache
    session.execute.assert_awaited_once()
