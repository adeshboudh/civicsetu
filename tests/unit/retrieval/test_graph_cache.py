from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from civicsetu.retrieval.cache import graph_cache, make_key


@pytest.mark.asyncio
async def test_graph_retrieve_cache_hit():
    graph_cache.clear()

    fake_chunks = [MagicMock()]
    graph_cache[make_key("18", "all", 2)] = fake_chunks

    with patch(
        "civicsetu.stores.graph_store.GraphStore.get_referenced_sections",
        side_effect=RuntimeError("should not run"),
    ):
        from civicsetu.retrieval.graph_retriever import GraphRetriever

        result = await GraphRetriever.retrieve(
            query="What does Section 18 reference?",
            jurisdiction=None,
        )

    assert result == fake_chunks


@pytest.mark.asyncio
async def test_graph_retrieve_cache_miss_populates_cache():
    graph_cache.clear()

    with (
        patch("civicsetu.stores.graph_store.GraphStore.get_referenced_sections", new=AsyncMock(return_value=[])),
        patch("civicsetu.stores.graph_store.GraphStore.get_sections_referencing", new=AsyncMock(return_value=[])),
        patch("civicsetu.stores.graph_store.GraphStore.get_derived_act_sections", new=AsyncMock(return_value=[])),
        patch("civicsetu.stores.graph_store.GraphStore.get_deriving_rule_sections", new=AsyncMock(return_value=[])),
        patch("civicsetu.stores.vector_store.VectorStore.get_by_section", new=AsyncMock(return_value=[])),
    ):
        from civicsetu.retrieval.graph_retriever import GraphRetriever

        result = await GraphRetriever.retrieve(
            query="What does Section 18 say?",
            jurisdiction=None,
        )

    assert result == []
    assert make_key("18", "all", 2) in graph_cache
