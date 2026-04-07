from __future__ import annotations

from unittest.mock import patch

from civicsetu.retrieval import cached_embed
from civicsetu.retrieval.cache import embedding_cache, make_key


def test_embedding_cache_hit_skips_model_call():
    embedding_cache.clear()

    query = "What are promoter obligations?"
    fake_embedding = [0.1] * 768
    embedding_cache[make_key(query)] = fake_embedding

    with patch("civicsetu.ingestion.embedder.Embedder.embed_query", side_effect=RuntimeError("should not run")):
        result = cached_embed(query)

    assert result == fake_embedding


def test_embedding_cache_miss_calls_model_and_populates_cache():
    embedding_cache.clear()

    query = "unique query xyz"
    fake_embedding = [0.5] * 768

    with patch("civicsetu.ingestion.embedder.Embedder.embed_query", return_value=fake_embedding) as mock_embed:
        result = cached_embed(query)

    assert result == fake_embedding
    assert embedding_cache[make_key(query)] == fake_embedding
    mock_embed.assert_called_once_with(query)
