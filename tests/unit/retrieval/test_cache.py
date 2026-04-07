from __future__ import annotations

from cachetools import TTLCache

from civicsetu.retrieval.cache import embedding_cache, graph_cache, make_key, retrieval_cache


def test_make_key_normalizes_whitespace():
    assert make_key("  What are RERA penalties?  ") == make_key("What are RERA penalties?")


def test_make_key_is_hex_string():
    assert len(make_key("hello")) == 64


def test_make_key_different_inputs_differ():
    assert make_key("abc") != make_key("xyz")


def test_embedding_cache_is_ttl_cache():
    assert isinstance(embedding_cache, TTLCache)
    assert embedding_cache.maxsize == 512


def test_retrieval_cache_is_ttl_cache():
    assert isinstance(retrieval_cache, TTLCache)
    assert retrieval_cache.maxsize == 256


def test_graph_cache_is_ttl_cache():
    assert isinstance(graph_cache, TTLCache)
    assert graph_cache.maxsize == 256
