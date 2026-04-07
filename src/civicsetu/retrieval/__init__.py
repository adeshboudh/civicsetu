from __future__ import annotations

import structlog

from civicsetu.ingestion.embedder import Embedder
from civicsetu.retrieval.cache import embedding_cache, make_key

log = structlog.get_logger(__name__)
_embedder = Embedder()


def cached_embed(query: str) -> list[float]:
    """Embed query text with a short-lived in-process cache."""
    key = make_key(query)
    cached = embedding_cache.get(key)
    if cached is not None:
        log.debug("embedding_cache_hit", query=query[:60])
        return cached

    log.debug("embedding_cache_miss", query=query[:60])
    embedding = _embedder.embed_query(query)
    embedding_cache[key] = embedding
    return embedding


def warm_embedding_model() -> None:
    """Load the embedding model during startup instead of the first user request."""
    cached_embed("warmup")
