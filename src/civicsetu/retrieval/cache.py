from __future__ import annotations

import hashlib

from cachetools import TTLCache

embedding_cache: TTLCache = TTLCache(maxsize=512, ttl=3600)
retrieval_cache: TTLCache = TTLCache(maxsize=256, ttl=900)
graph_cache: TTLCache = TTLCache(maxsize=256, ttl=900)


def make_key(*parts: object) -> str:
    """Return a stable SHA-256 cache key for normalized stringified parts."""
    normalized = "|".join(str(part).strip().lower() for part in parts)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
