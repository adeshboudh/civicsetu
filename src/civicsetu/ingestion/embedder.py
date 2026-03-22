from __future__ import annotations

import httpx
from functools import lru_cache

import ollama
import structlog

from civicsetu.config.settings import get_settings

log = structlog.get_logger(__name__)
settings = get_settings()

MAX_EMBED_CHARS_QUERY = 6000
MAX_EMBED_CHARS_DOC   = 4000

@lru_cache(maxsize=1)
# def _get_ollama_client() -> ollama.Client:
#     """Singleton Ollama client — one connection for the process lifetime."""
#     return ollama.Client(host=settings.ollama_base_url)
def _get_model():
    from sentence_transformers import SentenceTransformer
    from huggingface_hub import login
    if settings.hf_token:
        login(token=settings.hf_token, add_to_git_credential=False)
    log.info("loading_embedding_model", model="nomic-ai/nomic-embed-text-v1.5")
    return SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)


class Embedder:
    """
    Wraps Ollama nomic-embed-text.
    All methods are synchronous — embedding is CPU/GPU bound,
    not I/O bound, so async adds no benefit here.
    Caller wraps in asyncio.to_thread() if needed in async context.
    """

    def __init__(self):
        # self.client = _get_ollama_client()
        # self.model = settings.embedding_model
        self.dimension = settings.embedding_dimension

    def embed_one(self, text: str) -> list[float]:
        """Embed a single string. Raises on empty input."""
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")
        model = _get_model()
        embedding = model.encode(text, normalize_embeddings=True).tolist()
        if len(embedding) != self.dimension:
            raise ValueError(
                f"Model returned dim={len(embedding)}, expected {self.dimension}."
            )
        return embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of strings. Fails fast on first empty input.
        Returns embeddings in same order as input.
        """
        if not texts: return []
        embeddings = []
        for i, text in enumerate(texts):
            try:
                embeddings.append(self.embed_one(text))
            except Exception as e:
                log.error("embedding_failed", index=i, error=str(e))
                raise
        log.info("batch_embedded", count=len(embeddings))
        return embeddings

    def embed_query(self, query: str) -> list[float]:
        """
        Identical to embed_one but semantically distinct —
        some models use different prompts for queries vs documents.
        nomic-embed-text supports 'search_query:' prefix for better retrieval.
        """
        if len(query) > MAX_EMBED_CHARS_QUERY:
            log.warning("embedding_truncated", original_len=len(query), truncated_to=MAX_EMBED_CHARS_QUERY)
            query = query[:MAX_EMBED_CHARS_QUERY]
        return self.embed_one(f"search_query: {query.strip()}")

    def embed_document(self, text: str) -> list[float]:
        """
        Document-side embedding with nomic-embed-text 'search_document:' prefix.
        Always use this for chunks at ingestion time.
        Always use embed_query() for user queries at retrieval time.
        """
        if len(text) > MAX_EMBED_CHARS_DOC:
            log.warning("embedding_truncated", original_len=len(text), truncated_to=MAX_EMBED_CHARS_DOC)
            text = text[:MAX_EMBED_CHARS_DOC]
        return self.embed_one(f"search_document: {text.strip()}")

    def embed_batch_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document texts using the search_document: prefix."""
        if not texts: return []
        embeddings = []
        for i, text in enumerate(texts):
            try:
                embeddings.append(self.embed_document(text))
            except Exception as e:
                log.error("embedding_failed", index=i, error=str(e))
                raise
        log.info("batch_embedded", count=len(embeddings))
        return embeddings
