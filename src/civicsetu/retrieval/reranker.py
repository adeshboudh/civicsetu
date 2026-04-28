from __future__ import annotations

import structlog

from civicsetu.config.settings import get_settings
from civicsetu.models.schemas import RetrievedChunk

log = structlog.get_logger(__name__)
settings = get_settings()

_ranker = None


def _get_ranker():
    global _ranker
    if _ranker is not None:
        return _ranker

    from flashrank import Ranker
    ranker = Ranker(model_name=settings.reranker_model, cache_dir=".cache/flashrank")

    # Do not cache unittest mocks — tests patching Ranker must not bleed into each other.
    if type(ranker).__module__ != "unittest.mock":
        _ranker = ranker
    return ranker


def _apply_score_gap(chunks: list[RetrievedChunk], gap: float) -> list[RetrievedChunk]:
    """
    Returns prefix of chunks (sorted by rerank_score desc).
    Stops when drop from one chunk to the next meets or exceeds `gap`.
    """
    if len(chunks) <= 1:
        return list(chunks)
    result = [chunks[0]]
    for prev, curr in zip(chunks, chunks[1:]):
        if (prev.rerank_score or 0.0) - (curr.rerank_score or 0.0) >= gap:
            break
        result.append(curr)
    return result


class Reranker:
    """
    FlashRank cross-encoder reranking.
    Pipeline: dedup → pin-separate → score → threshold → gap-filter → combine (max 5).
    """

    @staticmethod
    def rerank(chunks: list[RetrievedChunk], query: str) -> list[RetrievedChunk]:
        from flashrank import RerankRequest

        if not chunks:
            return []

        seen: set[tuple[str, str]] = set()
        unique_chunks: list[RetrievedChunk] = []
        for c in chunks:
            key = (c.chunk.section_id, c.chunk.doc_name)
            if key not in seen:
                seen.add(key)
                unique_chunks.append(c)

        log.info("reranker_dedup", unique_chunks=len(unique_chunks))

        pinned = [c for c in unique_chunks if c.is_pinned][:7]
        rankable = [c for c in unique_chunks if not c.is_pinned]

        try:
            ranker = _get_ranker()
            passages = [{"id": i, "text": c.chunk.text} for i, c in enumerate(rankable)]
            request = RerankRequest(query=query, passages=passages)
            results = ranker.rerank(request)

            id_to_chunk = {i: c for i, c in enumerate(rankable)}
            reranked_rankable: list[RetrievedChunk] = []
            for r in results:
                chunk = id_to_chunk[r["id"]]
                chunk.rerank_score = round(float(r["score"]), 4)
                reranked_rankable.append(chunk)

            above_threshold = [
                c for c in reranked_rankable
                if (c.rerank_score or 0.0) >= settings.reranker_score_threshold
            ]
            gap_filtered = _apply_score_gap(above_threshold, settings.reranker_score_gap)

            dropped = len(reranked_rankable) - len(gap_filtered)
            log.info(
                "reranker_filtered",
                before=len(reranked_rankable),
                after=len(gap_filtered),
                dropped=dropped,
                threshold=settings.reranker_score_threshold,
                gap=settings.reranker_score_gap,
            )

            slots = max(0, 7 - len(pinned))
            return pinned + gap_filtered[:slots]

        except Exception as e:
            log.warning("reranker_failed", error=str(e), fallback="vector_order")
            slots = max(0, 7 - len(pinned))
            return pinned + rankable[:slots]
