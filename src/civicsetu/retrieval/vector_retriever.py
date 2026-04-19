from __future__ import annotations

import re
import structlog

from civicsetu.models.enums import Jurisdiction
from civicsetu.models.schemas import RetrievedChunk
from civicsetu.stores.relational_store import AsyncSessionLocal
from civicsetu.stores.vector_store import VectorStore

log = structlog.get_logger(__name__)

_RRF_K = 60
_MAX_VECTOR_EXPANDED = 40


class VectorRetriever:
    """
    Hybrid retrieval: vector similarity + PostgreSQL FTS merged via Reciprocal Rank
    Fusion (RRF), with top base-section family expansion.
    Called by vector_retrieval_node, graph_retrieval_node fallback, hybrid_retrieval_node.
    """

    @staticmethod
    def rrf_merge(
        vector_results: list[RetrievedChunk],
        fts_results: list[RetrievedChunk],
        top_n: int,
    ) -> list[RetrievedChunk]:
        """
        RRF score = 1/(k + rank_vector) + 1/(k + rank_fts).
        Deduplicates by chunk_id; chunks in both lists score highest.
        """
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, RetrievedChunk] = {}

        for rank, rc in enumerate(vector_results, 1):
            cid = str(rc.chunk.chunk_id)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank)
            chunk_map[cid] = rc

        for rank, rc in enumerate(fts_results, 1):
            cid = str(rc.chunk.chunk_id)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank)
            if cid not in chunk_map:
                chunk_map[cid] = rc

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [chunk_map[cid] for cid, _ in ranked[:top_n]]

    @staticmethod
    async def retrieve(
        query: str,
        query_embedding: list[float],
        top_k: int,
        jurisdiction: str | None,
    ) -> list[RetrievedChunk]:
        """Run hybrid retrieval and section-family expansion."""
        async with AsyncSessionLocal() as session:
            vector_results = await VectorStore.similarity_search(
                session=session,
                query_embedding=query_embedding,
                top_k=top_k * 3,
                jurisdiction=jurisdiction,
                active_only=True,
            )
            fts_results = await VectorStore.full_text_search(
                session=session,
                query=query,
                top_k=top_k * 2,
                jurisdiction=jurisdiction,
                active_only=True,
            )

            merged = VectorRetriever.rrf_merge(vector_results, fts_results, top_n=top_k * 2)

            seen_ids: set[str] = {str(r.chunk.chunk_id) for r in merged}
            expanded: list[RetrievedChunk] = list(merged)

            for rc in merged[:3]:
                sid = rc.chunk.section_id
                jur = Jurisdiction(rc.chunk.jurisdiction)
                base_sid = re.sub(r'\([^)]*\)$', '', str(sid)).strip()
                for expand_sid in {str(sid), base_sid}:
                    family = await VectorStore.get_section_family(
                        session=session, section_id=expand_sid, jurisdiction=jur
                    )
                    for fc in family:
                        cid = str(fc.chunk.chunk_id)
                        if cid not in seen_ids:
                            seen_ids.add(cid)
                            expanded.append(fc)

            log.info(
                "rrf_retrieve_complete",
                vector_results=len(vector_results),
                fts_results=len(fts_results),
                merged=len(merged),
                results=min(len(expanded), _MAX_VECTOR_EXPANDED),
            )
            return expanded[:_MAX_VECTOR_EXPANDED]
