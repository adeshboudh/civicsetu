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

# Bonus added to RRF score for chunks whose section_id matches a section
# mentioned in the query.  This surfaces the *directly asked-about* section
# even when it ranks slightly lower in pure vector / FTS.
_SECTION_BOOST = 0.05

# Regex to extract section IDs from rewritten queries.
# Matches: "section 11", "sec. 5", "s. 18", "Section 19(3)", "Rule 3"
_SECTION_RE = re.compile(
    r'\b(?:section|sec\.?|s\.)\s*(\d+)(?:\([^)]*\))?[A-Z]?\b',
    re.IGNORECASE,
)
_RULE_RE = re.compile(r'\bRule\s+(\d+)(?:\([^)]*\))?[A-Z]?\b', re.IGNORECASE)


class VectorRetriever:
    """
    Hybrid retrieval: vector similarity + PostgreSQL FTS merged via Reciprocal Rank
    Fusion (RRF), with top base-section family expansion.
    Called by vector_retrieval_node, graph_retrieval_node fallback, hybrid_retrieval_node.
    """

    @staticmethod
    def _extract_query_section_ids(query: str) -> set[str]:
        """Extract base section IDs mentioned in the query text."""
        ids: set[str] = set()
        for m in _SECTION_RE.finditer(query):
            ids.add(m.group(1))
        for m in _RULE_RE.finditer(query):
            ids.add(m.group(1))
        return ids

    @staticmethod
    def rrf_merge(
        vector_results: list[RetrievedChunk],
        fts_results: list[RetrievedChunk],
        top_n: int,
        query: str = "",
    ) -> list[RetrievedChunk]:
        """
        RRF score = 1/(k + rank_vector) + 1/(k + rank_fts) + section_boost.
        Deduplicates by chunk_id; chunks in both lists score highest.
        Chunks whose section_id matches a section mentioned in the query
        receive an additional _SECTION_BOOST bonus to surface them.
        """
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, RetrievedChunk] = {}

        # Pre-compute which base section IDs the query mentions
        query_sids = VectorRetriever._extract_query_section_ids(query)

        for rank, rc in enumerate(vector_results, 1):
            cid = str(rc.chunk.chunk_id)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank)
            chunk_map[cid] = rc

        for rank, rc in enumerate(fts_results, 1):
            cid = str(rc.chunk.chunk_id)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank)
            if cid not in chunk_map:
                chunk_map[cid] = rc

        # Apply section-ID boost: if a chunk's base section_id matches
        # a section mentioned in the query, boost its score
        if query_sids:
            for cid, rc in chunk_map.items():
                base_sid = re.sub(r'\([^)]*\)$', '', rc.chunk.section_id).strip()
                if base_sid in query_sids:
                    rrf_scores[cid] = rrf_scores[cid] + _SECTION_BOOST
            log.info("rrf_section_boost_applied", query_sids=list(query_sids),
                     boosted=sum(1 for cid in chunk_map
                                 if re.sub(r'\([^)]*\)$', '', chunk_map[cid].chunk.section_id).strip() in query_sids))

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

            merged = VectorRetriever.rrf_merge(
                vector_results, fts_results, top_n=top_k * 2, query=query,
            )

            seen_ids: set[str] = {str(r.chunk.chunk_id) for r in merged}
            expanded: list[RetrievedChunk] = list(merged)

            _FAMILY_CAP = 6  # max chunks added per section per expand pass
            for rc in merged[:5]:  # expand top-5 (was top-3)
                sid = rc.chunk.section_id
                jur = Jurisdiction(rc.chunk.jurisdiction)
                # get_section_family now handles sub-section stripping internally,
                # so we only need to call it once per merged chunk.
                family = await VectorStore.get_section_family(
                    session=session, section_id=str(sid), jurisdiction=jur
                )
                added = 0
                for fc in family:
                    if added >= _FAMILY_CAP:
                        break
                    cid = str(fc.chunk.chunk_id)
                    if cid not in seen_ids:
                        seen_ids.add(cid)
                        expanded.append(fc)
                        added += 1

            log.info(
                "rrf_retrieve_complete",
                vector_results=len(vector_results),
                fts_results=len(fts_results),
                merged=len(merged),
                results=min(len(expanded), _MAX_VECTOR_EXPANDED),
            )
            return expanded[:_MAX_VECTOR_EXPANDED]
