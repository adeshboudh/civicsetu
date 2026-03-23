from __future__ import annotations

import asyncio
import structlog

from civicsetu.models.enums import Jurisdiction
from civicsetu.models.schemas import RetrievedChunk
from civicsetu.stores.graph_store import GraphStore
from civicsetu.stores.relational_store import AsyncSessionLocal
from civicsetu.stores.vector_store import VectorStore

log = structlog.get_logger(__name__)

# All jurisdictions searched when no filter is provided.
# Order matters: CENTRAL first so DERIVED_FROM incoming edges surface
# all state rules in one traversal pass.
_ALL_JURISDICTIONS = [
    "CENTRAL",
    "MAHARASHTRA",
    "UTTAR_PRADESH",
    "KARNATAKA",
    "TAMIL_NADU",
]

# Max chunks returned to reranker — prevents FlashRank serial inference blowup
# on high-connectivity sections (e.g. §9 has DERIVED_FROM edges from 8 state rules)
_MAX_GRAPH_CHUNKS = 20


class GraphRetriever:
    """
    Retrieves chunks via Neo4j graph traversal rather than vector similarity.

    Used for:
    - cross_reference queries: "What does Section 18 reference?"
    - penalty_lookup queries: "What are the penalties under Section 59?"
    - temporal queries: "Which sections were amended?"

    Strategy:
    1. Extract section_id from query (regex)
    2. Traverse REFERENCES + DERIVED_FROM edges in Neo4j (all 5 jurisdictions in parallel)
    3. Hydrate section_ids into LegalChunk objects via VectorStore.get_by_section()
    4. Dedup + cap at _MAX_GRAPH_CHUNKS before returning
    """

    @staticmethod
    async def retrieve(
        query: str,
        jurisdiction: Jurisdiction | None = None,
        depth: int = 2,
    ) -> list[RetrievedChunk]:
        section_id = GraphRetriever._extract_section_id(query)

        if not section_id:
            log.info("graph_retriever_no_section_found", query=query[:80])
            return []

        # Explicit filter → search only that jurisdiction.
        # No filter → search all jurisdictions (CENTRAL first).
        jurisdictions_to_search = (
            [jurisdiction.value] if jurisdiction
            else _ALL_JURISDICTIONS
        )

        log.info(
            "graph_retriever_traversing",
            section_id=section_id,
            depth=depth,
            jurisdictions=jurisdictions_to_search,
        )

        chunks: list[RetrievedChunk] = []
        seen_chunk_ids: set[str] = set()

        async def _fetch_jurisdiction(jur_str: str) -> list[RetrievedChunk]:
            async with AsyncSessionLocal() as session:  # ← session scoped per coroutine
                jur_enum = Jurisdiction(jur_str) if not jurisdiction else jurisdiction
                jur_chunks: list[RetrievedChunk] = []

                source = await VectorStore.get_by_section(
                    session=session, section_id=section_id, jurisdiction=jur_enum,
                )
                for rc in source:
                    rc.retrieval_source = "graph"
                    rc.graph_path = f"source:{section_id}@{jur_str}"
                    rc.is_pinned = rc.chunk.section_id == section_id
                jur_chunks.extend(source)

                outgoing = await GraphStore.get_referenced_sections(section_id, jur_str, depth)
                for node in outgoing:
                    hydrated = await VectorStore.get_by_section(
                        session=session,
                        section_id=node["section_id"],
                        jurisdiction=Jurisdiction(node["jurisdiction"]),
                    )
                    for rc in hydrated:
                        rc.retrieval_source = "graph"
                        rc.graph_path = f"{section_id} →[REFERENCES]→ {node['section_id']}"
                    jur_chunks.extend(hydrated)

                incoming = await GraphStore.get_sections_referencing(section_id, jur_str)
                for node in incoming:
                    hydrated = await VectorStore.get_by_section(
                        session=session,
                        section_id=node["section_id"],
                        jurisdiction=Jurisdiction(node["jurisdiction"]),
                    )
                    for rc in hydrated:
                        rc.retrieval_source = "graph"
                        rc.graph_path = f"{node['section_id']} →[REFERENCES]→ {section_id}"
                    jur_chunks.extend(hydrated)

                derived_act = await GraphStore.get_derived_act_sections(section_id, jur_str)
                for node in derived_act:
                    hydrated = await VectorStore.get_by_section(
                        session=session,
                        section_id=node["section_id"],
                        jurisdiction=Jurisdiction(node["jurisdiction"]),
                    )
                    for rc in hydrated:
                        rc.retrieval_source = "graph"
                        rc.graph_path = f"{section_id}@{jur_str} →[DERIVED_FROM]→ {node['section_id']}@{node['jurisdiction']}"
                    jur_chunks.extend(hydrated)

                deriving = await GraphStore.get_deriving_rule_sections(section_id, jur_str)
                for node in deriving:
                    hydrated = await VectorStore.get_by_section(
                        session=session,
                        section_id=node["section_id"],
                        jurisdiction=Jurisdiction(node["jurisdiction"]),
                    )
                    for rc in hydrated:
                        rc.retrieval_source = "graph"
                        rc.graph_path = f"{node['section_id']}@{node['jurisdiction']} →[DERIVED_FROM]→ {section_id}@{jur_str}"
                    jur_chunks.extend(hydrated)

                return jur_chunks

        # gather is now OUTSIDE the session context
        all_results = await asyncio.gather(
            *[_fetch_jurisdiction(j) for j in jurisdictions_to_search],
            return_exceptions=True,
        )
        for result in all_results:
            if isinstance(result, Exception):
                log.warning("graph_jurisdiction_fetch_failed", error=str(result))
            else:
                chunks.extend(result)

        # Dedup by chunk_id (same chunk can arrive via multiple traversal paths)
        deduped: list[RetrievedChunk] = []
        for rc in chunks:
            cid = str(rc.chunk.chunk_id)
            if cid not in seen_chunk_ids:
                seen_chunk_ids.add(cid)
                deduped.append(rc)

        # Pinned chunks always included; fill remaining slots from deduped
        pinned = [rc for rc in deduped if rc.is_pinned]
        rest = [rc for rc in deduped if not rc.is_pinned]
        final = (pinned + rest)[:_MAX_GRAPH_CHUNKS]

        log.info(
            "graph_retriever_complete",
            section_id=section_id,
            jurisdictions_searched=jurisdictions_to_search,
            chunks_hydrated=len(final),
            pinned=len(pinned),
            deduped_total=len(deduped),
        )
        return final

    @staticmethod
    def _extract_section_id(query: str) -> str | None:
        """
        Extract a section number from natural language query.
        Handles: "Section 18", "section 18A", "s. 18", "sec 18", "Rule 3"
        Returns normalized section_id as stored in DB (e.g. "18", "18A").
        """
        import re
        section_pattern = re.compile(
            r'\b(?:section|sec\.?|s\.)\s*(\d+[A-Z]?)\b',
            re.IGNORECASE,
        )
        rule_pattern = re.compile(
            r'\bRule\s+(\d+[A-Z]?)\b',
            re.IGNORECASE,
        )
        m = section_pattern.search(query)
        if m:
            return m.group(1)
        m = rule_pattern.search(query)
        if m:
            return m.group(1)
        return None
