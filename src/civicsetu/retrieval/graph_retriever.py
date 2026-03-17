from __future__ import annotations

import asyncio
import structlog

from civicsetu.models.enums import Jurisdiction
from civicsetu.models.schemas import RetrievedChunk
from civicsetu.stores.graph_store import GraphStore
from civicsetu.stores.relational_store import AsyncSessionLocal
from civicsetu.stores.vector_store import VectorStore

log = structlog.get_logger(__name__)


class GraphRetriever:
    """
    Retrieves chunks via Neo4j graph traversal rather than vector similarity.

    Used for:
      - cross_reference queries: "What does Section 18 reference?"
      - temporal queries: "Which sections were amended?" (Phase 2)

    Strategy:
      1. Extract section_id mentioned in query (via regex or LLM — Phase 1 uses regex)
      2. Traverse REFERENCES edges in Neo4j to find related section_ids
      3. Hydrate those section_ids into full LegalChunk objects via VectorStore.get_by_section()
      4. Return as RetrievedChunk list — identical interface to VectorStore output
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

        jurisdiction_str = jurisdiction.value if jurisdiction else "CENTRAL"
        log.info("graph_retriever_traversing", section_id=section_id, depth=depth)

        chunks: list[RetrievedChunk] = []

        async with AsyncSessionLocal() as session:
            # 1 — Source section itself
            source_chunks = await VectorStore.get_by_section(
                session=session, section_id=section_id, jurisdiction=jurisdiction,
            )
            for rc in source_chunks:
                rc.retrieval_source = "graph"
                rc.graph_path = f"source:{section_id}"
            chunks.extend(source_chunks)

            # 2 — Outgoing: sections that section_id cites
            outgoing = await GraphStore.get_referenced_sections(
                section_id=section_id, jurisdiction=jurisdiction_str, depth=depth,
            )
            for node in outgoing:
                hydrated = await VectorStore.get_by_section(
                    session=session, section_id=node["section_id"], jurisdiction=jurisdiction,
                )
                for rc in hydrated:
                    rc.retrieval_source = "graph"
                    rc.graph_path = f"{section_id} →[REFERENCES]→ {node['section_id']}"
                chunks.extend(hydrated)

            # 3 — Incoming: sections that cite section_id
            incoming = await GraphStore.get_sections_referencing(
                section_id=section_id, jurisdiction=jurisdiction_str,
            )
            for node in incoming:
                hydrated = await VectorStore.get_by_section(
                    session=session, section_id=node["section_id"], jurisdiction=jurisdiction,
                )
                for rc in hydrated:
                    rc.retrieval_source = "graph"
                    rc.graph_path = f"{node['section_id']} →[REFERENCES]→ {section_id}"
                chunks.extend(hydrated)

        log.info(
            "graph_retriever_complete",
            section_id=section_id,
            outgoing=len(outgoing),
            incoming=len(incoming),
            chunks_hydrated=len(chunks),
        )
        return chunks

    @staticmethod
    def _extract_section_id(query: str) -> str | None:
        """
        Extract a section number from natural language query.
        Handles: "Section 18", "section 18A", "s. 18", "sec 18"
        Returns normalized section_id as it appears in the DB (e.g. "18", "18A").
        """
        import re
        pattern = re.compile(
            r'\b(?:section|sec\.?|s\.)\s*(\d+[A-Z]?)\b',
            re.IGNORECASE,
        )
        m = pattern.search(query)
        return m.group(1) if m else None
