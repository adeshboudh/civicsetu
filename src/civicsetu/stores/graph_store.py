from __future__ import annotations

from functools import lru_cache
from typing import Optional
from uuid import UUID

import structlog
from neo4j import AsyncGraphDatabase, AsyncDriver

from civicsetu.config.settings import get_settings

log = structlog.get_logger(__name__)
settings = get_settings()


@lru_cache(maxsize=1)
def _get_driver() -> AsyncDriver:
    """Singleton async Neo4j driver — one per process lifetime."""
    return AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
        notifications_min_severity="OFF",
    )


class GraphStore:
    """
    All Neo4j operations for CivicSetu.

    Graph schema:
      (:Document {doc_id, doc_name, jurisdiction, doc_type, effective_date})
      (:Section  {section_id, title, chunk_id, jurisdiction, doc_name, is_active})
      (:Document)-[:HAS_SECTION]->(:Section)
      (:Section) -[:REFERENCES]->(:Section)
      (:Section) -[:SUPERSEDES]->(:Section)   Phase 1
      (:Section) -[:AMENDED_BY]->(:Section)   Phase 1
    """

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    @staticmethod
    async def ping() -> bool:
        """Health check — verify Neo4j is reachable."""
        try:
            driver = _get_driver()
            await driver.verify_connectivity()
            return True
        except Exception as e:
            log.error("neo4j_ping_failed", error=str(e))
            return False

    @staticmethod
    async def close() -> None:
        """Close the driver — call on app shutdown."""
        driver = _get_driver()
        await driver.close()

    # ── Schema constraints ────────────────────────────────────────────────────

    @staticmethod
    async def create_constraints() -> None:
        """
        Idempotent — safe to run on every startup.
        Creates uniqueness constraints and indexes for fast lookup.
        """
        driver = _get_driver()
        async with driver.session() as session:
            constraints = [
                "CREATE CONSTRAINT doc_id_unique IF NOT EXISTS "
                "FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",

                "CREATE CONSTRAINT section_chunk_id_unique IF NOT EXISTS "
                "FOR (s:Section) REQUIRE s.chunk_id IS UNIQUE",

                "CREATE INDEX section_id_idx IF NOT EXISTS "
                "FOR (s:Section) ON (s.section_id)",

                "CREATE INDEX section_jurisdiction_idx IF NOT EXISTS "
                "FOR (s:Section) ON (s.jurisdiction)",
            ]
            for cypher in constraints:
                await session.run(cypher)

        log.info("neo4j_constraints_created")

    # ── Write operations ──────────────────────────────────────────────────────

    @staticmethod
    async def upsert_document(
        doc_id: str,
        doc_name: str,
        jurisdiction: str,
        doc_type: str,
        effective_date: str | None,
    ) -> None:
        """Create or update a Document node."""
        driver = _get_driver()
        async with driver.session() as session:
            await session.run(
                """
                MERGE (d:Document {doc_id: $doc_id})
                SET d.doc_name       = $doc_name,
                    d.jurisdiction   = $jurisdiction,
                    d.doc_type       = $doc_type,
                    d.effective_date = $effective_date
                """,
                doc_id=doc_id,
                doc_name=doc_name,
                jurisdiction=jurisdiction,
                doc_type=doc_type,
                effective_date=effective_date,
            )
        log.info("neo4j_document_upserted", doc_id=doc_id, doc_name=doc_name)

    @staticmethod
    async def upsert_section(
        chunk_id: str,
        doc_id: str,
        section_id: str,
        title: str,
        jurisdiction: str,
        doc_name: str,
        is_active: bool = True,
    ) -> None:
        """
        Create or update a Section node and wire HAS_SECTION edge to its Document.
        MERGE on chunk_id — each chunk maps to exactly one Section node.
        """
        driver = _get_driver()
        async with driver.session() as session:
            await session.run(
                """
                MERGE (s:Section {chunk_id: $chunk_id})
                SET s.section_id   = $section_id,
                    s.title        = $title,
                    s.jurisdiction = $jurisdiction,
                    s.doc_name     = $doc_name,
                    s.is_active    = $is_active

                WITH s
                MATCH (d:Document {doc_id: $doc_id})
                MERGE (d)-[:HAS_SECTION]->(s)
                """,
                chunk_id=chunk_id,
                doc_id=doc_id,
                section_id=section_id,
                title=title,
                jurisdiction=jurisdiction,
                doc_name=doc_name,
                is_active=is_active,
            )

    @staticmethod
    async def create_references_edges(
        source_chunk_id: str,
        referenced_section_ids: list[str],
        jurisdiction: str,
    ) -> int:
        """
        Creates REFERENCES edges from a Section to every section it cites.
        Matches target sections by section_id + jurisdiction to avoid
        cross-document false matches.
        Returns count of edges created.
        """
        if not referenced_section_ids:
            return 0

        driver = _get_driver()
        count = 0
        async with driver.session() as session:
            for ref_id in referenced_section_ids:
                result = await session.run(
                    """
                    MATCH (src:Section {chunk_id: $source_chunk_id})
                    MATCH (tgt:Section {section_id: $ref_id, jurisdiction: $jurisdiction})
                    MERGE (src)-[r:REFERENCES]->(tgt)
                    RETURN count(r) AS created
                    """,
                    source_chunk_id=source_chunk_id,
                    ref_id=ref_id,
                    jurisdiction=jurisdiction,
                )
                record = await result.single()
                if record:
                    count += record["created"]

        return count

    @staticmethod
    async def delete_document_graph(doc_id: str) -> None:
        """
        Hard delete all Section nodes and edges for a document.
        Used during re-ingestion — keeps graph in sync with pgvector.
        """
        driver = _get_driver()
        async with driver.session() as session:
            await session.run(
                """
                MATCH (d:Document {doc_id: $doc_id})-[:HAS_SECTION]->(s:Section)
                DETACH DELETE s
                """,
                doc_id=doc_id,
            )
            await session.run(
                "MATCH (d:Document {doc_id: $doc_id}) DETACH DELETE d",
                doc_id=doc_id,
            )
        log.info("neo4j_document_deleted", doc_id=doc_id)

    # ── Read operations ───────────────────────────────────────────────────────

    @staticmethod
    async def get_referenced_sections(
        section_id: str,
        jurisdiction: str,
        depth: int = 1,
    ) -> list[dict]:
        """
        Traverse REFERENCES edges up to `depth` hops from a given section.
        Returns list of {section_id, title, chunk_id, jurisdiction}.
        Used by graph_retriever in Phase 1.
        """
        driver = _get_driver()
        async with driver.session() as session:
            result = await session.run(
                f"""
                MATCH (src:Section {{section_id: $section_id, jurisdiction: $jurisdiction}})
                      -[:REFERENCES*1..{depth}]->(tgt:Section)
                WHERE tgt.is_active = true
                RETURN DISTINCT tgt.section_id   AS section_id,
                                tgt.title        AS title,
                                tgt.chunk_id     AS chunk_id,
                                tgt.jurisdiction AS jurisdiction
                """,
                section_id=section_id,
                jurisdiction=jurisdiction,
            )
            records = await result.data()
        return records

    @staticmethod
    async def get_sections_for_document(doc_id: str) -> list[dict]:
        """Return all Section nodes for a document — used in graph seeding verification."""
        driver = _get_driver()
        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (d:Document {doc_id: $doc_id})-[:HAS_SECTION]->(s:Section)
                RETURN s.section_id AS section_id, s.title AS title,
                       s.chunk_id AS chunk_id
                ORDER BY s.section_id
                """,
                doc_id=doc_id,
            )
            return await result.data()
        
    @staticmethod
    async def get_sections_referencing(
        section_id: str,
        jurisdiction: str,
    ) -> list[dict]:
        """Incoming REFERENCES edges — sections that cite this section."""
        driver = _get_driver()
        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (src:Section)-[:REFERENCES]->(tgt:Section {section_id: $section_id, jurisdiction: $jurisdiction})
                WHERE src.is_active = true
                RETURN DISTINCT src.section_id   AS section_id,
                                src.title        AS title,
                                src.chunk_id     AS chunk_id,
                                src.jurisdiction AS jurisdiction
                """,
                section_id=section_id,
                jurisdiction=jurisdiction,
            )
            return await result.data()

    @staticmethod
    async def graph_stats() -> dict:
        driver = _get_driver()
        async with driver.session() as session:
            result = await session.run(
                """
                RETURN
                    count { MATCH (d:Document) RETURN d }      AS docs,
                    count { MATCH (s:Section) RETURN s }       AS sections,
                    count { MATCH ()-[:REFERENCES]->() RETURN 1 }  AS refs,
                    count { MATCH ()-[:HAS_SECTION]->() RETURN 1 } AS has_sec
                """
            )
            record = await result.single()
            return dict(record) if record else {}

