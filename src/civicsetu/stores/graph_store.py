from __future__ import annotations

import asyncio
from typing import Optional
from uuid import UUID

import structlog
from neo4j import AsyncGraphDatabase, AsyncDriver

from civicsetu.config.settings import get_settings

log = structlog.get_logger(__name__)

_driver: AsyncDriver | None = None
_driver_lock: asyncio.Lock = asyncio.Lock()


async def _get_driver() -> AsyncDriver:
    """Cached driver with lazy init — avoids 25+ driver creations per graph retrieval."""
    global _driver
    if _driver is not None:
        return _driver
    async with _driver_lock:
        if _driver is not None:
            return _driver
        settings = get_settings()
        _driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        log.info("neo4j_driver_initialized")
    return _driver


async def close_driver() -> None:
    """Close the cached driver. Call on app shutdown."""
    global _driver
    async with _driver_lock:
        if _driver is not None:
            await _driver.close()
            _driver = None
            log.info("neo4j_driver_closed")


class GraphStore:
    """
    All Neo4j operations for CivicSetu.

    Graph schema:
    (:Document {doc_id, doc_name, jurisdiction, doc_type, effective_date})
    (:Section  {section_id, title, chunk_id, jurisdiction, doc_name, is_active})
    (:Document)-[:HAS_SECTION]->(:Section)
    (:Section) -[:REFERENCES]->(:Section)
    (:Section) -[:DERIVED_FROM]->(:Section)
    (:Document)-[:DERIVED_FROM]->(:Document)
    """

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    @staticmethod
    async def ping() -> bool:
        try:
            driver = await _get_driver()
            await driver.verify_connectivity()
            return True
        except Exception as e:
            log.error("neo4j_ping_failed", error=str(e))
            return False

    @staticmethod
    async def close() -> None:
        """Delegate to module-level close_driver()."""
        await close_driver()

    # ── Schema constraints ────────────────────────────────────────────────────

    @staticmethod
    async def create_constraints() -> None:
        driver = await _get_driver()
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
        driver = await _get_driver()
        async with driver.session() as session:
            await session.run(
                """
                MERGE (d:Document {doc_id: $doc_id})
                SET d.doc_name = $doc_name,
                    d.jurisdiction = $jurisdiction,
                    d.doc_type = $doc_type,
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
        driver = await _get_driver()
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
        if not referenced_section_ids:
            return 0

        driver = await _get_driver()
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
    async def create_cross_jurisdiction_references_edges(
        source_chunk_id: str,
        referenced_section_ids: list[str],
        target_jurisdiction: str,
    ) -> int:
        if not referenced_section_ids:
            return 0

        driver = await _get_driver()
        count = 0
        async with driver.session() as session:
            for ref_id in referenced_section_ids:
                result = await session.run(
                    """
                    MATCH (src:Section {chunk_id: $source_chunk_id})
                    MATCH (tgt:Section {section_id: $ref_id, jurisdiction: $target_jurisdiction})
                    MERGE (src)-[r:REFERENCES]->(tgt)
                    RETURN count(r) AS created
                    """,
                    source_chunk_id=source_chunk_id,
                    ref_id=ref_id,
                    target_jurisdiction=target_jurisdiction,
                )
                record = await result.single()
                if record:
                    count += record["created"]
        return count

    @staticmethod
    async def create_document_derived_from(
        derived_doc_id: str,
        parent_doc_id: str,
    ) -> None:
        driver = await _get_driver()
        async with driver.session() as session:
            await session.run(
                """
                MATCH (child:Document {doc_id: $derived_doc_id})
                MATCH (parent:Document {doc_id: $parent_doc_id})
                MERGE (child)-[:DERIVED_FROM]->(parent)
                """,
                derived_doc_id=derived_doc_id,
                parent_doc_id=parent_doc_id,
            )
        log.info(
            "neo4j_document_derived_from_created",
            derived_doc_id=derived_doc_id,
            parent_doc_id=parent_doc_id,
        )

    @staticmethod
    async def create_section_derived_from(
        rule_chunk_id: str,
        act_chunk_id: str,
    ) -> None:
        driver = await _get_driver()
        async with driver.session() as session:
            await session.run(
                """
                MATCH (rule_sec:Section {chunk_id: $rule_chunk_id})
                MATCH (act_sec:Section {chunk_id: $act_chunk_id})
                MERGE (rule_sec)-[:DERIVED_FROM]->(act_sec)
                """,
                rule_chunk_id=rule_chunk_id,
                act_chunk_id=act_chunk_id,
            )

    @staticmethod
    async def delete_document_graph(doc_id: str) -> None:
        driver = await _get_driver()
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
        driver = await _get_driver()          # ← add await
        async with driver.session() as session:
            result = await session.run(
                f"""
                MATCH (src:Section {{section_id: $section_id, jurisdiction: $jurisdiction}})
                -[:REFERENCES*1..{depth}]->(tgt:Section)
                WHERE tgt.is_active = true
                RETURN DISTINCT tgt.section_id AS section_id,
                    tgt.title AS title,
                    tgt.chunk_id AS chunk_id,
                    tgt.jurisdiction AS jurisdiction
                """,
                section_id=section_id,
                jurisdiction=jurisdiction,
            )
            return await result.data()

    @staticmethod
    async def get_sections_referencing(
        section_id: str,
        jurisdiction: str,
    ) -> list[dict]:
        driver = await _get_driver()
        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (src:Section)-[:REFERENCES]->
                (tgt:Section {section_id: $section_id, jurisdiction: $jurisdiction})
                WHERE src.is_active = true
                RETURN DISTINCT src.section_id AS section_id,
                    src.title AS title,
                    src.chunk_id AS chunk_id,
                    src.jurisdiction AS jurisdiction
                """,
                section_id=section_id,
                jurisdiction=jurisdiction,
            )
            return await result.data()

    @staticmethod
    async def get_derived_act_sections(
        rule_section_id: str,
        rule_jurisdiction: str,
    ) -> list[dict]:
        driver = await _get_driver()
        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (rule_sec:Section {section_id: $section_id, jurisdiction: $jurisdiction})
                -[:DERIVED_FROM]->(act_sec:Section)
                WHERE act_sec.is_active = true
                RETURN DISTINCT act_sec.section_id AS section_id,
                    act_sec.title AS title,
                    act_sec.chunk_id AS chunk_id,
                    act_sec.jurisdiction AS jurisdiction,
                    act_sec.doc_name AS doc_name
                """,
                section_id=rule_section_id,
                jurisdiction=rule_jurisdiction,
            )
            return await result.data()

    @staticmethod
    async def get_deriving_rule_sections(
        act_section_id: str,
        act_jurisdiction: str = "CENTRAL",
    ) -> list[dict]:
        driver = await _get_driver()
        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (rule_sec:Section)-[:DERIVED_FROM]->
                (act_sec:Section {section_id: $section_id, jurisdiction: $jurisdiction})
                WHERE rule_sec.is_active = true
                RETURN DISTINCT rule_sec.section_id AS section_id,
                    rule_sec.title AS title,
                    rule_sec.chunk_id AS chunk_id,
                    rule_sec.jurisdiction AS jurisdiction,
                    rule_sec.doc_name AS doc_name
                """,
                section_id=act_section_id,
                jurisdiction=act_jurisdiction,
            )
            return await result.data()

    @staticmethod
    async def get_sections_for_document(doc_id: str) -> list[dict]:
        driver = await _get_driver()
        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (d:Document {doc_id: $doc_id})-[:HAS_SECTION]->(s:Section)
                RETURN s.section_id AS section_id,
                        s.title      AS title,
                        s.chunk_id   AS chunk_id
                ORDER BY s.section_id
                """,
                doc_id=doc_id,
            )
            return await result.data()

    @staticmethod
    async def graph_stats() -> dict:
        driver = await _get_driver()
        async with driver.session() as session:
            result = await session.run(
                """
                RETURN
                count { MATCH (d:Document)          RETURN d } AS docs,
                count { MATCH (s:Section)           RETURN s } AS sections,
                count { MATCH ()-[:REFERENCES]->()  RETURN 1 } AS refs,
                count { MATCH ()-[:HAS_SECTION]->() RETURN 1 } AS has_sec,
                count { MATCH ()-[:DERIVED_FROM]->() RETURN 1 } AS derived_from
                """
            )
            record = await result.single()
            return dict(record) if record else {}
