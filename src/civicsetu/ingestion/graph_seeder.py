from __future__ import annotations

import asyncio
import structlog
from sqlalchemy import text

from civicsetu.stores.graph_store import GraphStore
from civicsetu.stores.relational_store import AsyncSessionLocal

log = structlog.get_logger(__name__)


class GraphSeeder:
    """
    Reads ingested chunks from PostgreSQL and seeds the Neo4j knowledge graph.

    Run after every ingestion batch — idempotent via MERGE.

    Seeding creates:
      (:Document) nodes
      (:Section)  nodes
      (:Document)-[:HAS_SECTION]->(:Section) edges
      (:Section)-[:REFERENCES]->(:Section)   edges (from MetadataExtractor._section_references)
    """

    @staticmethod
    async def seed_from_postgres(doc_id: str | None = None) -> dict:
        """
        Seed Neo4j from legal_chunks in Postgres.
        If doc_id is provided, seeds only that document.
        If None, seeds all documents.
        Returns counts of nodes and edges created.
        """
        await GraphStore.create_constraints()

        # Step 1 — Load documents
        docs = await GraphSeeder._load_documents(doc_id)
        log.info("graph_seeder_docs_loaded", count=len(docs))

        doc_count = 0
        section_count = 0
        edge_count = 0

        for doc in docs:
            # Step 2 — Upsert Document node
            await GraphStore.upsert_document(
                doc_id=str(doc["doc_id"]),
                doc_name=doc["doc_name"],
                jurisdiction=doc["jurisdiction"],
                doc_type=doc["doc_type"],
                effective_date=str(doc["effective_date"]) if doc["effective_date"] else None,
            )
            doc_count += 1

            # Step 3 — Load chunks for this document
            chunks = await GraphSeeder._load_chunks(str(doc["doc_id"]))
            log.info("graph_seeder_chunks_loaded", doc_name=doc["doc_name"], count=len(chunks))

            for chunk in chunks:
                # Step 4 — Upsert Section node + HAS_SECTION edge
                await GraphStore.upsert_section(
                    chunk_id=str(chunk["chunk_id"]),
                    doc_id=str(doc["doc_id"]),
                    section_id=chunk["section_id"],
                    title=chunk["section_title"],
                    jurisdiction=chunk["jurisdiction"],
                    doc_name=chunk["doc_name"],
                    is_active=chunk["status"] == "active",
                )
                section_count += 1

            # Step 5 — Create REFERENCES edges from extracted references
            ref_edges = await GraphSeeder._seed_references(
                str(doc["doc_id"]), doc["jurisdiction"]
            )
            edge_count += ref_edges

        stats = await GraphStore.graph_stats()
        log.info(
            "graph_seeding_complete",
            docs=doc_count,
            sections=section_count,
            ref_edges=edge_count,
            graph_stats=stats,
        )
        return stats

    @staticmethod
    async def _load_documents(doc_id: str | None) -> list[dict]:
        async with AsyncSessionLocal() as session:
            if doc_id:
                result = await session.execute(
                    text("SELECT * FROM documents WHERE doc_id = :doc_id"),
                    {"doc_id": doc_id},
                )
            else:
                result = await session.execute(text("SELECT * FROM documents WHERE is_active = true"))
            return [dict(row._mapping) for row in result.fetchall()]

    @staticmethod
    async def _load_chunks(doc_id: str) -> list[dict]:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                text("""
                    SELECT chunk_id, section_id, section_title, jurisdiction,
                           doc_name, status
                    FROM legal_chunks
                    WHERE doc_id = :doc_id
                """),
                {"doc_id": doc_id},
            )
            return [dict(row._mapping) for row in result.fetchall()]

    @staticmethod
    async def _load_chunks_with_refs(doc_id: str) -> list[dict]:
        """
        Load chunks joined with their section reference metadata.
        References are stored in a separate table seeded during ingestion.
        Phase 0: MetadataExtractor stores _section_references as transient
        chunk dict keys — not persisted to Postgres.

        For Phase 1 we need them persisted. This method currently re-runs
        the extractor over the stored text to recover references.
        """
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                text("""
                    SELECT chunk_id, section_id, text, jurisdiction
                    FROM legal_chunks
                    WHERE doc_id = :doc_id AND status = 'active'
                """),
                {"doc_id": doc_id},
            )
            return [dict(row._mapping) for row in result.fetchall()]

    @staticmethod
    async def _seed_references(doc_id: str, jurisdiction: str) -> int:
        """
        Re-extracts section references from chunk text and creates REFERENCES edges.
        This is a Phase 1 bridge — in Phase 2 references will be persisted at
        ingestion time in a dedicated chunk_references table.
        """
        from civicsetu.ingestion.metadata_extractor import MetadataExtractor

        chunks = await GraphSeeder._load_chunks_with_refs(doc_id)
        total_edges = 0

        for chunk in chunks:
            refs = MetadataExtractor.extract_section_references(chunk["text"])
            if refs:
                count = await GraphStore.create_references_edges(
                    source_chunk_id=str(chunk["chunk_id"]),
                    referenced_section_ids=refs,
                    jurisdiction=jurisdiction,
                )
                total_edges += count

        return total_edges
