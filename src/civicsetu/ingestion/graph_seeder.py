from __future__ import annotations

import asyncio
import structlog
from sqlalchemy import text

from civicsetu.stores.graph_store import GraphStore
from civicsetu.stores.relational_store import AsyncSessionLocal

log = structlog.get_logger(__name__)

# Static mapping: (rule_section_id, rule_jurisdiction) → [rera_section_ids]
# Derived from MahaRERA Rules 2017 ↔ RERA Act 2016 structural analysis.
_SECTION_DERIVED_FROM_MAP: list[tuple[str, str, str, str]] = [
    # (rule_id, rule_jurisdiction, act_section_id, act_jurisdiction)
    ("3",  "MAHARASHTRA", "4",  "CENTRAL"),  # Promoter registration info → Sec 4
    ("4",  "MAHARASHTRA", "4",  "CENTRAL"),  # Promoter declaration → Sec 4
    ("5",  "MAHARASHTRA", "4",  "CENTRAL"),  # Separate account → Sec 4
    ("6",  "MAHARASHTRA", "5",  "CENTRAL"),  # Grant/rejection of registration → Sec 5
    ("7",  "MAHARASHTRA", "6",  "CENTRAL"),  # Extension of registration → Sec 6
    ("8",  "MAHARASHTRA", "7",  "CENTRAL"),  # Revocation of registration → Sec 7
    ("10", "MAHARASHTRA", "11", "CENTRAL"),  # Agreement for Sale → Sec 13
    ("11", "MAHARASHTRA", "9",  "CENTRAL"),  # Agent registration application → Sec 9
    ("12", "MAHARASHTRA", "9",  "CENTRAL"),  # Agent grant/rejection → Sec 9
    ("14", "MAHARASHTRA", "10", "CENTRAL"),  # Obligations of real estate agents → Sec 10
    ("15", "MAHARASHTRA", "9",  "CENTRAL"),  # Revocation of agent registration → Sec 9
    ("17", "MAHARASHTRA", "10", "CENTRAL"),  # Other functions of agent → Sec 10
    ("18", "MAHARASHTRA", "19", "CENTRAL"),  # Rate of interest → Sec 19
    ("19", "MAHARASHTRA", "18", "CENTRAL"),  # Timelines for refund → Sec 18
    ("20", "MAHARASHTRA", "11", "CENTRAL"),  # Website disclosures (projects) → Sec 11
    ("21", "MAHARASHTRA", "11", "CENTRAL"),  # Website disclosures (agents) → Sec 11
    ("31", "MAHARASHTRA", "31", "CENTRAL"),  # Complaints to Authority → Sec 31
]

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
            doc_type_str = doc["doc_type"]
            ref_edges = await GraphSeeder._seed_references(
                str(doc["doc_id"]), doc["jurisdiction"], doc_type_str
            )
            edge_count += ref_edges

        # Step 6 — DERIVED_FROM edges (MahaRERA → RERA Act)
        # Resolve doc_ids from loaded docs by jurisdiction + doc_type
        derived_count = 0
        state_docs = [d for d in docs if d["jurisdiction"] != "CENTRAL" and d["doc_type"].upper() == "RULES"]
        central_docs = [d for d in docs if d["jurisdiction"] == "CENTRAL" and d["doc_type"].upper() == "ACT"]

        if state_docs and central_docs:
            for state_doc in state_docs:
                for central_doc in central_docs:
                    derived_count += await GraphSeeder._seed_derived_from(
                        derived_doc_id=str(state_doc["doc_id"]),
                        parent_doc_id=str(central_doc["doc_id"]),
                    )
            log.info("derived_from_total_section_edges", count=derived_count)

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
    async def _seed_references(doc_id: str, jurisdiction: str, doc_type: str) -> int:
        """
        Re-extracts section references from chunk text and creates REFERENCES edges.

        For ACT documents (CENTRAL): intra-doc RERA section refs only.
        For RULES documents (MAHARASHTRA):
        - Intra-doc Rule refs → REFERENCES edges within MAHARASHTRA
        - Cross-doc RERA Act refs → REFERENCES edges to CENTRAL sections
        """
        from civicsetu.ingestion.metadata_extractor import MetadataExtractor

        chunks = await GraphSeeder._load_chunks_with_refs(doc_id)
        total_edges = 0

        for chunk in chunks:
            chunk_id = str(chunk["chunk_id"])
            text = chunk["text"]

            if doc_type.upper() == "RULES":
                # Intra-doc: rule references within same jurisdiction
                rule_refs = MetadataExtractor.extract_rule_references(text)
                if rule_refs:
                    count = await GraphStore.create_references_edges(
                        source_chunk_id=chunk_id,
                        referenced_section_ids=rule_refs,
                        jurisdiction=jurisdiction,
                    )
                    total_edges += count

                # Cross-doc: RERA Act section references → CENTRAL
                rera_refs = MetadataExtractor.extract_section_references(text)
                if rera_refs:
                    count = await GraphStore.create_cross_jurisdiction_references_edges(
                        source_chunk_id=chunk_id,
                        referenced_section_ids=rera_refs,
                        target_jurisdiction="CENTRAL",
                    )
                    total_edges += count

            else:
                # ACT: intra-doc only
                refs = MetadataExtractor.extract_section_references(text)
                if refs:
                    count = await GraphStore.create_references_edges(
                        source_chunk_id=chunk_id,
                        referenced_section_ids=refs,
                        jurisdiction=jurisdiction,
                    )
                    total_edges += count

        return total_edges

    @staticmethod
    async def _seed_derived_from(
        derived_doc_id: str,
        parent_doc_id: str,
    ) -> int:
        """
        Seeds DERIVED_FROM edges:
        1. Document → Document (MahaRERA Rules → RERA Act)
        2. Section → Section   (Rule 3 → Section 4, etc.)
        Returns count of section-level edges created.
        """
        # Document-level edge
        await GraphStore.create_document_derived_from(
            derived_doc_id=derived_doc_id,
            parent_doc_id=parent_doc_id,
        )
        log.info("derived_from_doc_edge_created",
                derived=derived_doc_id, parent=parent_doc_id)

        # Section-level edges — resolve chunk_ids from Postgres
        edge_count = 0
        async with AsyncSessionLocal() as session:
            for rule_id, rule_jur, act_id, act_jur in _SECTION_DERIVED_FROM_MAP:
                # Get rule section chunk_id
                rule_result = await session.execute(
                    text("""
                        SELECT chunk_id FROM legal_chunks
                        WHERE doc_id = :doc_id
                        AND section_id = :section_id
                        AND status = 'active'
                        LIMIT 1
                    """),
                    {"doc_id": derived_doc_id, "section_id": rule_id},
                )
                rule_row = rule_result.fetchone()

                # Get act section chunk_id
                act_result = await session.execute(
                    text("""
                        SELECT chunk_id FROM legal_chunks
                        WHERE doc_id = :doc_id
                        AND section_id = :section_id
                        AND status = 'active'
                        LIMIT 1
                    """),
                    {"doc_id": parent_doc_id, "section_id": act_id},
                )
                act_row = act_result.fetchone()

                if not rule_row or not act_row:
                    log.warning(
                        "derived_from_section_not_found",
                        rule_id=rule_id, act_id=act_id,
                        rule_found=bool(rule_row), act_found=bool(act_row),
                    )
                    continue

                await GraphStore.create_section_derived_from(
                    rule_chunk_id=str(rule_row[0]),
                    act_chunk_id=str(act_row[0]),
                )
                edge_count += 1
                log.info("derived_from_section_edge_created",
                        rule_section=rule_id, act_section=act_id)

        log.info("derived_from_seeding_complete", section_edges=edge_count)
        return edge_count
