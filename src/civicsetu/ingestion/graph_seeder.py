from __future__ import annotations

import asyncio
import structlog
from sqlalchemy import text

from civicsetu.stores.graph_store import GraphStore
from civicsetu.stores.relational_store import AsyncSessionLocal

log = structlog.get_logger(__name__)

# Static mapping: (rule_section_id, rule_jurisdiction, act_section_id, act_jurisdiction)
# Covers MahaRERA Rules 2017 and UP RERA Rules 2016 → RERA Act 2016.
_SECTION_DERIVED_FROM_MAP: list[tuple[str, str, str, str]] = [
    # ── MahaRERA Rules 2017 → RERA Act 2016 ──────────────────────────────────
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

    # ── UP RERA Rules 2016 → RERA Act 2016 ───────────────────────────────────
    ("3",  "UTTAR_PRADESH", "4",  "CENTRAL"),  # Project registration application → Sec 4
    ("4",  "UTTAR_PRADESH", "4",  "CENTRAL"),  # Promoter declaration → Sec 4
    ("5",  "UTTAR_PRADESH", "4",  "CENTRAL"),  # Separate account obligation → Sec 4
    ("6",  "UTTAR_PRADESH", "5",  "CENTRAL"),  # Grant/rejection of registration → Sec 5
    ("7",  "UTTAR_PRADESH", "6",  "CENTRAL"),  # Extension of registration → Sec 6
    ("8",  "UTTAR_PRADESH", "7",  "CENTRAL"),  # Revocation of registration → Sec 7
    ("9",  "UTTAR_PRADESH", "9",  "CENTRAL"),  # Agent registration application → Sec 9
    ("10", "UTTAR_PRADESH", "9",  "CENTRAL"),  # Agent grant/rejection → Sec 9
    ("11", "UTTAR_PRADESH", "9",  "CENTRAL"),  # Revocation of agent registration → Sec 9
    ("12", "UTTAR_PRADESH", "10", "CENTRAL"),  # Obligations of real estate agents → Sec 10
    ("13", "UTTAR_PRADESH", "11", "CENTRAL"),  # Functions of promoters (website) → Sec 11
    ("14", "UTTAR_PRADESH", "13", "CENTRAL"),  # Agreement for sale → Sec 13
    ("15", "UTTAR_PRADESH", "18", "CENTRAL"),  # Timelines for refund → Sec 18
    ("16", "UTTAR_PRADESH", "19", "CENTRAL"),  # Rate of interest → Sec 19
    ("18", "UTTAR_PRADESH", "31", "CENTRAL"),  # Complaints to Authority → Sec 31
]


class GraphSeeder:
    """
    Reads ingested chunks from PostgreSQL and seeds the Neo4j knowledge graph.
    Run after every ingestion batch — idempotent via MERGE.
    """

    @staticmethod
    async def seed_from_postgres(doc_id: str | None = None) -> dict:
        await GraphStore.create_constraints()

        # Step 1 — Load target documents (1 if doc_id given, all if None)
        docs = await GraphSeeder._load_documents(doc_id)
        log.info("graph_seeder_docs_loaded", count=len(docs))

        doc_count = 0
        section_count = 0
        edge_count = 0

        for doc in docs:
            await GraphStore.upsert_document(
                doc_id=str(doc["doc_id"]),
                doc_name=doc["doc_name"],
                jurisdiction=doc["jurisdiction"],
                doc_type=doc["doc_type"],
                effective_date=str(doc["effective_date"]) if doc["effective_date"] else None,
            )
            doc_count += 1

            chunks = await GraphSeeder._load_chunks(str(doc["doc_id"]))
            log.info("graph_seeder_chunks_loaded", doc_name=doc["doc_name"], count=len(chunks))

            for chunk in chunks:
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

            ref_edges = await GraphSeeder._seed_references(
                str(doc["doc_id"]), doc["jurisdiction"], doc["doc_type"]
            )
            edge_count += ref_edges

        # Step 6 — DERIVED_FROM edges
        # ── BUG FIX: when doc_id is given, `docs` has 1 entry.
        # Central act docs would never appear → edges silently skipped.
        # Load ALL active docs unconditionally to resolve central act doc_ids.
        all_active_docs = await GraphSeeder._load_documents(None)

        state_docs = [
            d for d in docs
            if d["jurisdiction"] != "CENTRAL" and d["doc_type"].upper() == "RULES"
        ]
        central_docs = [
            d for d in all_active_docs
            if d["jurisdiction"] == "CENTRAL" and d["doc_type"].upper() == "ACT"
        ]

        if state_docs and central_docs:
            derived_count = 0
            for state_doc in state_docs:
                for central_doc in central_docs:
                    derived_count += await GraphSeeder._seed_derived_from(
                        derived_doc_id=str(state_doc["doc_id"]),
                        rule_jurisdiction=state_doc["jurisdiction"],
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
                result = await session.execute(
                    text("SELECT * FROM documents WHERE is_active = true")
                )
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
        from civicsetu.ingestion.metadata_extractor import MetadataExtractor

        chunks = await GraphSeeder._load_chunks_with_refs(doc_id)
        total_edges = 0

        for chunk in chunks:
            chunk_id = str(chunk["chunk_id"])
            chunk_text = chunk["text"]

            if doc_type.upper() == "RULES":
                rule_refs = MetadataExtractor.extract_rule_references(chunk_text)
                if rule_refs:
                    total_edges += await GraphStore.create_references_edges(
                        source_chunk_id=chunk_id,
                        referenced_section_ids=rule_refs,
                        jurisdiction=jurisdiction,
                    )
                rera_refs = MetadataExtractor.extract_section_references(chunk_text)
                if rera_refs:
                    total_edges += await GraphStore.create_cross_jurisdiction_references_edges(
                        source_chunk_id=chunk_id,
                        referenced_section_ids=rera_refs,
                        target_jurisdiction="CENTRAL",
                    )
            else:
                refs = MetadataExtractor.extract_section_references(chunk_text)
                if refs:
                    total_edges += await GraphStore.create_references_edges(
                        source_chunk_id=chunk_id,
                        referenced_section_ids=refs,
                        jurisdiction=jurisdiction,
                    )

        return total_edges

    @staticmethod
    async def _seed_derived_from(
        derived_doc_id: str,
        rule_jurisdiction: str,
        parent_doc_id: str,
    ) -> int:
        """
        Seeds DERIVED_FROM edges for a specific state rules doc → central act doc.
        Filters _SECTION_DERIVED_FROM_MAP by rule_jurisdiction so Maharashtra
        entries never fire when seeding UP RERA and vice versa.
        """
        await GraphStore.create_document_derived_from(
            derived_doc_id=derived_doc_id,
            parent_doc_id=parent_doc_id,
        )
        log.info("derived_from_doc_edge_created",
                 derived=derived_doc_id, parent=parent_doc_id,
                 jurisdiction=rule_jurisdiction)

        # Filter map to only this jurisdiction's entries
        relevant_mappings = [
            (rule_id, act_id)
            for rule_id, rule_jur, act_id, act_jur in _SECTION_DERIVED_FROM_MAP
            if rule_jur == rule_jurisdiction
        ]
        log.info("derived_from_mappings_loaded",
                 jurisdiction=rule_jurisdiction, count=len(relevant_mappings))

        edge_count = 0
        async with AsyncSessionLocal() as session:
            for rule_id, act_id in relevant_mappings:
                rule_result = await session.execute(
                    text("""
                        SELECT chunk_id FROM legal_chunks
                        WHERE doc_id = :doc_id AND section_id = :section_id
                        AND status = 'active' LIMIT 1
                    """),
                    {"doc_id": derived_doc_id, "section_id": rule_id},
                )
                rule_row = rule_result.fetchone()

                act_result = await session.execute(
                    text("""
                        SELECT chunk_id FROM legal_chunks
                        WHERE doc_id = :doc_id AND section_id = :section_id
                        AND status = 'active' LIMIT 1
                    """),
                    {"doc_id": parent_doc_id, "section_id": act_id},
                )
                act_row = act_result.fetchone()

                if not rule_row or not act_row:
                    log.warning(
                        "derived_from_section_not_found",
                        jurisdiction=rule_jurisdiction,
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
                         jurisdiction=rule_jurisdiction,
                         rule_section=rule_id, act_section=act_id)

        log.info("derived_from_seeding_complete",
                 jurisdiction=rule_jurisdiction, section_edges=edge_count)
        return edge_count
