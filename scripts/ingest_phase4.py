# scripts/ingest_phase4.py
"""
Phase 4 ingestion — UP RERA (two documents):
  1. UP RERA Rules 2016           (pages 1-24 only; pages 25-52 are prescribed forms)
  2. UP RERA General Regulations 2019

Run:
    uv run python scripts/ingest_phase4.py

What it does:
1. Downloads UP RERA Rules 2016 PDF from up-rera.in (cached after first run)
2. Chunks using Rule-boundary regex (same chunker as MahaRERA Rules)
3. Embeds + persists to Postgres/pgvector under UTTAR_PRADESH jurisdiction
4. Seeds Neo4j (adds UP RERA Section nodes + REFERENCES + DERIVED_FROM edges)

This is the first multi-state expansion. Architecture proof:
- No new store interfaces needed — UTTAR_PRADESH already in Jurisdiction enum
- Chunker handles Rule-pattern documents without modification
- Graph seeder seeds cross-jurisdiction DERIVED_FROM edges automatically
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import structlog

from civicsetu.config.document_registry import DOCUMENT_REGISTRY
from civicsetu.ingestion.pipeline import IngestionPipeline
from civicsetu.ingestion.graph_seeder import GraphSeeder

log = structlog.get_logger(__name__)

def ingest_spec(pipeline: IngestionPipeline, key: str) -> object:
    spec = DOCUMENT_REGISTRY[key]
    log.info("phase4_ingest_start", doc=spec.name)
    doc = pipeline.ingest_document(
        source_url=spec.url,
        doc_name=spec.name,
        jurisdiction=spec.jurisdiction,
        doc_type=spec.doc_type,
        effective_date=spec.effective_date,
        dest_subdir=spec.dest_subdir,
        filename=spec.filename,
        max_pages=spec.max_pages,
    )
    log.info("phase4_ingest_complete", doc=spec.name, chunks=doc.total_chunks, doc_id=str(doc.doc_id))
    return doc

def main() -> None:
    pipeline = IngestionPipeline()

    # ── Document 1: UP RERA Rules 2016 (pages 1-24) ───────────────────────────
    doc_rules = ingest_spec(pipeline, "up_rera_rules_2016")

    # ── Document 2: UP RERA General Regulations 2019 ─────────────────────────
    doc_regs = ingest_spec(pipeline, "up_rera_general_regulations_2019")

    # ── Graph seed both documents ─────────────────────────────────────────────
    for doc in [doc_rules, doc_regs]:
        log.info("phase4_graph_seed_start", doc_id=str(doc.doc_id))
        stats = asyncio.run(GraphSeeder.seed_from_postgres(doc_id=str(doc.doc_id)))
        log.info("phase4_graph_seed_complete", stats=stats)

    # ── Smoke test: UP→CENTRAL DERIVED_FROM edges ─────────────────────────────
    from civicsetu.stores.graph_store import _get_driver

    async def _smoke():
        driver = _get_driver()
        async with driver.session() as s:
            r = await s.run("""
                MATCH (r:Section {jurisdiction: 'UTTAR_PRADESH'})-[:DERIVED_FROM]->(a:Section)
                RETURN count(*) AS cnt
            """)
            row = await r.single()
            return row["cnt"]

    up_edges = asyncio.run(_smoke())

    print("\n" + "=" * 60)
    print("Phase 4 ingestion complete")
    print(f"  UP RERA Rules 2016   : {doc_rules.total_chunks} chunks")
    print(f"  UP RERA Regs 2019    : {doc_regs.total_chunks} chunks")
    print(f"  DERIVED_FROM (UP→CENTRAL): {up_edges}")
    print("=" * 60)

    if doc_rules.total_chunks == 0:
        print("\n⚠ WARNING: Rules doc has 0 chunks — max_pages=24 may be cutting too early")
    elif up_edges == 0:
        print("\n⚠ WARNING: No DERIVED_FROM edges — check _SECTION_DERIVED_FROM_MAP section IDs")
    else:
        print("\n✓ UP RERA wired. Multi-state expansion complete.")


if __name__ == "__main__":
    main()