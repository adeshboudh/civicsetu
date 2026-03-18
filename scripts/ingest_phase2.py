# scripts/ingest_phase2.py
"""
Phase 2 ingestion: Maharashtra Real Estate Rules 2017.

Run:
    uv run python scripts/ingest_phase2.py

What it does:
    1. Downloads MahaRERA Rules 2017 PDF (cached after first run)
    2. Chunks using Rule-boundary regex
    3. Embeds + persists to Postgres/pgvector
    4. Re-seeds Neo4j (adds MahaRERA Section nodes + REFERENCES + DERIVED_FROM edges)
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Ensure src/ is on PYTHONPATH when run directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import structlog

from civicsetu.config.document_registry import DOCUMENT_REGISTRY
from civicsetu.ingestion.pipeline import IngestionPipeline
from civicsetu.ingestion.graph_seeder import GraphSeeder
from civicsetu.stores.graph_store import _get_driver

log = structlog.get_logger(__name__)


def main() -> None:
    spec = DOCUMENT_REGISTRY["mahrera_rules_2017"]

    log.info("phase2_ingest_start", doc=spec.name, url=spec.url)

    # Step 1 — Ingest into Postgres + pgvector
    pipeline = IngestionPipeline()
    doc = pipeline.ingest_document(
        source_url=spec.url,
        doc_name=spec.name,
        jurisdiction=spec.jurisdiction,
        doc_type=spec.doc_type,
        effective_date=spec.effective_date,
        dest_subdir=spec.dest_subdir,
        filename=spec.filename,
    )

    log.info("phase2_ingest_complete", doc_id=str(doc.doc_id), chunks=doc.total_chunks)

    # Step 2 — Re-seed Neo4j for this document only
    log.info("phase2_graph_seed_start", doc_id=str(doc.doc_id))
    stats = asyncio.run(GraphSeeder.seed_from_postgres(doc_id=str(doc.doc_id)))
    log.info("phase2_graph_seed_complete", stats=stats)

    # Step 3 — Print summary
    print("\n" + "="*60)
    print("Phase 2 ingestion complete")
    print(f"   Document  : {spec.name}")
    print(f"   Doc ID    : {doc.doc_id}")
    print(f"   Chunks    : {doc.total_chunks}")
    print(f"   Sections  : {stats.get('sections')}")
    print(f"   REFERENCES: {stats.get('refs')}")
    print("="*60)



if __name__ == "__main__":
    main()
