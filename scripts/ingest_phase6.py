# scripts/ingest_phase6.py
"""
Phase 6 ingestion — Tamil Nadu RERA:
  1. Tamil Nadu RERA Rules 2017

Run:
    uv run python scripts/ingest_phase6.py
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


def main() -> None:
    pipeline = IngestionPipeline()
    spec = DOCUMENT_REGISTRY["tn_rera_rules_2017"]

    log.info("phase6_ingest_start", doc=spec.name)
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
    log.info("phase6_ingest_complete",
             doc=spec.name, chunks=doc.total_chunks, doc_id=str(doc.doc_id))

    if doc.total_chunks == 0:
        print("\n✗ ABORT: 0 chunks — PDF likely scanned or wrong format.")
        return

    stats = asyncio.run(GraphSeeder.seed_from_postgres(doc_id=str(doc.doc_id)))

    print("\n" + "=" * 60)
    print("Phase 6 — Tamil Nadu RERA ingestion complete")
    print(f"  Chunks       : {doc.total_chunks}")
    print(f"  Doc ID       : {doc.doc_id}")
    print(f"  Graph stats  : {stats}")
    print("=" * 60)

    if doc.total_chunks < 10:
        print("\n⚠ WARNING: Very few chunks — check scanned_pages in logs")
    else:
        print("\n✓ Tamil Nadu wired.")


if __name__ == "__main__":
    main()
