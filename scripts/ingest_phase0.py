"""
Phase 0 ingestion: RERA Act 2016 (Central).
Idempotent — safe to re-run, upserts on conflict.

Run:
    uv run python scripts/ingest_phase0.py
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


def main() -> None:
    spec = DOCUMENT_REGISTRY["rera_act_2016"]

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

    log.info("phase0_ingest_complete", doc_id=str(doc.doc_id), chunks=doc.total_chunks)
    stats = asyncio.run(GraphSeeder.seed_from_postgres(doc_id=str(doc.doc_id)))
    log.info("phase0_graph_seed_complete", stats=stats)


if __name__ == "__main__":
    main()
