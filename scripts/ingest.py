"""
Ingest all CivicSetu documents across all jurisdictions.
Reads from document_registry — single source of truth.

Usage:
    uv run python scripts/ingest.py
    uv run python scripts/ingest.py --jurisdiction MAHARASHTRA
    uv run python scripts/ingest.py --dry-run
"""

import argparse
import asyncio
import sys
import time

import structlog

from civicsetu.config.document_registry import DOCUMENT_REGISTRY
from civicsetu.ingestion.pipeline import IngestionPipeline
from civicsetu.ingestion.graph_seeder import GraphSeeder

log = structlog.get_logger(__name__)


async def ingest_all(jurisdiction_filter: str | None = None, dry_run: bool = False):
    docs = list(DOCUMENT_REGISTRY.values())
    if jurisdiction_filter:
        docs = [d for d in docs if d.jurisdiction.value == jurisdiction_filter.upper()]
        if not docs:
            log.error("no_docs_found", jurisdiction=jurisdiction_filter)
            sys.exit(1)

    log.info("ingest_start", total_docs=len(docs), dry_run=dry_run)

    if dry_run:
        for doc in docs:
            print(f"  [{doc.jurisdiction.value}] {doc.name}")
        return

    pipeline = IngestionPipeline()
    results = {"success": [], "failed": []}

    for i, doc in enumerate(docs, 1):
        log.info("ingesting_document", index=i, total=len(docs),
                 jurisdiction=doc.jurisdiction.value, doc_name=doc.name)
        t0 = time.perf_counter()
        try:
            pipeline.ingest_document(
                source_url=doc.url,
                doc_name=doc.name,
                jurisdiction=doc.jurisdiction,
                doc_type=doc.doc_type,
                effective_date=doc.effective_date,
                dest_subdir=doc.dest_subdir,
                filename=doc.filename,
                max_pages=doc.max_pages,
            )
            elapsed = time.perf_counter() - t0
            log.info("ingestion_complete", doc_name=doc.name, elapsed_s=round(elapsed, 1))
            results["success"].append(doc.name)
        except Exception as e:
            elapsed = time.perf_counter() - t0
            log.error("ingestion_failed", doc_name=doc.name,
                      error=str(e), elapsed_s=round(elapsed, 1))
            results["failed"].append((doc.name, str(e)))

    log.info("seeding_graph_edges")
    try:
        seeder = GraphSeeder()
        await seeder.seed_all()
        log.info("graph_seeding_complete")
    except Exception as e:
        log.error("graph_seeding_failed", error=str(e))
        results["failed"].append(("graph_seeder", str(e)))

    # Summary
    print(f"\n{'='*50}")
    print(f"Ingestion complete: {len(results['success'])}/{len(docs)} succeeded")
    for name in results["success"]:
        print(f"  ✓ {name}")
    if results["failed"]:
        print(f"\nFailed ({len(results['failed'])}):")
        for name, err in results["failed"]:
            print(f"  ✗ {name}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest CivicSetu documents")
    parser.add_argument("--jurisdiction", help="Filter by jurisdiction (e.g. MAHARASHTRA)")
    parser.add_argument("--dry-run", action="store_true", help="List docs without ingesting")
    args = parser.parse_args()

    asyncio.run(ingest_all(
        jurisdiction_filter=args.jurisdiction,
        dry_run=args.dry_run,
    ))