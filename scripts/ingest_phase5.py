# scripts/ingest_phase5.py
"""
Phase 5 ingestion — Karnataka RERA:
  1. Karnataka RERA Rules 2017

Run:
    uv run python scripts/ingest_phase5.py
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

# Karnataka Rule → RERA Act Section (hand-mapped)
# Karnataka Rules 2017 implements RERA Act 2016 under Section 84
_KA_DERIVED_FROM_MAP = {
    "3":  "4",   # Application for registration of project → promoter obligations
    "4":  "5",   # Grant / rejection of registration
    "5":  "6",   # Extension of registration
    "6":  "7",   # Revocation of registration
    "7":  "9",   # Registration of real estate agents
    "8":  "9",   # Grant / rejection of agent registration
    "9":  "10",  # Functions of real estate agents
    "10": "11",  # Agreement for Sale
    "11": "13",  # Responsibility of promoter post-possession
    "12": "17",  # Conveyance deed
    "13": "18",  # Timelines for refund
    "14": "19",  # Rate of interest
    "15": "25",  # Functions of regulatory authority
    "16": "31",  # Complaints to authority
    "17": "43",  # Appellate Tribunal composition
    "18": "58",  # Offences by companies
    "19": "66",  # Compounding of offences
}


def main() -> None:
    pipeline = IngestionPipeline()
    spec = DOCUMENT_REGISTRY["karnataka_rera_rules_2017"]

    log.info("phase5_ingest_start", doc=spec.name)
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
    log.info("phase5_ingest_complete",
             doc=spec.name, chunks=doc.total_chunks, doc_id=str(doc.doc_id))

    if doc.total_chunks == 0:
        print("\n✗ ABORT: 0 chunks — PDF is likely fully scanned.")
        print("  Fallback: use NAREDCO mirror and re-run.")
        print("  URL: https://naredco.in/notification/pdfs/Karnataka%20Real%20Estate%20(Regulation%20and%20Development)%20Rules,%202017.pdf")
        return

    stats = asyncio.run(GraphSeeder.seed_from_postgres(doc_id=str(doc.doc_id)))

    print("\n" + "=" * 60)
    print("Phase 5 — Karnataka RERA ingestion complete")
    print(f"  Chunks       : {doc.total_chunks}")
    print(f"  Graph stats  : {stats}")
    print("=" * 60)

    if doc.total_chunks < 10:
        print("\n⚠ WARNING: Very few chunks — check scanned_pages in log above")
    else:
        print("\n✓ Karnataka wired.")


if __name__ == "__main__":
    main()
