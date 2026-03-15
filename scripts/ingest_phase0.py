"""
Phase 0 ingestion runner.
Run with: make ingest  OR  uv run python scripts/ingest_phase0.py
"""
from datetime import date
from civicsetu.ingestion.pipeline import IngestionPipeline
from civicsetu.models.enums import DocType, Jurisdiction

PHASE_0_DOCUMENTS = [
    {
        "url": (
            "https://indiacode.nic.in/bitstream/123456789/15131/1/"
            "the_real_estate_(regulation_and_development)_act,_2016.pdf"
        ),
        "doc_name": "RERA Act 2016",
        "jurisdiction": Jurisdiction.CENTRAL,
        "doc_type": DocType.ACT,
        "effective_date": date(2016, 5, 1),
        "subdir": "acts",
        "filename": "rera_act_2016.pdf",
    },
    # Add MahaRERA Rules 2017 here when URL confirmed
    # Add MahaRERA Circulars here in Phase 1
]


def main():
    pipeline = IngestionPipeline()
    for doc in PHASE_0_DOCUMENTS:
        print(f"\n── Ingesting: {doc['doc_name']} ──")
        result = pipeline.ingest_document(
            source_url=doc["url"],
            doc_name=doc["doc_name"],
            jurisdiction=doc["jurisdiction"],
            doc_type=doc["doc_type"],
            effective_date=doc["effective_date"],
            dest_subdir=doc["subdir"],
            filename=doc["filename"],
        )
        print(f"   ✓ {result.total_chunks} chunks stored (doc_id: {result.doc_id})")

    print("\nPhase 0 ingestion complete.")


if __name__ == "__main__":
    main()
