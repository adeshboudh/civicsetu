# scripts/seed_phase3.py
"""
Phase 3 graph re-seed.
Re-seeds Neo4j for all documents to:
- Fix cross-jurisdiction REFERENCES (MahaRERA → RERA Act sections)
- Add DERIVED_FROM edges (Document + Section level)
"""
from __future__ import annotations
import asyncio, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import structlog
from civicsetu.ingestion.graph_seeder import GraphSeeder
from civicsetu.stores.graph_store import GraphStore

log = structlog.get_logger(__name__)

async def main():
    log.info("phase3_seed_start")
    # Full re-seed: doc_id=None seeds all active documents
    stats = await GraphSeeder.seed_from_postgres(doc_id=None)
    log.info("phase3_seed_complete", stats=stats)
    print("\n" + "="*60)
    print("Phase 3 graph seed complete")
    print(f"  Docs          : {stats.get('docs')}")
    print(f"  Sections      : {stats.get('sections')}")
    print(f"  REFERENCES    : {stats.get('refs')}")
    print(f"  DERIVED_FROM  : {stats.get('derived_from')}")
    print("="*60)
    await GraphStore.close()

if __name__ == "__main__":
    asyncio.run(main())
