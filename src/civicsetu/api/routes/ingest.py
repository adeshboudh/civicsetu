from fastapi import APIRouter

router = APIRouter()

# Phase 0: ingestion is triggered via scripts/ingest_phase0.py
# This endpoint is scaffolded for Phase 1 admin API
@router.post("/ingest")
async def ingest():
    return {"status": "use scripts/ingest_phase0.py for now"}
