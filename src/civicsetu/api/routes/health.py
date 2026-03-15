from fastapi import APIRouter

from civicsetu.stores.relational_store import AsyncSessionLocal, RelationalStore

router = APIRouter()


@router.get("/health")
async def health():
    async with AsyncSessionLocal() as session:
        db_ok = await RelationalStore.ping(session)
    return {
        "status": "ok" if db_ok else "degraded",
        "db": "connected" if db_ok else "unreachable",
    }
