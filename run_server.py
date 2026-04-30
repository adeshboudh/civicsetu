import asyncio
import sys
import uvicorn

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    uvicorn.run("src.civicsetu.api.main:app", host="0.0.0.0", port=8000, reload=True)
