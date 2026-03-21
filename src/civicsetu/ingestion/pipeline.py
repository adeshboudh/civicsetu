from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path
from uuid import uuid4

import structlog

from civicsetu.config.settings import get_settings
from civicsetu.ingestion.chunker import LegalChunker
from civicsetu.ingestion.downloader import Downloader
from civicsetu.ingestion.embedder import Embedder
from civicsetu.ingestion.metadata_extractor import MetadataExtractor
from civicsetu.ingestion.parser import PDFParser
from civicsetu.models.enums import DocType, Jurisdiction
from civicsetu.models.schemas import IngestedDocument, LegalChunk
from civicsetu.stores.relational_store import AsyncSessionLocal, RelationalStore

log = structlog.get_logger(__name__)
settings = get_settings()


class IngestionPipeline:
    """
    Orchestrates the full document ingestion flow:
      Download → Parse → Chunk → Extract Metadata → Embed → Store

    One method: ingest_document()
    Everything else is private.
    """

    def __init__(self):
        self.embedder = Embedder()

    def ingest_document(
        self,
        source_url: str,
        doc_name: str,
        jurisdiction: Jurisdiction,
        doc_type: DocType,
        effective_date: date | None = None,
        dest_subdir: str = "",
        filename: str | None = None,
        force_redownload: bool = False,
        max_pages: int | None = None,
    ) -> IngestedDocument:
        """
        Full pipeline: URL → vectors in pgvector.
        Synchronous entry point — handles async store writes internally.
        Returns IngestedDocument with total_chunks count.
        """
        log.info("ingestion_started", doc_name=doc_name, url=source_url)

        # Step 1 — Download
        dest_dir = Path(settings.data_raw_dir) / dest_subdir
        local_path = Downloader.download(
            url=source_url,
            dest_dir=dest_dir,
            filename=filename,
            force=force_redownload,
        )

        # Step 2 — Parse
        parsed = PDFParser.parse(local_path, max_pages=max_pages)

        # Step 3 — Chunk
        doc_id = uuid4()
        raw_chunks = LegalChunker.chunk(
            parsed_doc=parsed,
            doc_type=doc_type,
            doc_id=doc_id,
            doc_name=doc_name,
            jurisdiction=jurisdiction,
            source_url=source_url,
            effective_date=effective_date,
        )

        # Step 4 — Enrich metadata
        raw_chunks = MetadataExtractor.enrich_chunks(raw_chunks, filename=local_path.name)

        # Step 5 — Embed
        texts = [c["text"] for c in raw_chunks]
        embeddings = self.embedder.embed_batch_documents(texts)
        for chunk_dict, embedding in zip(raw_chunks, embeddings):
            chunk_dict["embedding"] = embedding

        # Step 6 — Build LegalChunk objects (strips private _ keys)
        legal_chunks = [
            LegalChunk(**{k: v for k, v in c.items() if not k.startswith("_")})
            for c in raw_chunks
        ]

        # Step 7 — Persist to Postgres + pgvector
        doc = IngestedDocument(
            doc_id=doc_id,
            doc_name=doc_name,
            jurisdiction=jurisdiction,
            doc_type=doc_type,
            source_url=source_url,
            effective_date=effective_date,
            total_chunks=len(legal_chunks),
        )
        asyncio.run(self._persist(doc, legal_chunks))

        log.info(
            "ingestion_complete",
            doc_name=doc_name,
            chunks=len(legal_chunks),
            doc_id=str(doc_id),
        )
        return doc

    async def _persist(
        self, doc: IngestedDocument, chunks: list[LegalChunk]
    ) -> None:
        async with AsyncSessionLocal() as session:
            async with session.begin():
                await RelationalStore.upsert_document(session, doc)

        async with AsyncSessionLocal() as session:
            async with session.begin():
                await RelationalStore.bulk_insert_chunks(session, chunks)

