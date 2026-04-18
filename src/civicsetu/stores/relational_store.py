from __future__ import annotations

from uuid import UUID

import structlog
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    Boolean,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from civicsetu.config.settings import get_settings
from civicsetu.models.enums import ChunkStatus, Jurisdiction
from civicsetu.models.schemas import IngestedDocument, LegalChunk

log = structlog.get_logger(__name__)
settings = get_settings()


# ── ORM Base ───────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── ORM Models ─────────────────────────────────────────────────────────────────

class DocumentRow(Base):
    __tablename__ = "documents"

    doc_id          = Column(PG_UUID(as_uuid=True), primary_key=True)
    doc_name        = Column(Text, nullable=False)
    jurisdiction    = Column(String(50), nullable=False)
    doc_type        = Column(String(50), nullable=False)
    source_url      = Column(Text, nullable=False)
    effective_date  = Column(Date, nullable=True)
    gazette_number  = Column(String(100), nullable=True)
    total_chunks    = Column(Integer, default=0)
    ingested_at     = Column(DateTime(timezone=True), server_default=func.now())
    is_active       = Column(Boolean, default=True)


class LegalChunkRow(Base):
    __tablename__ = "legal_chunks"

    chunk_id            = Column(PG_UUID(as_uuid=True), primary_key=True)
    doc_id              = Column(PG_UUID(as_uuid=True), ForeignKey("documents.doc_id", ondelete="CASCADE"))
    jurisdiction        = Column(String(50), nullable=False)
    doc_type            = Column(String(50), nullable=False)
    doc_name            = Column(Text, nullable=False)
    section_id          = Column(Text, nullable=False)
    section_title       = Column(Text, nullable=False)
    section_hierarchy   = Column(ARRAY(Text), nullable=False, default=list)
    text                = Column(Text, nullable=False)
    effective_date      = Column(Date, nullable=True)
    superseded_by       = Column(PG_UUID(as_uuid=True), ForeignKey("legal_chunks.chunk_id"), nullable=True)
    status              = Column(String(20), default=ChunkStatus.ACTIVE.value)
    source_url          = Column(Text, nullable=False)
    page_number         = Column(Integer, nullable=False)
    embedding           = Column(Vector(settings.embedding_dimension), nullable=True)


# ── Engine & Session Factory ───────────────────────────────────────────────────

def _build_engine():
    return create_async_engine(
        settings.postgres_dsn,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,       # drops stale connections silently
        echo=False,
    )


_engine = _build_engine()
AsyncSessionLocal = async_sessionmaker(_engine, expire_on_commit=False)


# ── Store Interface ────────────────────────────────────────────────────────────

class RelationalStore:
    """Async interface to Postgres. All methods are stateless — pass session in."""

    # ── Document ops ──────────────────────────────────────────────────────────

    @staticmethod
    async def upsert_document(
        session: AsyncSession, doc: IngestedDocument
    ) -> None:
        stmt = text("""
            INSERT INTO documents
                (doc_id, doc_name, jurisdiction, doc_type, source_url,
                 effective_date, gazette_number, total_chunks, is_active)
            VALUES
                (:doc_id, :doc_name, :jurisdiction, :doc_type, :source_url,
                 :effective_date, :gazette_number, :total_chunks, :is_active)
            ON CONFLICT (doc_id) DO UPDATE SET
                total_chunks = EXCLUDED.total_chunks,
                is_active    = EXCLUDED.is_active
        """)
        await session.execute(stmt, {
            "doc_id":        str(doc.doc_id),
            "doc_name":      doc.doc_name,
            "jurisdiction":  doc.jurisdiction.value,
            "doc_type":      doc.doc_type.value,
            "source_url":    doc.source_url,
            "effective_date": doc.effective_date,
            "gazette_number": doc.gazette_number,
            "total_chunks":  doc.total_chunks,
            "is_active":     doc.is_active,
        })
        log.info("upserted_document", doc_name=doc.doc_name, doc_id=str(doc.doc_id))

    # ── Chunk ops ─────────────────────────────────────────────────────────────

    @staticmethod
    async def insert_chunk(session: AsyncSession, chunk: LegalChunk) -> None:
        if chunk.embedding is None:
            raise ValueError(f"chunk {chunk.chunk_id} has no embedding — embed before inserting")

        stmt = text("""
            INSERT INTO legal_chunks
                (chunk_id, doc_id, jurisdiction, doc_type, doc_name, section_id,
                 section_title, section_hierarchy, text, effective_date,
                 superseded_by, status, source_url, page_number, embedding)
            VALUES
                (:chunk_id, :doc_id, :jurisdiction, :doc_type, :doc_name, :section_id,
                 :section_title, :section_hierarchy, :text, :effective_date,
                 :superseded_by, :status, :source_url, :page_number, :embedding)
            ON CONFLICT (chunk_id) DO NOTHING
        """)
        await session.execute(stmt, {
            "chunk_id":          str(chunk.chunk_id),
            "doc_id":            str(chunk.doc_id),
            "jurisdiction":      chunk.jurisdiction.value,
            "doc_type":          chunk.doc_type.value,
            "doc_name":          chunk.doc_name,
            "section_id":        chunk.section_id,
            "section_title":     chunk.section_title,
            "section_hierarchy": chunk.section_hierarchy,
            "text":              chunk.text,
            "effective_date":    chunk.effective_date,
            "superseded_by":     str(chunk.superseded_by) if chunk.superseded_by else None,
            "status":            chunk.status.value,
            "source_url":        chunk.source_url,
            "page_number":       chunk.page_number,
            "embedding":         str(chunk.embedding),
        })

    @staticmethod
    async def delete_chunks_by_doc(session: AsyncSession, doc_id: UUID) -> int:
        result = await session.execute(
            text("DELETE FROM legal_chunks WHERE doc_id = :doc_id"),
            {"doc_id": str(doc_id)},
        )
        deleted = result.rowcount
        log.info("deleted_existing_chunks", doc_id=str(doc_id), count=deleted)
        return deleted

    @staticmethod
    async def bulk_insert_chunks(
        session: AsyncSession, chunks: list[LegalChunk]
    ) -> int:
        """Insert multiple chunks in a single transaction. Returns count inserted."""
        inserted = 0
        for chunk in chunks:
            await RelationalStore.insert_chunk(session, chunk)
            inserted += 1
        log.info("bulk_inserted_chunks", count=inserted)
        return inserted

    @staticmethod
    async def get_chunk_by_id(
        session: AsyncSession, chunk_id: UUID
    ) -> LegalChunkRow | None:
        result = await session.get(LegalChunkRow, chunk_id)
        return result

    @staticmethod
    async def get_chunks_by_section(
        session: AsyncSession,
        section_id: str,
        jurisdiction: Jurisdiction | None = None,
    ) -> list[LegalChunkRow]:
        filters = "WHERE section_id = :section_id AND status = 'active'"
        params: dict = {"section_id": section_id}
        if jurisdiction:
            filters += " AND jurisdiction = :jurisdiction"
            params["jurisdiction"] = jurisdiction.value

        result = await session.execute(
            text(f"SELECT * FROM legal_chunks {filters}"), params
        )
        return result.fetchall()

    # ── Health ────────────────────────────────────────────────────────────────

    @staticmethod
    async def ping(session: AsyncSession) -> bool:
        result = await session.execute(text("SELECT 1"))
        return result.scalar() == 1
