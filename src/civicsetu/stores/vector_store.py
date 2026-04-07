from __future__ import annotations

from uuid import UUID

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from civicsetu.config.settings import get_settings
from civicsetu.models.enums import DocType, Jurisdiction
from civicsetu.models.schemas import LegalChunk, RetrievedChunk
from civicsetu.retrieval.cache import make_key, retrieval_cache

log = structlog.get_logger(__name__)
settings = get_settings()


class VectorStore:
    """Semantic similarity search over legal_chunks via pgvector HNSW index."""

    @staticmethod
    async def similarity_search(
        session: AsyncSession,
        query_embedding: list[float],
        top_k: int = 5,
        jurisdiction: Jurisdiction | None = None,
        doc_type: DocType | None = None,
        active_only: bool = True,
    ) -> list[RetrievedChunk]:
        """
        Returns top_k chunks ranked by cosine similarity to query_embedding.
        Filters by jurisdiction, doc_type, and status when provided.
        """
        if len(query_embedding) != settings.embedding_dimension:
            raise ValueError(
                f"Expected embedding of dim {settings.embedding_dimension}, "
                f"got {len(query_embedding)}"
            )

        cache_key = make_key(str(query_embedding), jurisdiction, doc_type, top_k, active_only)
        cached = retrieval_cache.get(cache_key)
        if cached is not None:
            log.debug(
                "vector_retrieval_cache_hit",
                jurisdiction=str(jurisdiction) if jurisdiction else None,
                doc_type=str(doc_type) if doc_type else None,
                top_k=top_k,
            )
            return cached

        # Build dynamic WHERE clause
        filters = []
        params: dict = {
            "embedding": str(query_embedding),
            "top_k": top_k,
        }

        if active_only:
            filters.append("status = 'active'")
        if jurisdiction:
            filters.append("jurisdiction = :jurisdiction")
            params["jurisdiction"] = jurisdiction.value
        if doc_type:
            filters.append("doc_type = :doc_type")
            params["doc_type"] = doc_type.value

        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

        stmt = text(f"""
            SELECT
                chunk_id, doc_id, jurisdiction, doc_type, doc_name,
                section_id, section_title, section_hierarchy,
                text, effective_date, superseded_by, status,
                source_url, page_number,
                1 - (embedding <=> CAST(:embedding AS vector)) AS cosine_similarity
            FROM legal_chunks
            {where_clause}
            ORDER BY embedding <=> CAST(:embedding AS vector)
            LIMIT :top_k
        """)

        result = await session.execute(stmt, params)
        rows = result.fetchall()

        retrieved = []
        for row in rows:
            chunk = LegalChunk(
                chunk_id=row.chunk_id,
                doc_id=row.doc_id,
                jurisdiction=row.jurisdiction,
                doc_type=row.doc_type,
                doc_name=row.doc_name,
                section_id=row.section_id,
                section_title=row.section_title,
                section_hierarchy=list(row.section_hierarchy),
                text=row.text,
                effective_date=row.effective_date,
                superseded_by=row.superseded_by,
                status=row.status,
                source_url=row.source_url,
                page_number=row.page_number,
                embedding=None,  # don't return raw vectors to callers
            )
            retrieved.append(RetrievedChunk(
                chunk=chunk,
                vector_score=round(float(row.cosine_similarity), 4),
                retrieval_source="vector",
            ))

        log.info(
            "vector_search_complete",
            top_k=top_k,
            results=len(retrieved),
            jurisdiction=jurisdiction,
        )
        retrieval_cache[cache_key] = retrieved
        return retrieved

    @staticmethod
    async def get_by_section(
        session: AsyncSession,
        section_id: str,
        jurisdiction: Jurisdiction | None = None,
    ) -> list[RetrievedChunk]:
        """
        Exact section_id lookup — used by graph_retriever to hydrate
        graph nodes into full LegalChunk objects.
        """
        filters = ["section_id = :section_id", "status = 'active'"]
        params: dict = {"section_id": section_id}

        if jurisdiction:
            filters.append("jurisdiction = :jurisdiction")
            params["jurisdiction"] = jurisdiction.value

        stmt = text(f"""
            SELECT
                chunk_id, doc_id, jurisdiction, doc_type, doc_name,
                section_id, section_title, section_hierarchy,
                text, effective_date, superseded_by, status,
                source_url, page_number
            FROM legal_chunks
            WHERE {' AND '.join(filters)}
        """)

        result = await session.execute(stmt, params)
        rows = result.fetchall()

        return [
            RetrievedChunk(
                chunk=LegalChunk(
                    chunk_id=row.chunk_id,
                    doc_id=row.doc_id,
                    jurisdiction=row.jurisdiction,
                    doc_type=row.doc_type,
                    doc_name=row.doc_name,
                    section_id=row.section_id,
                    section_title=row.section_title,
                    section_hierarchy=list(row.section_hierarchy),
                    text=row.text,
                    effective_date=row.effective_date,
                    superseded_by=row.superseded_by,
                    status=row.status,
                    source_url=row.source_url,
                    page_number=row.page_number,
                    embedding=None,
                ),
                retrieval_source="vector",
            )
            for row in rows
        ]

    @staticmethod
    async def delete_by_doc_id(session: AsyncSession, doc_id: UUID) -> int:
        """Hard delete all chunks for a document. Used during re-ingestion."""
        result = await session.execute(
            text("DELETE FROM legal_chunks WHERE doc_id = :doc_id"),
            {"doc_id": str(doc_id)},
        )
        count = result.rowcount
        log.info("deleted_chunks", doc_id=str(doc_id), count=count)
        return count
    
    @staticmethod
    async def get_section_family(
        session: AsyncSession,
        section_id: str,
        jurisdiction: Jurisdiction,
    ) -> list[RetrievedChunk]:
        import re
        if re.search(r'\(', section_id):
            return []

        result = await session.execute(
            text("""
                SELECT
                    chunk_id, doc_id, jurisdiction, doc_type, doc_name,
                    section_id, section_title, section_hierarchy,
                    text, effective_date, superseded_by, status,
                    source_url, page_number
                FROM legal_chunks
                WHERE jurisdiction = :jur
                AND (section_id = :sid OR section_id LIKE :pattern)
                AND status = 'active'
                ORDER BY section_id
            """),
            {"jur": jurisdiction.value, "sid": section_id, "pattern": f"{section_id}(%"},
        )
        rows = result.fetchall()
        return [
            RetrievedChunk(
                chunk=LegalChunk(
                    chunk_id=row.chunk_id,
                    doc_id=row.doc_id,
                    jurisdiction=row.jurisdiction,
                    doc_type=row.doc_type,
                    doc_name=row.doc_name,
                    section_id=row.section_id,
                    section_title=row.section_title,
                    section_hierarchy=list(row.section_hierarchy),
                    text=row.text,
                    effective_date=row.effective_date,
                    superseded_by=row.superseded_by,
                    status=row.status,
                    source_url=row.source_url,
                    page_number=row.page_number,
                    embedding=None,  # never return raw vectors to callers
                ),
                retrieval_source="vector",
            )
            for row in rows
        ]

