from __future__ import annotations

from datetime import date, datetime
from typing import Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, computed_field

from civicsetu.models.enums import (
		ChunkStatus,
		ConfidenceLevel,
		DocType,
		Jurisdiction,
		QueryType,
)

# ── Ingestion Schemas ──────────────────────────────────────────────────────────

class LegalChunk(BaseModel):
		"""Canonical unit produced by the ingestion pipeline.
		Every downstream store (pgvector, Neo4j, Postgres) is populated from this."""

		chunk_id: UUID = Field(default_factory=uuid4)
		doc_id: UUID
		jurisdiction: Jurisdiction
		doc_type: DocType
		doc_name: str
		section_id: str                          # e.g. "Section 18", "Rule 12"
		section_title: str
		section_hierarchy: list[str]             # e.g. ["Part IV", "Chapter 2", "Section 18"]
		text: str
		effective_date: date | None = None
		superseded_by: UUID | None = None
		status: ChunkStatus = ChunkStatus.ACTIVE
		source_url: str
		page_number: int
		embedding: list[float] | None = None  # populated after embedding step

		@computed_field
		@property
		def citation_label(self) -> str:
				"""Human-readable citation string."""
				return f"{self.doc_name} — {self.section_id}"


class IngestedDocument(BaseModel):
		"""Metadata record for a fully ingested document."""

		doc_id: UUID = Field(default_factory=uuid4)
		doc_name: str
		jurisdiction: Jurisdiction
		doc_type: DocType
		source_url: str
		effective_date: date | None = None
		gazette_number: str | None = None
		total_chunks: int = 0
		ingested_at: datetime = Field(default_factory=datetime.utcnow)
		is_active: bool = True


# ── Retrieval Schemas ──────────────────────────────────────────────────────────

class RetrievedChunk(BaseModel):
		"""A LegalChunk enriched with retrieval scores."""

		chunk: LegalChunk
		vector_score: float | None = None     # cosine similarity from pgvector
		rerank_score: float | None = None     # cross-encoder score from FlashRank
		retrieval_source: str = "vector"         # "vector" | "graph" | "metadata"
		graph_path: Optional[str] = None
		is_pinned: bool = False


# ── Response Schemas ───────────────────────────────────────────────────────────

class Citation(BaseModel):
		"""Traceable citation attached to every answer — non-negotiable."""

		section_id: str
		doc_name: str
		jurisdiction: Jurisdiction
		effective_date: date | None
		source_url: str
		chunk_id: UUID


class ChatMessage(BaseModel):
		"""A single conversational message."""

		role: Literal["user", "assistant"]
		content: str


class CivicSetuResponse(BaseModel):
		"""The immutable public response contract. Shape never changes between phases."""

		answer: str
		citations: list[Citation] = Field(min_length=1)
		confidence_score: float = Field(ge=0.0, le=1.0)
		query_type_resolved: QueryType
		conflict_warnings: list[str] = Field(default_factory=list)
		amendment_notice: str | None = None
		session_id: str | None = None
		disclaimer: str = (
				"This is AI-generated information, not legal advice. "
				"Consult a qualified lawyer for your specific situation."
		)

		@computed_field
		@property
		def confidence_level(self) -> ConfidenceLevel:
				if self.confidence_score >= 0.75:
						return ConfidenceLevel.HIGH
				elif self.confidence_score >= 0.50:
						return ConfidenceLevel.MEDIUM
				return ConfidenceLevel.LOW


class InsufficientInfoResponse(BaseModel):
		"""Returned when no citation can be found. Prevents hallucinated answers."""

		answer: str = "Insufficient information found in indexed documents."
		searched_query: str
		session_id: str | None = None
		disclaimer: str = (
				"This is AI-generated information, not legal advice. "
				"Consult a qualified lawyer for your specific situation."
		)


# ── API Request Schemas ────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
		query: str = Field(min_length=5, max_length=1000)
		session_id: str | None = None
		jurisdiction_filter: Jurisdiction | None = None
		top_k: int = Field(default=5, ge=1, le=20)


class IngestRequest(BaseModel):
		source_url: str
		doc_name: str
		jurisdiction: Jurisdiction
		doc_type: DocType
		effective_date: date | None = None
		gazette_number: str | None = None

# ── Graph API Schemas ──────────────────────────────────────────────────────────

class GraphNode(BaseModel):
		chunk_id: str
		section_id: str
		title: str
		jurisdiction: str
		doc_name: str
		is_active: bool
		connection_count: int


class GraphEdge(BaseModel):
		source: str   # chunk_id
		target: str   # chunk_id
		edge_type: str  # "REFERENCES" | "DERIVED_FROM"


class GraphTopologyResponse(BaseModel):
		nodes: list[GraphNode]
		edges: list[GraphEdge]
		stats: dict


class SectionChunkOut(BaseModel):
		chunk_id: str
		text: str
		page_number: int


class ConnectedSectionOut(BaseModel):
		section_id: str
		title: str
		jurisdiction: str
		edge_type: str  # REFERENCES_OUT | REFERENCES_IN | DERIVED_FROM_OUT | DERIVED_FROM_IN


class SectionContentResponse(BaseModel):
		section_id: str
		title: str
		doc_name: str
		jurisdiction: str
		effective_date: str | None
		source_url: str
		chunks: list[SectionChunkOut]
		connected_sections: list[ConnectedSectionOut]


class SectionContextQueryRequest(BaseModel):
		query: str = Field(min_length=5, max_length=1000)
		section_id: str
		jurisdiction: str
		session_id: str | None = None