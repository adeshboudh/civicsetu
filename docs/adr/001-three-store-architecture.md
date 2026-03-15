# ADR 001 — Three-Store Architecture

**Date:** March 2026
**Status:** Accepted
**Deciders:** CivicSetu core team

---

## Context

Legal documents contain three distinct types of queryable information:
1. Semantic content — what a section *means*
2. Structural relationships — how sections *relate* to each other
3. Administrative metadata — when a section is *effective*, what *jurisdiction* it applies to

No single data store handles all three efficiently.

## Decision

Use three specialized stores in parallel:

| Store | Technology | Handles |
|---|---|---|
| Vector store | pgvector | Semantic similarity search |
| Graph store | Neo4j | Section relationships, amendments, cross-references |
| Relational store | PostgreSQL | Metadata, filtering, structured lookups |

## Consequences

**Positive:**
- Each store is optimally tuned for its query type
- Phase 0 ships with only pgvector — Neo4j added in Phase 1 without breaking anything
- Graph traversal enables queries no vector DB can answer

**Negative:**
- Three stores = three failure points
- Data consistency must be maintained across all three at ingestion time
- Higher operational complexity than single-store approaches

## Alternatives Rejected

- **Qdrant only:** Cannot do graph traversal or temporal queries
- **Neo4j only:** Poor semantic search without vector index
- **Elasticsearch:** Operational overhead not justified at Phase 0 scale