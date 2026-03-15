# ADR 002 — Section-Boundary Chunking

**Date:** March 2026
**Status:** Accepted

---

## Context

Standard RAG chunking strategies (fixed token count, sliding window) destroy
legal citation accuracy. A chunk split mid-section produces:
- Incomplete legal text (missing the operative clause)
- Unciteable chunks (no clean section_id)
- Broken cross-reference edges in Neo4j

## Decision

Chunk Indian legal documents at section boundaries using document-type-specific
regex patterns. Each chunk maps to exactly one legal section with a clean `section_id`.

**Primary pattern for Indian Acts/Rules:**
```

```
^\s*(?P<id>\d+[A-Z]?)\.\s+(?P<title>[A-Za-z][^—\n]{3,80})\.?—
```

```

**Fallback:** Double-newline paragraph splitting (non-standard documents only).

## Consequences

**Positive:**
- Every chunk has a clean `section_id` → citations are always precise
- Enables Neo4j graph seeding — section nodes map 1:1 to chunks
- Query "What does Section 18 say?" retrieves the exact section

**Negative:**
- Regex patterns must be maintained per doc type
- Regex requires inspection of raw PDF text before writing (cannot be generic)
- Fallback chunking (Para-1, Para-2) produces weaker citations

## Validation

RERA Act 2016: 86 sections detected, 224 chunks produced (subsection splits).
Section 18 correctly extracted with title "Return of amount and compensation".