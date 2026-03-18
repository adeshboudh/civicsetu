# ADR 005 — Document Registry as Single Source of Truth

**Date:** March 2026
**Status:** Accepted

---

## Context

Phase 2 introduced a second document. With two documents, ingestion scripts started
duplicating URL strings, jurisdiction values, and doc_name strings across:

- `scripts/ingest_phase0.py`
- `scripts/ingest_phase2.py`
- Tests
- Any future migration or re-ingestion scripts

A URL change (e.g. NAREDCO moves their PDF) would require grep-and-replace across
multiple files with no compile-time safety.

## Decision

Introduce `src/civicsetu/config/document_registry.py` as the single authoritative
source for all document metadata:

```python
@dataclass(frozen=True)
class DocumentSpec:
    name: str
    url: str
    jurisdiction: Jurisdiction
    doc_type: DocType
    effective_date: date | None = None

DOCUMENT_REGISTRY: dict[str, DocumentSpec] = {
    "rera_act_2016": DocumentSpec(
        name="RERA Act 2016",
        url="https://...",
        jurisdiction=Jurisdiction.CENTRAL,
        doc_type=DocType.ACT,
        effective_date=date(2016, 5, 26),
    ),
    "mahrera_rules_2017": DocumentSpec(
        name="Maharashtra Real Estate (Regulation and Development) Rules 2017",
        url="https://naredco.in/...",
        jurisdiction=Jurisdiction.MAHARASHTRA,
        doc_type=DocType.RULES,
        effective_date=date(2017, 4, 21),
    ),
}
```

All ingestion scripts import from `document_registry`. No URL strings appear outside
this file.

## Consequences

**Positive:**

- URL change = one-line edit, guaranteed to propagate everywhere
- `DocumentSpec` is a frozen dataclass — immutable, hashable, diffable in git
- Phase 4 (multi-state expansion) is a registry append, not a script rewrite
- Tests can iterate `DOCUMENT_REGISTRY.values()` for fixture generation

**Negative:**

- Adding a document requires a code change + deploy (not a DB insert)
- Acceptable for Phase 0–3 volume (~10 documents); revisit for Phase 4+


## Alternatives Rejected

- **Database table for document registry:** Correct long-term, premature for current
volume. Adds a DB round-trip to every ingestion bootstrap.
- **Environment variables per document:** Unscalable beyond 2–3 documents;
no structure, no type safety
- **YAML/TOML config file:** Adds a parsing layer with no type safety; dataclass
achieves the same with Python's own type checker


