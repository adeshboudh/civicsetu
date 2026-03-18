# ADR 004 — Multi-format Legal Document Chunker

**Date:** March 2026
**Status:** Accepted

---

## Context

Phase 2 required ingesting Maharashtra Real Estate Rules 2017 alongside the RERA Act
2016. Both are Indian legal PDFs but use structurally different numbering formats:

**Act format (RERA Act 2016):**


18. Return of amount and compensation.—
(1) If the promoter fails to complete...

Section title and em-dash are on the same line as the section number.

**Rule format (Maharashtra Rules 2017):**


3. 

Information to be furnished by the promoter...


Section number is on its own line, followed by a blank line, then the title.

The existing Act-format regex (`^\s*\d+[A-Z]?\.\s+[A-Z][^—\n]{3,80}\.?—`) produces
zero section boundaries on MahaRERA, triggering fallback paragraph chunking.
Paragraph chunking on MahaRERA produces 80+ chunks with no section_id metadata —
breaking citation accuracy entirely.

## Decision

Extend `LegalChunker` with a second boundary pattern for Rule format, applied as
a sequential fallback:

```python
PATTERNS = [
    ```
    ("act",  r'^\s*(?P<id>\d+[A-Z]?)\.?\s*(?P<title>[A-Z][^\n—]{3,80})\.?—'),
    ```
    ```
    ("rule", r'\n(?P<id>\d+)\.\s*\n(?P<title>[A-Z][^\n]{3,80})\n'),
    ```
]

for name, pattern in PATTERNS:
    matches = list(re.finditer(pattern, text, re.MULTILINE))
    if len(matches) >= MIN_SECTIONS:
        log.info("chunker_pattern_selected", pattern=name, sections=len(matches))
        break
```

`MIN_SECTIONS = 5` — fewer than 5 matches is treated as noise, not real boundaries.

The chunker logs which pattern was selected per document. Paragraph fallback is only
reached if both patterns fail.

## Consequences

**Positive:**

- MahaRERA produces 214 meaningful chunks with proper section_id metadata (44 sections)
- Citation accuracy preserved — every chunk maps to an identifiable Rule number
- Pattern selection is logged — observable, not silent
- Adding a third pattern (e.g. circular format) requires one array entry

**Negative:**

- Pattern priority is implicit — if a document accidentally matches Rule pattern first
with >= 5 hits, it bypasses Act pattern (mitigated by trying Act first)
- Regex fragility: PDFs with unusual whitespace will still hit fallback


## Alternatives Rejected

- **Hardcode document type in ingestion config:** Requires caller to know format ahead
of time; breaks the "any PDF URL" contract of the ingestion pipeline
- **ML-based section detector:** Overkill for deterministic numbered formats; adds
model dependency with no recall benefit on well-formatted government PDFs
- **Single universal regex:** No single pattern can match both `18. Title.—` and
`\n18.\n\nTitle\n` without catastrophic false positives