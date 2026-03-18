from __future__ import annotations

import re
from dataclasses import dataclass
from uuid import UUID, uuid4

import structlog

from civicsetu.ingestion.parser import ParsedDocument
from civicsetu.models.enums import DocType, Jurisdiction

log = structlog.get_logger(__name__)


@dataclass
class RawSection:
    """A legal section extracted by boundary detection — before schema conversion."""
    section_id: str          # "Section 18", "Rule 12", "Clause 4"
    section_title: str       # "Obligations of Promoter"
    hierarchy: list[str]     # ["Part IV", "Chapter 2", "Section 18"]
    text: str
    page_number: int         # page where section starts


class LegalChunker:
    """
    Section-boundary chunker for Indian legal documents.

    Strategy:
    1. Run regex patterns against full document text to find section boundaries
    2. Split text at each boundary — one chunk per section
    3. If a section exceeds MAX_CHARS, split on subsection markers
    4. Never split mid-sentence

    Pattern priority (first match wins per line):
      ACT/RULES  → "Section 18", "Section 18A"
      RULES      → "Rule 12", "Rule 12A"
      CIRCULARS  → "Clause 4", "Para 3"
      ORDERS     → numbered paragraphs "1.", "2."
    """

    MAX_CHARS = 1500      # hard cap per chunk — forces subsection split
    MIN_CHARS = 100       # discard chunks below this (headers, page numbers)

    # ── Section boundary patterns per doc type ────────────────────────────────

    PATTERNS = {
        DocType.ACT: [
            # Matches: "1. Short title.—" or " 2. Definitions.—" or "18A. Special provisions.—"
            re.compile(
                r'^\s*(?P<id>\d+[A-Z]?)\.\s+(?P<title>[A-Za-z][^—\n]{3,80})\.?—',
                re.MULTILINE,
            ),
        ],
        DocType.RULES: [
            # Format 1: "\n2. \nDefinition: -" — number, dot, space/newline, title on next line
            # Matches MahaRERA Rules 2017 actual PDF format
            re.compile(
                '\n(?P<id>\\d+[A-Z]?)\\.\\s*\n\\s*(?P<title>[A-Za-z][^\\n]{3,80})',
            ),
            # Format 2: "3. Application for registration.—" same-line dash format
            re.compile(
                '^\\s*(?P<id>\\d+[A-Z]?)\\.\\s+(?P<title>[A-Za-z][^—\\n]{3,80})\\.?—',
                re.MULTILINE,
            ),
            # Format 3: "Rule 3 - Application" explicit Rule prefix
            re.compile(
                '^Rule\\s+(?P<id>\\d+[A-Z]?)\\s*[.\\-\u2013]\\s*(?P<title>[A-Z][^\\n]{3,80})',
                re.MULTILINE,
            ),
        ],
        DocType.CIRCULAR: [
            re.compile(
                r'^(?P<id>(?:Clause|Para(?:graph)?|Condition)\s+\d+[A-Z]?)\s*[\.\-–]\s*(?P<title>[A-Z][^\n]{3,80})',
                re.MULTILINE | re.IGNORECASE,
            ),
            re.compile(
                r'^\s*(?P<id>\d+\.(?:\d+)?)\s+(?P<title>[A-Z][^\n]{3,60})',
                re.MULTILINE,
            ),
        ],
        DocType.ORDER: [
            re.compile(
                r'^\s*(?P<id>\d+)\.\s+(?P<title>[A-Z][^\n]{3,80})',
                re.MULTILINE,
            ),
        ],
        DocType.NOTIFICATION: [
            re.compile(
                r'^\s*(?P<id>\d+[A-Z]?)\.\s+(?P<title>[A-Za-z][^—\n]{3,80})\.?—',
                re.MULTILINE,
            ),
        ],
    }

    # Fallback for unknown types
    FALLBACK_PATTERN = re.compile(
        r'^(?P<id>(?:Section|Rule|Clause|Article|Para)\s+\d+[A-Z]?)\s*[\.\-–]?\s*(?P<title>[A-Z][^\n]{0,80})',
        re.MULTILINE | re.IGNORECASE,
    )

    @classmethod
    def chunk(
        cls,
        parsed_doc: ParsedDocument,
        doc_type: DocType,
        doc_id: UUID,
        doc_name: str,
        jurisdiction: Jurisdiction,
        source_url: str,
        effective_date=None,
    ) -> list[dict]:
        """
        Main entry point. Returns list of dicts ready for LegalChunk construction.
        Separated from schema construction so this method stays testable in isolation.
        """
        full_text = parsed_doc.full_text
        patterns = cls.PATTERNS.get(doc_type, [cls.FALLBACK_PATTERN])

        # Collect all boundary matches across all patterns
        matches = []
        for pattern in patterns:
            for m in pattern.finditer(full_text):
                matches.append((m.start(), m.group("id").strip(), m.group("title").strip()))

        # Sort by position in document
        matches.sort(key=lambda x: x[0])

        if not matches:
            log.warning(
                "no_section_boundaries_found",
                doc_name=doc_name,
                doc_type=doc_type.value,
                action="falling back to paragraph chunking",
            )
            return cls._fallback_paragraph_chunks(
                full_text, doc_id, doc_name, jurisdiction,
                doc_type, source_url, effective_date
            )

        # Build sections from boundary positions
        sections = []
        for i, (start, section_id, title) in enumerate(matches):
            end = matches[i + 1][0] if i + 1 < len(matches) else len(full_text)
            text = full_text[start:end].strip()

            # Find approximate page number
            chars_before = start
            page_num = cls._estimate_page(parsed_doc, chars_before)

            if len(text) >= cls.MIN_CHARS:
                sections.append(RawSection(
                    section_id=section_id,
                    section_title=title,
                    hierarchy=cls._build_hierarchy(section_id, title, doc_name),
                    text=text,
                    page_number=page_num,
                ))

        # Split oversized sections at subsection boundaries
        final_sections = []
        for section in sections:
            if len(section.text) > cls.MAX_CHARS:
                final_sections.extend(cls._split_large_section(section))
            else:
                final_sections.append(section)

        log.info(
            "chunking_complete",
            doc_name=doc_name,
            sections_found=len(matches),
            chunks_produced=len(final_sections),
        )

        # Convert to dicts for LegalChunk construction
        return [
            {
                "chunk_id": uuid4(),
                "doc_id": doc_id,
                "jurisdiction": jurisdiction,
                "doc_type": doc_type,
                "doc_name": doc_name,
                "section_id": s.section_id,
                "section_title": s.section_title,
                "section_hierarchy": s.hierarchy,
                "text": s.text,
                "effective_date": effective_date,
                "source_url": source_url,
                "page_number": s.page_number,
            }
            for s in final_sections
        ]

    @staticmethod
    def _build_hierarchy(section_id: str, title: str, doc_name: str) -> list[str]:
        """
        Constructs section hierarchy path.
        Simplified for Phase 0 — Phase 1 will extract Part/Chapter from graph.
        """
        return [doc_name, section_id]

    @staticmethod
    def _estimate_page(parsed_doc: ParsedDocument, char_offset: int) -> int:
        """Estimate page number from character offset in full_text."""
        cumulative = 0
        for page in parsed_doc.pages:
            cumulative += len(page.text) + 2  # +2 for '\n\n' separator
            if cumulative >= char_offset:
                return page.page_number
        return parsed_doc.total_pages

    @classmethod
    def _split_large_section(cls, section: RawSection) -> list[RawSection]:
        """
        Splits a section that exceeds MAX_CHARS.
        Splits on subsection markers like (1), (2), (a), (b) first.
        Falls back to sentence boundary split if no markers found.
        """
        sub_pattern = re.compile('\\n\\s*\\((?:\\d+|[a-z]{1,3})\\)\\s+')
        parts = sub_pattern.split(section.text)

        if len(parts) <= 1:
            # No subsection markers — split on sentence boundary near MAX_CHARS
            text = section.text
            chunks = []
            while len(text) > cls.MAX_CHARS:
                split_at = text.rfind('. ', 0, cls.MAX_CHARS)
                if split_at == -1:
                    split_at = cls.MAX_CHARS
                chunks.append(text[:split_at + 1].strip())
                text = text[split_at + 1:].strip()
            if text:
                chunks.append(text)
            parts = chunks

        result = []
        for i, part in enumerate(parts):
            part = part.strip()
            if len(part) >= cls.MIN_CHARS:
                result.append(RawSection(
                    section_id=f"{section.section_id}({i + 1})" if i > 0 else section.section_id,
                    section_title=section.section_title,
                    hierarchy=section.hierarchy,
                    text=part,
                    page_number=section.page_number,
                ))
        return result

    @classmethod
    def _fallback_paragraph_chunks(
        cls, full_text, doc_id, doc_name, jurisdiction,
        doc_type, source_url, effective_date
    ) -> list[dict]:
        """
        Last-resort chunker when no section patterns match.
        Splits on double newlines — preserves paragraph integrity.
        Only used for circulars or orders with non-standard formatting.
        """
        paragraphs = [p.strip() for p in full_text.split('\n\n') if len(p.strip()) >= cls.MIN_CHARS]
        log.warning("fallback_paragraph_chunking", count=len(paragraphs))

        return [
            {
                "chunk_id": uuid4(),
                "doc_id": doc_id,
                "jurisdiction": jurisdiction,
                "doc_type": doc_type,
                "doc_name": doc_name,
                "section_id": f"Para-{i + 1}",
                "section_title": f"Paragraph {i + 1}",
                "section_hierarchy": [doc_name, f"Para-{i + 1}"],
                "text": para,
                "effective_date": effective_date,
                "source_url": source_url,
                "page_number": 1,
            }
            for i, para in enumerate(paragraphs)
        ]
