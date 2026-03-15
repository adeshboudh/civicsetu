from __future__ import annotations

import re
from datetime import date, datetime

import structlog

from civicsetu.models.enums import Jurisdiction

log = structlog.get_logger(__name__)


# ── Date patterns found in Indian legal documents ──────────────────────────────

_DATE_PATTERNS = [
    # "25th March, 2016" / "1st May, 2017"
    re.compile(
        r'\b(\d{1,2})(?:st|nd|rd|th)\s+(January|February|March|April|May|June|'
        r'July|August|September|October|November|December),?\s+(\d{4})\b',
        re.IGNORECASE,
    ),
    # "March 25, 2016"
    re.compile(
        r'\b(January|February|March|April|May|June|July|August|September|'
        r'October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b',
        re.IGNORECASE,
    ),
    # ISO / numeric: "2016-05-01" or "01/05/2016"
    re.compile(r'\b(\d{4})-(\d{2})-(\d{2})\b'),
    re.compile(r'\b(\d{2})/(\d{2})/(\d{4})\b'),
]

_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

# ── Amendment signals found in circulars and notifications ────────────────────

_AMENDMENT_SIGNALS = re.compile(
    r'\b(amend(?:ed|ment|ing|s)?|substitut(?:ed|ing|es)?|replac(?:ed|ing|es)?|'
    r'supersed(?:ed|ing|es)?|insert(?:ed|ing|s)?|delet(?:ed|ing|es)?|'
    r'modif(?:ied|ying|ication|ies)?)\b',
    re.IGNORECASE,
)

_SUPERSEDES_PATTERN = re.compile(
    r'(?:supersed(?:es|ed|ing)|replac(?:es|ed|ing)|in\s+lieu\s+of)\s+'
    r'(?:circular|order|notification|rule)?\s*(?:no\.?\s*)?'
    r'([\w/\-\.\s]+?(?:\d{4}))',    # capture until 4-digit year
    re.IGNORECASE,
)

_REFERENCES_PATTERN = re.compile(
    r'(?:under|pursuant\s+to|referred\s+to\s+in|as\s+per|'
    r'in\s+accordance\s+with|subject\s+to)\s+'
    r'[Ss]ection\s+(\d+[A-Z]?)',
)

# ── Gazette / circular number patterns ────────────────────────────────────────

_GAZETTE_PATTERN = re.compile(
    r'(?:Gazette\s+(?:of\s+India|Notification)|'
    r'No\.?\s*)([A-Z0-9/\(\)\-]+(?:/\d{4})?)',
    re.IGNORECASE,
)

_CIRCULAR_NO_PATTERN = re.compile(
    r'(?:Circular|Order|Notification)\s+No\.?\s*:?\s*([A-Z0-9/\-]+)',
    re.IGNORECASE,
)


class MetadataExtractor:
    """
    Extracts structured metadata from raw legal text chunks.
    All methods are static and stateless — safe to call in parallel.

    Extracted fields feed:
      - LegalChunk.effective_date
      - Graph edges: REFERENCES, SUPERSEDES, AMENDED_BY (Phase 1)
      - Search filters: jurisdiction, doc_type, is_active
    """

    @staticmethod
    def extract_effective_date(text: str) -> date | None:
        """
        Extract the most likely effective/enactment date from legal text.
        Returns the earliest valid date found — acts are dated at enactment.
        """
        candidates = []

        for pattern in _DATE_PATTERNS:
            for m in pattern.finditer(text):
                parsed = MetadataExtractor._parse_date_match(m)
                if parsed and 1947 <= parsed.year <= datetime.now().year:
                    candidates.append(parsed)

        if not candidates:
            return None

        # Return earliest date — enactment date, not amendment dates
        return min(candidates)

    @staticmethod
    def extract_section_references(text: str) -> list[str]:
        """
        Extract all section IDs referenced within a chunk's text.
        e.g. "under Section 11" → ["11"]
        Used to build REFERENCES edges in Neo4j (Phase 1).
        """
        refs = _REFERENCES_PATTERN.findall(text)
        return list(set(refs))  # deduplicate

    @staticmethod
    def extract_amendment_signals(text: str) -> dict:
        """
        Detect whether this chunk contains amendment language.
        Returns structured signals for graph edge creation in Phase 1.
        """
        signals = _AMENDMENT_SIGNALS.findall(text)
        supersedes = _SUPERSEDES_PATTERN.findall(text)

        return {
            "has_amendment_language": len(signals) > 0,
            "amendment_verbs": list(set(s.lower() for s in signals)),
            "supersedes_refs": supersedes,          # circular/order numbers superseded
        }

    @staticmethod
    def extract_gazette_number(text: str) -> str | None:
        """Extract gazette or circular number from document header text."""
        m = _GAZETTE_PATTERN.search(text)
        if m:
            return m.group(1).strip()
        m = _CIRCULAR_NO_PATTERN.search(text)
        if m:
            return m.group(1).strip()
        return None

    @staticmethod
    def infer_jurisdiction(text: str, filename: str = "") -> Jurisdiction:
        """
        Infer jurisdiction from document text and filename.
        Explicit markers take priority over filename hints.
        """
        text_lower = text.lower()
        filename_lower = filename.lower()

        jurisdiction_markers = {
            Jurisdiction.MAHARASHTRA: [
                "maharashtra", "mahrera", "mahaonline", "mumbai", "pune",
                "state of maharashtra",
            ],
            Jurisdiction.KARNATAKA: ["karnataka", "bengaluru", "bangalore", "k-rera"],
            Jurisdiction.UTTAR_PRADESH: ["uttar pradesh", "up rera", "lucknow"],
            Jurisdiction.TAMIL_NADU: ["tamil nadu", "tnrera", "chennai"],
        }

        for jurisdiction, markers in jurisdiction_markers.items():
            if any(m in text_lower or m in filename_lower for m in markers):
                return jurisdiction

        # Default: Central if no state marker found
        return Jurisdiction.CENTRAL

    @staticmethod
    def enrich_chunks(chunks: list[dict], filename: str = "") -> list[dict]:
        """
        Run all extractors over a list of raw chunk dicts.
        Mutates in place and returns the same list.
        Called at end of ingestion pipeline before embedding.
        """
        for chunk in chunks:
            text = chunk.get("text", "")

            # Only extract date if not already set by caller
            if chunk.get("effective_date") is None:
                chunk["effective_date"] = MetadataExtractor.extract_effective_date(text)

            # Attach extracted metadata as extra fields for graph seeding
            chunk["_section_references"] = MetadataExtractor.extract_section_references(text)
            chunk["_amendment_signals"] = MetadataExtractor.extract_amendment_signals(text)

            # Infer jurisdiction if not explicitly set
            if chunk.get("jurisdiction") == Jurisdiction.CENTRAL:
                inferred = MetadataExtractor.infer_jurisdiction(text, filename)
                # Don't override explicit jurisdiction — only fill UNKNOWN
                if chunk["jurisdiction"].value == "UNKNOWN":
                    chunk["jurisdiction"] = inferred

        log.info("metadata_enriched", count=len(chunks))
        return chunks

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _parse_date_match(m: re.Match) -> date | None:
        try:
            groups = m.groups()

            if len(groups) == 3:
                g0, g1, g2 = groups[0], groups[1], groups[2]

                # Pattern: "25th March, 2016" → ('25', 'March', '2016')
                if g0.isdigit() and not g1.isdigit() and g2.isdigit():
                    day = int(g0)
                    month = _MONTH_MAP.get(g1.lower(), 0)
                    year = int(g2)
                    if month and 1 <= day <= 31 and 1947 <= year <= 2100:
                        return date(year, month, day)

                # Pattern: "March 25, 2016" → ('March', '25', '2016')
                elif not g0.isdigit() and g1.isdigit() and g2.isdigit():
                    month = _MONTH_MAP.get(g0.lower(), 0)
                    day = int(g1)
                    year = int(g2)
                    if month and 1 <= day <= 31 and 1947 <= year <= 2100:
                        return date(year, month, day)

                # Pattern: ISO "2016-05-01" → ('2016', '05', '01')
                elif all(g.isdigit() for g in [g0, g1, g2]):
                    if len(g0) == 4:
                        return date(int(g0), int(g1), int(g2))
                    # DD/MM/YYYY → ('01', '05', '2016')
                    elif len(g2) == 4:
                        return date(int(g2), int(g1), int(g0))

        except (ValueError, TypeError):
            pass
        return None

