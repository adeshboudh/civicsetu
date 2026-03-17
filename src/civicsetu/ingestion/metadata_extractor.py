# src/civicsetu/ingestion/metadata_extractor.py
from __future__ import annotations

import re
from datetime import date, datetime

import structlog

from civicsetu.models.enums import Jurisdiction

log = structlog.get_logger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

# RERA Act 2016 has sections 1–92 only.
# Any section number above this is a cross-act reference (IPC, CPC, Companies Act, etc.)
_RERA_MAX_SECTION = 92


# ── Date patterns ─────────────────────────────────────────────────────────────

_DATE_PATTERNS = [
    # "25th March, 2016" / "1st May, 2017"
    re.compile(
        '\\b(\\d{1,2})(?:st|nd|rd|th)\\s+(January|February|March|April|May|June|'
        'July|August|September|October|November|December),?\\s+(\\d{4})\\b',
        re.IGNORECASE,
    ),
    # "March 25, 2016"
    re.compile(
        '\\b(January|February|March|April|May|June|July|August|September|'
        'October|November|December)\\s+(\\d{1,2}),?\\s+(\\d{4})\\b',
        re.IGNORECASE,
    ),
    # ISO: "2016-05-01"
    re.compile('\\b(\\d{4})-(\\d{2})-(\\d{2})\\b'),
    # DD/MM/YYYY: "01/05/2016"
    re.compile('\\b(\\d{2})/(\\d{2})/(\\d{4})\\b'),
]

_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


# ── Section reference patterns ────────────────────────────────────────────────

# Singular: matches "of section 46", "under section 18", "contravention of section 4"
# Negative lookahead (?!\s+of\s+the\b) blocks cross-act refs:
#   "section 2 of the Companies Act" → rejected
#   "section 9 of the Consumer Protection Act" → rejected
_REF_SINGULAR = re.compile(
    '(?:under|pursuant\\s+to|referred\\s+to\\s+in|as\\s+per|'
    'in\\s+accordance\\s+with|subject\\s+to|proviso\\s+to|'
    'contravention\\s+of|also|of|in|for)\\s+'
    '(?:sub-section\\s+\\(\\d+\\)\\s+of\\s+)?'
    '[Ss]ection\\s+(\\d+[A-Z]?)',
    re.IGNORECASE,
)

# Plural: matches "sections 12, 14, 18 and section 19"
# Negative lookahead blocks "sections 193, 219 of the Indian Penal Code"
_REF_PLURAL_SPAN = re.compile(
    '[Ss]ections?\\s+'
    '((?:\\d+[A-Z]?(?:\\s*,\\s*|\\s+and\\s+(?:[Ss]ection\\s+)?))+\\d+[A-Z]?)',
    re.IGNORECASE,
)

# Word-boundary number extractor — prevents "19" matching inside "196"
_NUM_IN_SPAN = re.compile('\\b(\\d+[A-Z]?)\\b')


# ── Amendment / gazette patterns ──────────────────────────────────────────────

_AMENDMENT_SIGNALS = re.compile(
    '\\b(amend(?:ed|ment|ing|s)?|substitut(?:ed|ing|es)?|replac(?:ed|ing|es)?|'
    'supersed(?:ed|ing|es)?|insert(?:ed|ing|s)?|delet(?:ed|ing|es)?|'
    'modif(?:ied|ying|ication|ies)?)\\b',
    re.IGNORECASE,
)

_SUPERSEDES_PATTERN = re.compile(
    '(?:supersed(?:es|ed|ing)|replac(?:es|ed|ing)|in\\s+lieu\\s+of)\\s+'
    '(?:circular|order|notification|rule)?\\s*(?:no\\.?\\s*)?'
    '([\\w/\\-\\.\\s]+?(?:\\d{4}))',
    re.IGNORECASE,
)

_GAZETTE_PATTERN = re.compile(
    '(?:Gazette\\s+(?:of\\s+India|Notification)|No\\.?\\s*)'
    '([A-Z0-9/\\(\\)\\-]+(?:/\\d{4})?)',
    re.IGNORECASE,
)

_CIRCULAR_NO_PATTERN = re.compile(
    '(?:Circular|Order|Notification)\\s+No\\.?\\s*:?\\s*([A-Z0-9/\\-]+)',
    re.IGNORECASE,
)

# Used to REMOVE spans before extraction
_CROSS_ACT_REF = re.compile(
    '[Ss]ections?\\s+[\\d,\\s]+(?:and\\s+(?:[Ss]ection\\s+)?[\\d]+)?'
    '\\s+(?:of|for\\s+the\\s+purposes\\s+of)\\s+(?:section\\s+\\d+\\s+of\\s+)?'
    'the\\s+[A-Z][\\w\\s]+?(?:Act|Code),?\\s*\\d*',
    re.IGNORECASE,
)

# Matches "section X of the Y Act/Code (year)" — full cross-act citation
_CROSS_ACT_SPAN = re.compile(
    '[Ss]ection\\s+(\\d+[A-Z]?)\\s+of\\s+the\\s+[A-Z][\\w\\s,]+?(?:Act|Code)'
    '(?:\\s*\\(\\d+\\s+of\\s+\\d+\\))?',
    re.IGNORECASE,
)

# Plural cross-act: "sections 193, 219 and 228 for the purposes of section 196 of the IPC"
_CROSS_ACT_SPAN_PLURAL = re.compile(
    '[Ss]ections?\\s+[\\d][\\d,\\s]+(?:and\\s+(?:[Ss]ection\\s+)?\\d+\\s+)?'
    'of\\s+the\\s+[A-Z][\\w\\s,]+?(?:Act|Code)'
    '(?:\\s*\\(\\d+\\s+of\\s+\\d+\\))?',
    re.IGNORECASE,
)

_KNOWN_CROSS_ACTS = re.compile(
    'Companies\\s+Act|Indian\\s+Penal\\s+Code|Code\\s+of\\s+Civil\\s+Procedure|'
    'Consumer\\s+Protection\\s+Act|Chartered\\s+Accountants\\s+Act|'
    'Income\\s+Tax\\s+Act|Constitution\\s+of\\s+India|Arbitration\\s+and\\s+Conciliation',
    re.IGNORECASE,
)

_REF_BARE = re.compile('[Ss]ection\\s+(\\d+[A-Z]?)', re.IGNORECASE)

# ── Extractor ─────────────────────────────────────────────────────────────────

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
    def extract_section_references(text: str) -> list[str]:
        """
        Extract all RERA section IDs referenced within a chunk's text.

        Handles:
          - Singular: "under section 18", "of section 46", "contravention of section 4"
          - Plural:   "sections 12, 14, 18 and section 19"
          - Sub-sec:  "under sub-section (2) of section 40"

        Excludes:
          - Cross-act refs: "section 2 of the Companies Act"
          - Section numbers > 92 (outside RERA range)
          - Substring matches: "19" inside "196"
        """
        sentences = re.split(r'(?<=[.;])\s+', text)
        clean_sentences = []
        for sent in sentences:
            if _KNOWN_CROSS_ACTS.search(sent):
                # Remove only the specific cross-act section spans
                sent = _CROSS_ACT_SPAN.sub('', sent)
                sent = _CROSS_ACT_SPAN_PLURAL.sub('', sent)
            clean_sentences.append(sent)
        clean = ' '.join(clean_sentences)

        refs: set[str] = set()

        for m in _REF_BARE.finditer(clean):
            sec = m.group(1)
            if _is_rera_section(sec):
                refs.add(sec)

        for m in _REF_SINGULAR.finditer(clean):
            sec = m.group(1)
            if _is_rera_section(sec):
                refs.add(sec)
        for m in _REF_PLURAL_SPAN.finditer(clean):
            for sec in _NUM_IN_SPAN.findall(m.group(1)):
                if _is_rera_section(sec):
                    refs.add(sec)

        return list(refs)

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
        return min(candidates) if candidates else None

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
            "supersedes_refs": supersedes,
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

            if chunk.get("effective_date") is None:
                chunk["effective_date"] = MetadataExtractor.extract_effective_date(text)

            chunk["_section_references"] = MetadataExtractor.extract_section_references(text)
            chunk["_amendment_signals"] = MetadataExtractor.extract_amendment_signals(text)

            if chunk.get("jurisdiction") == Jurisdiction.CENTRAL:
                inferred = MetadataExtractor.infer_jurisdiction(text, filename)
                if chunk["jurisdiction"].value == "UNKNOWN":
                    chunk["jurisdiction"] = inferred

        log.info("metadata_enriched", count=len(chunks))
        return chunks

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _parse_date_match(m: re.Match) -> date | None:
        try:
            g0, g1, g2 = m.group(1), m.group(2), m.group(3)

            # "25th March, 2016" → ('25', 'March', '2016')
            if g0.isdigit() and not g1.isdigit() and g2.isdigit():
                month = _MONTH_MAP.get(g1.lower(), 0)
                if month and 1 <= int(g0) <= 31:
                    return date(int(g2), month, int(g0))

            # "March 25, 2016" → ('March', '25', '2016')
            elif not g0.isdigit() and g1.isdigit() and g2.isdigit():
                month = _MONTH_MAP.get(g0.lower(), 0)
                if month and 1 <= int(g1) <= 31:
                    return date(int(g2), month, int(g1))

            # ISO "2016-05-01" → ('2016', '05', '01')
            elif all(x.isdigit() for x in [g0, g1, g2]):
                if len(g0) == 4:
                    return date(int(g0), int(g1), int(g2))
                elif len(g2) == 4:
                    return date(int(g2), int(g1), int(g0))

        except (ValueError, TypeError):
            pass
        return None


# ── Module-level helper ───────────────────────────────────────────────────────

def _is_rera_section(sec: str) -> bool:
    """Return True only if sec is a valid RERA section number (1–92)."""
    digits = re.sub('[A-Z]', '', sec, flags=re.IGNORECASE)
    return digits.isdigit() and 1 <= int(digits) <= _RERA_MAX_SECTION
