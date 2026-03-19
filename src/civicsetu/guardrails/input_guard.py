from __future__ import annotations

import re

import structlog

log = structlog.get_logger(__name__)

# ── PII patterns ──────────────────────────────────────────────────────────────
_PII_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("AADHAAR",  re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")),
    ("PHONE_IN", re.compile(r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b")),
    ("EMAIL",    re.compile(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b")),
    ("PAN",      re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")),
]

# ── Off-topic patterns ────────────────────────────────────────────────────────
_OFF_TOPIC_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(cricket|ipl|bollywood|recipe|cook|movie|song|weather|stock|crypto|bitcoin)\b", re.IGNORECASE),
    re.compile(r"\b(write me a|generate code|write code|tell me a joke|poem|story)\b", re.IGNORECASE),
]

_CIVIC_KEYWORDS = re.compile(
    r"\b(rera|section|rule|act|promoter|allottee|registrar|authority|"
    r"real estate|flat|apartment|project|carpet area|penalty|refund|"
    r"MahaRERA|maharashtra|jurisdiction|legal|law|provision|clause)\b",
    re.IGNORECASE,
)

# ── Presidio — module-level singleton, None if unavailable ───────────────────
# Initialized once at import time. Catches ImportError (not installed) AND
# SystemExit/Exception (spaCy model download fails in pip-less envs).
_presidio_analyzer = None
try:
    from presidio_analyzer import AnalyzerEngine  # type: ignore
    _presidio_analyzer = AnalyzerEngine()
    log.info("presidio_initialized")
except BaseException as e:
    log.warning("presidio_unavailable", reason=str(e)[:120], fallback="regex_only")


class InputGuard:
    """
    Validates and sanitizes incoming queries before graph invocation.

    Checks (in order):
    1. Regex PII — Aadhaar, phone, email, PAN
    2. Presidio NER PII — if available (singleton, not re-initialized per request)
    3. Off-topic filter — non-civic queries rejected unless civic keywords present
    """

    __slots__ = ()

    @staticmethod
    def check(query: str) -> "GuardResult":
        # 1 — Regex PII
        for label, pattern in _PII_PATTERNS:
            if pattern.search(query):
                log.warning("input_guard_pii_detected", pii_type=label, query_len=len(query))
                return GuardResult(
                    is_safe=False,
                    reason=f"Query contains personal information ({label}). "
                           "Please rephrase without personal details.",
                    sanitized_query=query,
                )

        # 2 — Presidio NER (reuse singleton, skip entirely if None)
        if _presidio_analyzer is not None:
            try:
                results = _presidio_analyzer.analyze(text=query, language="en")
                high_confidence = [r for r in results if r.score >= 0.75]
                if high_confidence:
                    entity_types = list({r.entity_type for r in high_confidence})
                    log.warning("input_guard_presidio_pii", entities=entity_types)
                    return GuardResult(
                        is_safe=False,
                        reason="Query contains personal information. Please rephrase.",
                        sanitized_query=query,
                    )
            except Exception as e:
                log.warning("presidio_analyze_failed", error=str(e)[:80])

        # 3 — Off-topic filter
        for pattern in _OFF_TOPIC_PATTERNS:
            if pattern.search(query):
                if _CIVIC_KEYWORDS.search(query):
                    break  # civic context present — benefit of doubt
                log.warning("input_guard_off_topic", query_preview=query[:80])
                return GuardResult(
                    is_safe=False,
                    reason="Query does not appear to be about Indian civic or legal documents. "
                           "CivicSetu answers questions about RERA, MahaRERA, and related laws.",
                    sanitized_query=query,
                )

        return GuardResult(is_safe=True, reason=None, sanitized_query=query.strip())


class GuardResult:
    __slots__ = ("is_safe", "reason", "sanitized_query")

    def __init__(self, is_safe: bool, reason: str | None, sanitized_query: str):
        self.is_safe = is_safe
        self.reason = reason
        self.sanitized_query = sanitized_query
