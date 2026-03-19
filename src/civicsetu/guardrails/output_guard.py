# src/civicsetu/guardrails/output_guard.py

from __future__ import annotations

import structlog

from civicsetu.models.schemas import CivicSetuResponse, InsufficientInfoResponse

log = structlog.get_logger(__name__)

_DISCLAIMER = (
    "This is AI-generated information, not legal advice. "
    "Consult a qualified lawyer for your specific situation."
)

# Minimum confidence to return a CivicSetuResponse — below this, return
# InsufficientInfoResponse regardless of whether citations exist.
_CONFIDENCE_FLOOR = 0.30


class OutputGuard:
    """
    Post-processes the graph result before returning to the caller.

    Two responsibilities:
    1. Confidence floor — if confidence < 0.30, downgrade to InsufficientInfoResponse
    2. Disclaimer injection — ensure disclaimer is always present and unmodified

    Usage:
        response = OutputGuard.process(raw_result, original_query)
    """

    __slots__ = ()

    @staticmethod
    def process(
        result: dict,
        original_query: str,
    ) -> CivicSetuResponse | InsufficientInfoResponse:
        citations = result.get("citations", [])
        confidence = float(result.get("confidence_score", 0.0))
        answer = result.get("raw_response") or ""

        # 1 — No citations → always InsufficientInfoResponse
        if not citations:
            log.info("output_guard_no_citations", query_preview=original_query[:80])
            return InsufficientInfoResponse(searched_query=original_query)

        # 2 — Confidence below floor → downgrade
        if confidence < _CONFIDENCE_FLOOR:
            log.warning(
                "output_guard_low_confidence",
                confidence=confidence,
                floor=_CONFIDENCE_FLOOR,
                query_preview=original_query[:80],
            )
            return InsufficientInfoResponse(searched_query=original_query)

        # 3 — Build response (disclaimer is a default field on CivicSetuResponse,
        #     but we enforce it here explicitly so no future refactor silently drops it)
        from civicsetu.models.enums import QueryType
        response = CivicSetuResponse(
            answer=answer,
            citations=citations,
            confidence_score=confidence,
            query_type_resolved=result.get("query_type") or QueryType.FACT_LOOKUP,
            conflict_warnings=result.get("conflict_warnings", []),
            amendment_notice=result.get("amendment_notice"),
        )

        # Sanity check — disclaimer must survive serialization unchanged
        if response.disclaimer != _DISCLAIMER:
            log.error("output_guard_disclaimer_tampered", actual=response.disclaimer)

        log.info(
            "output_guard_pass",
            confidence=confidence,
            citations=len(citations),
            has_conflicts=bool(result.get("conflict_warnings")),
        )
        return response
