from __future__ import annotations

import pytest

from civicsetu.guardrails.input_guard import InputGuard
from civicsetu.guardrails.output_guard import OutputGuard, _CONFIDENCE_FLOOR
from civicsetu.models.enums import Jurisdiction, QueryType


# ── InputGuard ────────────────────────────────────────────────────────────────

class TestInputGuard:

    def test_aadhaar_rejected(self):
        result = InputGuard.check("My Aadhaar is 1234 5678 9012, what are my RERA rights?")
        assert result.is_safe is False
        assert "AADHAAR" in result.reason

    def test_pan_rejected(self):
        result = InputGuard.check("ABCDE1234F is my PAN, am I eligible under RERA?")
        assert result.is_safe is False
        assert "PAN" in result.reason

    def test_email_rejected(self):
        result = InputGuard.check("Contact me at test@example.com about RERA Section 18")
        assert result.is_safe is False
        assert "EMAIL" in result.reason

    def test_phone_rejected(self):
        result = InputGuard.check("Call me on 9876543210 about my RERA complaint")
        assert result.is_safe is False
        assert "PHONE_IN" in result.reason

    def test_off_topic_rejected(self):
        result = InputGuard.check("Who won IPL 2024?")
        assert result.is_safe is False
        assert "civic" in result.reason.lower() or "RERA" in result.reason

    def test_off_topic_with_civic_keyword_passes(self):
        # "cricket" + "RERA" — civic context present, benefit of doubt
        result = InputGuard.check("Does RERA apply to cricket stadium construction projects?")
        assert result.is_safe is True

    def test_valid_legal_query_passes(self):
        result = InputGuard.check("What are promoter obligations under Section 11 of RERA?")
        assert result.is_safe is True
        assert result.sanitized_query == "What are promoter obligations under Section 11 of RERA?"

    def test_sanitized_query_stripped(self):
        result = InputGuard.check("  What is Section 18?  ")
        assert result.is_safe is True
        assert result.sanitized_query == "What is Section 18?"


# ── OutputGuard ───────────────────────────────────────────────────────────────

def _make_citation(section_id="18"):
    import uuid
    from datetime import date
    from civicsetu.models.schemas import Citation
    return Citation(
        section_id=section_id,
        doc_name="RERA Act 2016",
        jurisdiction=Jurisdiction.CENTRAL,
        effective_date=date(2016, 5, 1),
        source_url="https://example.com/rera.pdf",
        chunk_id=uuid.uuid4(),
    )


def _make_result(**overrides):
    base = {
        "raw_response": "Under Section 18...",
        "citations": [_make_citation()],
        "confidence_score": 0.9,
        "query_type": QueryType.CROSS_REFERENCE,
        "conflict_warnings": [],
        "amendment_notice": None,
    }
    base.update(overrides)
    return base


class TestOutputGuard:

    def test_valid_result_returns_civic_setu_response(self):
        from civicsetu.models.schemas import CivicSetuResponse
        response = OutputGuard.process(_make_result(), original_query="test query")
        assert isinstance(response, CivicSetuResponse)

    def test_no_citations_returns_insufficient(self):
        from civicsetu.models.schemas import InsufficientInfoResponse
        result = _make_result(citations=[])
        response = OutputGuard.process(result, original_query="some query")
        assert isinstance(response, InsufficientInfoResponse)

    def test_low_confidence_returns_insufficient(self):
        from civicsetu.models.schemas import InsufficientInfoResponse
        result = _make_result(confidence_score=_CONFIDENCE_FLOOR - 0.01)
        response = OutputGuard.process(result, original_query="some query")
        assert isinstance(response, InsufficientInfoResponse)

    def test_confidence_at_floor_passes(self):
        from civicsetu.models.schemas import CivicSetuResponse
        result = _make_result(confidence_score=_CONFIDENCE_FLOOR)
        response = OutputGuard.process(result, original_query="some query")
        assert isinstance(response, CivicSetuResponse)

    def test_disclaimer_always_present(self):
        response = OutputGuard.process(_make_result(), original_query="test query")
        assert hasattr(response, "disclaimer")
        assert len(response.disclaimer) > 0

    def test_conflict_warnings_passed_through(self):
        from civicsetu.models.schemas import CivicSetuResponse
        result = _make_result(conflict_warnings=["Rule 3 conflicts with Section 4"])
        response = OutputGuard.process(result, original_query="conflict query")
        assert isinstance(response, CivicSetuResponse)
        assert len(response.conflict_warnings) == 1
