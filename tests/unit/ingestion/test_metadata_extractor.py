from __future__ import annotations

import pytest

from civicsetu.ingestion.metadata_extractor import MetadataExtractor


@pytest.fixture
def extractor():
    return MetadataExtractor()


# ── Section reference extraction ──────────────────────────────────────────────

def test_extracts_section_references(extractor):
    text = "As per Section 18 and section 4A of the Act, the promoter shall..."
    refs = extractor.extract_section_references(text)
    assert "18" in refs
    assert "4A" in refs or "4a" in refs or "4A" in [r.upper() for r in refs]


def test_extracts_rule_references(extractor):
    text = "As required under Rule 3 and Rule 12A of these Rules..."
    refs = extractor.extract_rule_references(text)
    assert "3" in refs or "Rule 3" in refs


def test_cross_act_scrub_removes_ipc(extractor):
    text = "Section 420 of IPC and Section 18 of RERA apply here."
    refs = extractor.extract_section_references(text)
    # Section 420 is from IPC — should be scrubbed or 18 should appear
    # Either scrubbing works or only valid RERA sections (1-92) returned
    for ref in refs:
        num = int("".join(c for c in ref if c.isdigit()) or "0")
        assert num <= 92, f"Section {ref} exceeds RERA bounds — cross-act scrub failed"


def test_no_false_positives_from_large_numbers(extractor):
    text = "Section 196 of Companies Act and Section 11 of RERA apply."
    refs = extractor.extract_section_references(text)
    assert "196" not in refs
    assert "11" in refs


def test_empty_text_returns_empty(extractor):
    assert extractor.extract_section_references("") == []
