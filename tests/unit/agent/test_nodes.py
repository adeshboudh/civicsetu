from __future__ import annotations

import json
import uuid
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import _make_rc, _base_state

# ── reranker_node ─────────────────────────────────────────────────────────────

def test_reranker_empty_chunks():
    from civicsetu.agent.nodes import reranker_node
    result = reranker_node(_base_state(retrieved_chunks=[], reranked_chunks=[]))
    assert result["reranked_chunks"] == []


def test_reranker_pinned_chunks_always_first():
    from civicsetu.agent.nodes import reranker_node

    pinned = _make_rc(section_id="18", is_pinned=True)
    unpinned_1 = _make_rc(section_id="3")
    unpinned_2 = _make_rc(section_id="7")

    mock_results = [{"id": 0, "score": 0.9}, {"id": 1, "score": 0.8}]

    with patch("flashrank.Ranker") as MockRanker:
        instance = MockRanker.return_value
        instance.rerank.return_value = mock_results
        state = _base_state(
            retrieved_chunks=[pinned, unpinned_1, unpinned_2],
            reranked_chunks=[],
            query="test query",
        )
        result = reranker_node(state)

    reranked = result["reranked_chunks"]
    assert reranked[0].is_pinned is True
    assert reranked[0].chunk.section_id == "18"


def test_reranker_max_two_pinned():
    from civicsetu.agent.nodes import reranker_node

    pinned_chunks = [_make_rc(section_id=str(i), is_pinned=True) for i in range(4)]
    mock_results = []

    with patch("flashrank.Ranker") as MockRanker:
        instance = MockRanker.return_value
        instance.rerank.return_value = mock_results
        state = _base_state(
            retrieved_chunks=pinned_chunks,
            reranked_chunks=[],
            query="test query",
        )
        result = reranker_node(state)

    pinned_in_result = [c for c in result["reranked_chunks"] if c.is_pinned]
    assert len(pinned_in_result) <= 2


def test_reranker_deduplicates_by_chunk_id():
    from civicsetu.agent.nodes import reranker_node

    chunk = _make_rc(section_id="18")
    duplicate = _make_rc(section_id="18")
    # Force same chunk_id
    duplicate.chunk.chunk_id = chunk.chunk.chunk_id

    mock_results = [{"id": 0, "score": 0.9}]
    with patch("flashrank.Ranker") as MockRanker:
        instance = MockRanker.return_value
        instance.rerank.return_value = mock_results
        state = _base_state(
            retrieved_chunks=[chunk, duplicate],
            reranked_chunks=[],
            query="test query",
        )
        result = reranker_node(state)

    all_ids = [str(c.chunk.chunk_id) for c in result["reranked_chunks"]]
    assert len(all_ids) == len(set(all_ids))


# ── generator_node — citation anchoring ───────────────────────────────────────

def _llm_response(**kwargs) -> str:
    payload = {
        "answer": "Under Section 18...",
        "confidence_score": 0.9,
        "cited_chunks": [1],
        "amendment_notice": None,
        "conflict_warnings": [],
    }
    payload.update(kwargs)
    return json.dumps(payload)


def test_generator_cites_only_referenced_chunks():
    from civicsetu.agent.nodes import generator_node

    chunks = [_make_rc(section_id=str(i)) for i in range(1, 6)]
    # LLM says it only used chunks [1, 3]
    llm_out = _llm_response(cited_chunks=[1, 3])

    with patch("civicsetu.agent.nodes._llm_call", return_value=llm_out):
        state = _base_state(
            query="What does Section 1 say?",
            reranked_chunks=chunks,
        )
        result = generator_node(state)

    assert len(result["citations"]) == 2
    cited_ids = {c.section_id for c in result["citations"]}
    assert cited_ids == {"1", "3"}


def test_generator_fallback_when_cited_chunks_empty():
    from civicsetu.agent.nodes import generator_node

    chunks = [_make_rc(section_id=str(i)) for i in range(1, 4)]
    llm_out = _llm_response(cited_chunks=[])

    with patch("civicsetu.agent.nodes._llm_call", return_value=llm_out):
        state = _base_state(reranked_chunks=chunks)
        result = generator_node(state)

    # Fallback: all 3 chunks cited
    assert len(result["citations"]) == 3


def test_generator_filters_invalid_indices():
    from civicsetu.agent.nodes import generator_node

    chunks = [_make_rc(section_id=str(i)) for i in range(1, 4)]
    # 0 is out of range (1-based), 99 is out of range, "x" is not an int
    llm_out = _llm_response(cited_chunks=[0, 99, "x", 2])

    with patch("civicsetu.agent.nodes._llm_call", return_value=llm_out):
        state = _base_state(reranked_chunks=chunks)
        result = generator_node(state)

    # Only index 2 is valid → 1 citation
    assert len(result["citations"]) == 1
    assert result["citations"][0].section_id == "2"


def test_generator_returns_empty_on_no_chunks():
    from civicsetu.agent.nodes import generator_node

    result = generator_node(_base_state(reranked_chunks=[]))
    assert result["citations"] == []
    assert result["confidence_score"] == 0.0


def test_generator_handles_malformed_llm_json():
    from civicsetu.agent.nodes import generator_node

    chunks = [_make_rc(section_id="18")]
    with patch("civicsetu.agent.nodes._llm_call", return_value="not json {{{{"):
        state = _base_state(reranked_chunks=chunks)
        result = generator_node(state)

    assert result["confidence_score"] == 0.0
    assert result["citations"] == []


def test_generator_deduplicates_citations():
    from civicsetu.agent.nodes import generator_node

    # Two chunks with same (section_id, doc_name) — different chunk_ids
    chunk_a = _make_rc(section_id="18", doc_name="RERA Act 2016")
    chunk_b = _make_rc(section_id="18", doc_name="RERA Act 2016")
    llm_out = _llm_response(cited_chunks=[1, 2])

    with patch("civicsetu.agent.nodes._llm_call", return_value=llm_out):
        state = _base_state(reranked_chunks=[chunk_a, chunk_b])
        result = generator_node(state)

    assert len(result["citations"]) == 1


def test_generator_system_prompt_uses_plain_language_persona():
    from civicsetu.agent.nodes import generator_node
    from civicsetu.models.enums import QueryType

    captured = {}

    def fake_llm_call(prompt: str, system: str, temperature: float = 0.0) -> str:
        captured["prompt"] = prompt
        captured["system"] = system
        return _llm_response(answer="If a builder misses the deadline, the buyer gets a remedy.")

    with patch("civicsetu.agent.nodes._llm_call", side_effect=fake_llm_call):
        state = _base_state(
            query_type=QueryType.FACT_LOOKUP,
            reranked_chunks=[_make_rc(section_id="18")],
        )
        generator_node(state)

    assert "plain-language guide to Indian RERA laws" in captured["system"]
    assert "explain what the law means in practice" in captured["system"]
    assert "Respond with valid JSON only" in captured["system"]
    assert "plain-English summary" in captured["prompt"]
    assert "Do NOT open with \"According to Section X...\"" in captured["prompt"]


@pytest.mark.parametrize(
    ("query_type", "expected_hint"),
    [
        ("fact_lookup", "Give a direct answer and include one helpful analogy."),
        ("penalty_lookup", "Lead with the consequence"),
        ("cross_reference", "connection between sections as a narrative"),
        ("conflict_detection", "Explicitly flag the contradiction"),
        ("temporal", "Explain what changed, when, and why it matters."),
    ],
)
def test_generator_system_prompt_includes_query_type_tone_hint(query_type, expected_hint):
    from civicsetu.agent.nodes import generator_node

    captured = {}

    def fake_llm_call(prompt: str, system: str, temperature: float = 0.0) -> str:
        captured["system"] = system
        return _llm_response()

    with patch("civicsetu.agent.nodes._llm_call", side_effect=fake_llm_call):
        state = _base_state(
            query_type=query_type,
            reranked_chunks=[_make_rc(section_id="18")],
        )
        generator_node(state)

    assert expected_hint in captured["system"]
