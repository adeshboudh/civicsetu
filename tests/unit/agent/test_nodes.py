from __future__ import annotations

import json
import uuid
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import _make_rc, _base_state

# ── settings defaults ─────────────────────────────────────────────────────────

def test_reranker_settings_defaults():
    """Settings ship with safe, sensible defaults — no .env needed."""
    from civicsetu.config.settings import Settings
    s = Settings()
    assert s.reranker_model == "rank-T5-flan"
    assert s.reranker_score_threshold == 0.05
    assert s.reranker_score_gap == 0.3


def test_reranker_settings_env_override(monkeypatch):
    """Env vars must override reranker defaults at runtime."""
    from civicsetu.config.settings import Settings, get_settings
    get_settings.cache_clear()  # clear lru_cache so new Settings() is fresh

    monkeypatch.setenv("RERANKER_MODEL", "ms-marco-electra-base")
    monkeypatch.setenv("RERANKER_SCORE_THRESHOLD", "0.5")
    monkeypatch.setenv("RERANKER_SCORE_GAP", "0.4")

    s = Settings()
    assert s.reranker_model == "ms-marco-electra-base"
    assert s.reranker_score_threshold == 0.5
    assert s.reranker_score_gap == 0.4

    get_settings.cache_clear()  # leave cache clean for other tests

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


def test_reranker_keeps_pinned_chunks_up_to_context_limit():
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
    assert len(pinned_in_result) == 4


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

def test_eval_pinned_section_refs_are_fetched_by_expected_jurisdiction():
    from civicsetu.agent.nodes import _pinned_section_specs
    from civicsetu.models.enums import Jurisdiction

    specs = _pinned_section_specs(
        ["Section 6", "Rule 7", "62"],
        Jurisdiction.KARNATAKA,
    )

    assert specs == [
        ("6", Jurisdiction.CENTRAL),
        ("7", Jurisdiction.KARNATAKA),
        ("62", Jurisdiction.KARNATAKA),
    ]


def test_pin_relevance_prefers_ground_truth_terms():
    from civicsetu.agent.nodes import _sort_pinned_family

    intro = _make_rc(section_id="7", is_pinned=True)
    intro.chunk.text = "7. Revocation of registration after satisfaction by the Authority."
    grounds = _make_rc(section_id="7(2)", is_pinned=True)
    grounds.chunk.text = "the promoter makes default in doing anything required under this Act"
    notice = _make_rc(section_id="7(9)", is_pinned=True)
    notice.chunk.text = "thirty days notice in writing stating the grounds for revocation"
    unfair = _make_rc(section_id="7(4)", is_pinned=True)
    unfair.chunk.text = "the promoter is involved in unfair practice or irregularities"

    ranked = _sort_pinned_family(
        [intro, grounds, notice, unfair],
        "promoter does not comply fraudulent practices thirty days notice grounds",
    )

    assert [c.chunk.section_id for c in ranked[:3]] == ["7(9)", "7(2)", "7(4)"]


def test_pin_relevance_prefers_specific_subclauses_over_base_header_on_tie():
    from civicsetu.agent.nodes import _sort_pinned_family

    header = _make_rc(section_id="7", is_pinned=True)
    header.chunk.text = "Revocation of registration."
    penalty = _make_rc(section_id="7(6)", is_pinned=True)
    penalty.chunk.text = "the promoter violates section 38 or fails to pay any penalty imposed"

    ranked = _sort_pinned_family(
        [header, penalty],
        "grounds for revocation penalties imposed",
    )

    assert [c.chunk.section_id for c in ranked] == ["7(6)", "7"]


def test_prepend_pinned_sections_promotes_existing_matches(monkeypatch):
    from civicsetu.agent.nodes import _prepend_pinned_sections
    from civicsetu.models.enums import Jurisdiction

    noisy = _make_rc(section_id="4(10)", jurisdiction=Jurisdiction.KARNATAKA)
    rule_5 = _make_rc(section_id="5", jurisdiction=Jurisdiction.KARNATAKA)
    section_4 = _make_rc(section_id="4(14)", jurisdiction=Jurisdiction.CENTRAL)

    def fake_run(coro):
        coro.close()
        return []

    monkeypatch.setattr("civicsetu.agent.nodes.asyncio.run", fake_run)

    result = _prepend_pinned_sections(
        {
            "pinned_section_refs": ["Rule 5", "Section 4"],
            "pinned_section_jurisdiction": Jurisdiction.KARNATAKA,
            "pinned_section_hint": "separate bank account seventy per cent",
        },
        [noisy, rule_5, section_4],
    )

    assert [c.chunk.section_id for c in result] == ["5", "4(14)", "4(10)"]
    assert result[0].is_pinned is True
    assert result[1].is_pinned is True


def test_reranker_node_trims_eval_context_to_pinned_families():
    from civicsetu.agent.nodes import reranker_node
    from civicsetu.models.enums import Jurisdiction

    matched_rule = _make_rc(section_id="5", jurisdiction=Jurisdiction.KARNATAKA, is_pinned=True)
    matched_section = _make_rc(section_id="4(14)", jurisdiction=Jurisdiction.CENTRAL, is_pinned=True)
    noisy_central = _make_rc(section_id="4(10)", jurisdiction=Jurisdiction.CENTRAL)
    noisy_state = _make_rc(section_id="3", jurisdiction=Jurisdiction.UTTAR_PRADESH)

    with patch(
        "civicsetu.retrieval.reranker.Reranker.rerank",
        return_value=[matched_rule, matched_section, noisy_central, noisy_state],
    ):
        result = reranker_node(
            _base_state(
                query="Which Karnataka rule implements project account maintenance?",
                rewritten_query="Which Karnataka rule implements project account maintenance?",
                retrieved_chunks=[matched_rule, matched_section, noisy_central, noisy_state],
                pinned_section_refs=["Rule 5", "Section 4"],
                pinned_section_jurisdiction=Jurisdiction.KARNATAKA,
            )
        )

    assert [c.chunk.section_id for c in result["reranked_chunks"]] == ["5", "4(14)"]


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
    # Salvage path: non-empty malformed text is returned as answer with 0.3 confidence
    assert result["confidence_score"] == 0.3
    assert len(result["citations"]) == 1  # salvage cites all provided chunks


def test_generator_salvages_plain_text_when_json_missing():
    from civicsetu.agent.nodes import generator_node

    chunks = [_make_rc(section_id="18"), _make_rc(section_id="31")]
    raw = "Promoter must register project, disclose details, and honor timelines."

    with patch("civicsetu.agent.nodes._llm_call", return_value=raw):
        result = generator_node(_base_state(reranked_chunks=chunks))

    assert result["raw_response"] == raw
    assert result["confidence_score"] == 0.3
    assert len(result["citations"]) == 2


def test_classifier_extracts_json_from_reasoning_wrapper():
    from civicsetu.agent.nodes import classifier_node
    from civicsetu.models.enums import QueryType

    wrapped = """
Thinking:
- classify legal query

```json
{"query_type":"cross_reference","rewritten_query":"Section 18 refund obligations under RERA Act"}
```
"""

    with patch("civicsetu.agent.nodes._llm_call", return_value=wrapped):
        result = classifier_node(_base_state(query="What does Section 18 say?"))

    assert result["query_type"] == QueryType.CROSS_REFERENCE
    assert result["rewritten_query"] == "Section 18 refund obligations under RERA Act"


def test_generator_extracts_json_from_reasoning_wrapper():
    from civicsetu.agent.nodes import generator_node

    chunks = [_make_rc(section_id="18")]
    wrapped = """
Here is structured answer.

```json
{"answer":"Refund due under Section 18.","confidence_score":0.8,"cited_chunks":[1],"amendment_notice":null,"conflict_warnings":[]}
```
"""

    with patch("civicsetu.agent.nodes._llm_call", return_value=wrapped):
        result = generator_node(_base_state(reranked_chunks=chunks))

    assert result["raw_response"] == "Refund due under Section 18."
    assert result["confidence_score"] == 0.8
    assert len(result["citations"]) == 1


def test_llm_call_uses_json_mode_for_osmapi(monkeypatch):
    import civicsetu.agent.nodes as nodes_mod

    monkeypatch.setenv("OPENAI_API_BASE", "https://api.osmapi.com/v1")

    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content='{"ok": true}'))]
    fake_response.usage = None

    with patch("civicsetu.agent.nodes.litellm.completion", return_value=fake_response) as completion:
        result = nodes_mod._llm_call("prompt", "system")

    assert result == '{"ok": true}'
    assert completion.call_args.kwargs["response_format"] == {"type": "json_object"}


def test_llm_call_does_not_force_no_reasoning_for_osmapi(monkeypatch):
    import civicsetu.agent.nodes as nodes_mod

    monkeypatch.setenv("OPENAI_API_BASE", "https://api.osmapi.com/v1")

    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content='{"ok": true}'))]
    fake_response.usage = None

    original_models = nodes_mod.THINKING_MODELS[:]
    nodes_mod.THINKING_MODELS[:] = ["openai/gpt-4o-mini"]

    with patch("civicsetu.agent.nodes.litellm.completion", return_value=fake_response) as completion:
        try:
            nodes_mod._llm_call("prompt", "system")
        finally:
            nodes_mod.THINKING_MODELS[:] = original_models

    assert "extra_body" not in completion.call_args.kwargs


def test_llm_call_does_not_attach_deepseek_kwargs_to_non_deepseek_models():
    import civicsetu.agent.nodes as nodes_mod

    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content='{"ok": true}'))]
    fake_response.usage = None

    original_models = nodes_mod.THINKING_MODELS[:]
    nodes_mod.THINKING_MODELS[:] = ["groq/llama-3.3-70b-versatile"]

    with patch("civicsetu.agent.nodes.litellm.completion", return_value=fake_response) as completion:
        try:
            nodes_mod._llm_call("prompt", "system", tier="thinking")
        finally:
            nodes_mod.THINKING_MODELS[:] = original_models

    assert completion.call_args.kwargs["model"] == "groq/llama-3.3-70b-versatile"
    assert "extra_body" not in completion.call_args.kwargs


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
        ("fact_lookup", "Open with 1-2 sentences directly answering the exact question asked"),
        ("penalty_lookup", "Lead with the consequence"),
        ("cross_reference", "Explain what the cited section says first"),
        ("conflict_detection", "Explicitly flag the contradiction"),
        ("temporal", "Lead with the specific time period or deadline"),
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


# ── _get_ranker uses settings model ──────────────────────────────────────────

def test_get_ranker_uses_settings_model():
    """_get_ranker() must pass settings.reranker_model to Ranker constructor."""
    import civicsetu.retrieval.reranker as reranker_mod
    from unittest.mock import patch, MagicMock
    reranker_mod._ranker = None  # clear module-level cache

    with patch("civicsetu.retrieval.reranker.settings") as mock_settings:
        mock_settings.reranker_model = "rank-T5-flan"
        with patch("flashrank.Ranker") as MockRanker:
            MockRanker.return_value = MagicMock()
            reranker_mod._get_ranker()
            MockRanker.assert_called_once_with(
                model_name="rank-T5-flan", cache_dir=".cache/flashrank"
            )
    reranker_mod._ranker = None  # clean up


# ── _apply_score_gap ──────────────────────────────────────────────────────────

def test_score_gap_empty():
    from civicsetu.retrieval.reranker import _apply_score_gap
    assert _apply_score_gap([], gap=0.35) == []


def test_score_gap_single():
    from civicsetu.retrieval.reranker import _apply_score_gap
    c = _make_rc(rerank_score=0.8)
    assert _apply_score_gap([c], gap=0.35) == [c]


def test_score_gap_no_gap():
    """All chunks pass when consecutive drops are below the gap threshold."""
    from civicsetu.retrieval.reranker import _apply_score_gap
    chunks = [
        _make_rc(rerank_score=0.85),
        _make_rc(rerank_score=0.75),
        _make_rc(rerank_score=0.65),
    ]
    result = _apply_score_gap(chunks, gap=0.35)
    assert len(result) == 3


def test_score_gap_stops_at_cliff():
    """Stops after the chunk BEFORE the large drop."""
    from civicsetu.retrieval.reranker import _apply_score_gap
    chunks = [
        _make_rc(rerank_score=0.88),
        _make_rc(rerank_score=0.82),
        _make_rc(rerank_score=0.40),  # gap from prev = 0.42 >= 0.35 → stop before this
        _make_rc(rerank_score=0.35),
    ]
    result = _apply_score_gap(chunks, gap=0.35)
    assert len(result) == 2
    assert result[0].rerank_score == 0.88
    assert result[1].rerank_score == 0.82


def test_score_gap_gap_at_first_pair():
    """If the very first pair has a cliff, only the top chunk is kept."""
    from civicsetu.retrieval.reranker import _apply_score_gap
    chunks = [
        _make_rc(rerank_score=0.90),
        _make_rc(rerank_score=0.40),
    ]
    result = _apply_score_gap(chunks, gap=0.35)
    assert len(result) == 1
    assert result[0].rerank_score == 0.90


def test_score_gap_old_threshold_cuts_aggressively():
    """Old gap=0.35 cuts after position 1 when second chunk drops by 0.36."""
    from civicsetu.retrieval.reranker import _apply_score_gap
    chunks = [
        _make_rc(rerank_score=0.88),
        _make_rc(rerank_score=0.52),  # gap = 0.36 >= 0.35 → cut
        _make_rc(rerank_score=0.40),
    ]
    result = _apply_score_gap(chunks, gap=0.35)
    assert len(result) == 1


def test_score_gap_new_threshold_keeps_more():
    """New gap=0.6 keeps chunks unless there's a 0.6+ cliff."""
    from civicsetu.retrieval.reranker import _apply_score_gap
    chunks = [
        _make_rc(rerank_score=0.88),
        _make_rc(rerank_score=0.52),  # gap = 0.36 < 0.6 → keep
        _make_rc(rerank_score=0.40),  # gap = 0.12 < 0.6 → keep
    ]
    result = _apply_score_gap(chunks, gap=0.6)
    assert len(result) == 3


def test_score_gap_new_threshold_still_cuts_on_cliff():
    """New gap=0.6 still cuts when there's a genuine cliff."""
    from civicsetu.retrieval.reranker import _apply_score_gap
    chunks = [
        _make_rc(rerank_score=0.88),
        _make_rc(rerank_score=0.20),  # gap = 0.68 >= 0.6 → cut
    ]
    result = _apply_score_gap(chunks, gap=0.6)
    assert len(result) == 1
    assert result[0].rerank_score == 0.88


# ── reranker_node threshold + gap filtering ───────────────────────────────────

def test_reranker_drops_below_threshold():
    """Chunks scoring below reranker_score_threshold must not appear in output."""
    from civicsetu.agent.nodes import reranker_node
    from unittest.mock import patch, MagicMock

    c_high = _make_rc(section_id="1")
    c_low  = _make_rc(section_id="2")

    mock_results = [
        {"id": 0, "score": 0.80},   # above 0.3 threshold
        {"id": 1, "score": 0.10},   # below 0.3 threshold → dropped
    ]
    with patch("flashrank.Ranker") as MockRanker:
        instance = MockRanker.return_value
        instance.rerank.return_value = mock_results
        with patch("civicsetu.retrieval.reranker.settings") as ms:
            ms.reranker_model = "ms-marco-MiniLM-L-12-v2"
            ms.reranker_score_threshold = 0.3
            ms.reranker_score_gap = 0.35
            state = _base_state(
                retrieved_chunks=[c_high, c_low],
                reranked_chunks=[],
                query="test query",
            )
            result = reranker_node(state)

    section_ids = [c.chunk.section_id for c in result["reranked_chunks"]]
    assert "1" in section_ids
    assert "2" not in section_ids


def test_reranker_applies_score_gap():
    """A large score cliff stops inclusion even if remaining chunks exceed threshold."""
    from civicsetu.agent.nodes import reranker_node
    from unittest.mock import patch, MagicMock

    c1 = _make_rc(section_id="1")
    c2 = _make_rc(section_id="2")
    c3 = _make_rc(section_id="3")

    # gap between id=1 (0.82) and id=2 (0.40) = 0.42 >= 0.35 → id=2 dropped
    mock_results = [
        {"id": 0, "score": 0.88},
        {"id": 1, "score": 0.82},
        {"id": 2, "score": 0.40},
    ]
    with patch("flashrank.Ranker") as MockRanker:
        instance = MockRanker.return_value
        instance.rerank.return_value = mock_results
        with patch("civicsetu.retrieval.reranker.settings") as ms:
            ms.reranker_model = "ms-marco-MiniLM-L-12-v2"
            ms.reranker_score_threshold = 0.3
            ms.reranker_score_gap = 0.35
            state = _base_state(
                retrieved_chunks=[c1, c2, c3],
                reranked_chunks=[],
                query="test query",
            )
            result = reranker_node(state)

    section_ids = [c.chunk.section_id for c in result["reranked_chunks"]]
    assert "1" in section_ids
    assert "2" in section_ids
    assert "3" not in section_ids


def test_reranker_filtered_count_logged():
    """reranker_filtered log event must report correct before/after/dropped counts."""
    from civicsetu.agent.nodes import reranker_node
    from unittest.mock import patch, MagicMock

    c1 = _make_rc(section_id="1")
    c2 = _make_rc(section_id="2")

    mock_results = [
        {"id": 0, "score": 0.80},
        {"id": 1, "score": 0.05},  # dropped below threshold
    ]
    with patch("flashrank.Ranker") as MockRanker:
        instance = MockRanker.return_value
        instance.rerank.return_value = mock_results
        with patch("civicsetu.retrieval.reranker.settings") as ms:
            ms.reranker_model = "ms-marco-MiniLM-L-12-v2"
            ms.reranker_score_threshold = 0.3
            ms.reranker_score_gap = 0.35
            with patch("civicsetu.retrieval.reranker.log") as mock_log:
                state = _base_state(
                    retrieved_chunks=[c1, c2],
                    reranked_chunks=[],
                    query="test query",
                )
                result = reranker_node(state)

    # Verify reranked_chunks output
    assert len(result["reranked_chunks"]) == 1
    assert result["reranked_chunks"][0].chunk.section_id == "1"

    # Verify the reranker_filtered log event was emitted with correct counts
    log_calls = {call.args[0]: call.kwargs for call in mock_log.info.call_args_list}
    assert "reranker_filtered" in log_calls, "reranker_filtered log event not emitted"
    rf = log_calls["reranker_filtered"]
    assert rf["before"] == 2
    assert rf["after"] == 1
    assert rf["dropped"] == 1
