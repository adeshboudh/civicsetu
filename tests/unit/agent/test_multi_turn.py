from __future__ import annotations

from civicsetu.models.schemas import ChatMessage
from tests.conftest import _base_state, _make_rc


def test_state_has_messages_field():
    from civicsetu.agent.state import CivicSetuState

    assert "messages" in CivicSetuState.__annotations__


def test_turn_reset_clears_ephemeral_fields():
    from civicsetu.agent.nodes import turn_reset_node

    state = _base_state(
        query_type="fact_lookup",
        rewritten_query="old rewritten",
        retrieved_chunks=["chunk1"],
        reranked_chunks=["chunk1"],
        raw_response="old answer",
        confidence_score=0.9,
        retry_count=2,
        hallucination_flag=True,
        citations=["cite1"],
        conflict_warnings=["warning"],
        amendment_notice="amended",
        error="some error",
    )

    result = turn_reset_node(state)

    assert result["query_type"] is None
    assert result["rewritten_query"] is None
    assert result["retrieved_chunks"] == []
    assert result["reranked_chunks"] == []
    assert result["raw_response"] is None
    assert result["confidence_score"] == 0.0
    assert result["retry_count"] == 0
    assert result["hallucination_flag"] is False
    assert result["citations"] == []
    assert result["conflict_warnings"] == []
    assert result["amendment_notice"] is None
    assert result["error"] is None
    assert "messages" not in result


def test_generator_includes_history_in_prompt():
    from civicsetu.agent.nodes import generator_node

    state = _base_state(
        query="What about promoter obligations specifically?",
        messages=[
            ChatMessage(role="user", content="What is RERA?"),
            ChatMessage(role="assistant", content="RERA stands for Real Estate Regulation Act."),
        ],
        reranked_chunks=[_make_rc(section_id="11")],
    )
    captured = {}

    def fake_llm_call(prompt: str, system: str, temperature: float = 0.0) -> str:
        captured["prompt"] = prompt
        return (
            '{"answer":"Promoters must register.","confidence_score":0.9,'
            '"cited_chunks":[1],"conflict_warnings":[],"amendment_notice":null}'
        )

    from unittest.mock import patch

    with patch("civicsetu.agent.nodes._llm_call", side_effect=fake_llm_call):
        generator_node(state)

    assert "What is RERA?" in captured["prompt"]
    assert "RERA stands for Real Estate Regulation Act." in captured["prompt"]


def test_build_graph_compiles_with_checkpointer():
    from civicsetu.agent.graph import build_graph
    from langgraph.checkpoint.memory import InMemorySaver

    graph = build_graph()
    compiled = graph.compile(checkpointer=InMemorySaver())
    assert compiled is not None


# ── History accumulation ────────────────────────────────────────────────────────

def test_messages_list_is_annotated_with_operator_add():
    """operator.add on the messages field means LangGraph merges lists across turns."""
    import operator
    from typing import get_args, get_type_hints
    from civicsetu.agent.state import CivicSetuState

    hints = get_type_hints(CivicSetuState, include_extras=True)
    args = get_args(hints["messages"])
    # args == (list[ChatMessage], operator.add)
    assert args[1] is operator.add


def test_generator_limits_history_to_last_6_messages():
    """Only the 6 most recent messages are injected so context windows don't blow up."""
    from civicsetu.agent.nodes import generator_node

    old_messages = [
        ChatMessage(role="user", content=f"Old question {i}") for i in range(10)
    ]
    state = _base_state(
        query="Latest question",
        messages=old_messages,
        reranked_chunks=[_make_rc(section_id="11")],
    )
    captured = {}

    def fake_llm_call(prompt: str, system: str, temperature: float = 0.0) -> str:
        captured["prompt"] = prompt
        return (
            '{"answer":"Answer.","confidence_score":0.9,'
            '"cited_chunks":[1],"conflict_warnings":[],"amendment_notice":null}'
        )

    from unittest.mock import patch

    with patch("civicsetu.agent.nodes._llm_call", side_effect=fake_llm_call):
        generator_node(state)

    # Old question 0-3 must NOT appear; 4-9 (the last 6) must appear
    assert "Old question 0" not in captured["prompt"]
    assert "Old question 4" in captured["prompt"]
    assert "Old question 9" in captured["prompt"]


def test_generator_no_history_block_when_messages_empty():
    """When messages list is empty the history block is absent — no empty section injected."""
    from civicsetu.agent.nodes import generator_node

    state = _base_state(
        query="What is RERA?",
        messages=[],
        reranked_chunks=[_make_rc(section_id="3")],
    )
    captured = {}

    def fake_llm_call(prompt: str, system: str, temperature: float = 0.0) -> str:
        captured["prompt"] = prompt
        return (
            '{"answer":"RERA stands for Real Estate.","confidence_score":0.8,'
            '"cited_chunks":[1],"conflict_warnings":[],"amendment_notice":null}'
        )

    from unittest.mock import patch

    with patch("civicsetu.agent.nodes._llm_call", side_effect=fake_llm_call):
        generator_node(state)

    assert "Prior conversation" not in captured["prompt"]


def test_generator_history_uses_correct_role_labels():
    """User messages are labelled 'User:', assistant messages 'Assistant:'."""
    from civicsetu.agent.nodes import generator_node

    state = _base_state(
        query="What penalties apply?",
        messages=[
            ChatMessage(role="user", content="Tell me about RERA."),
            ChatMessage(role="assistant", content="RERA regulates real estate."),
        ],
        reranked_chunks=[_make_rc(section_id="63")],
    )
    captured = {}

    def fake_llm_call(prompt: str, system: str, temperature: float = 0.0) -> str:
        captured["prompt"] = prompt
        return (
            '{"answer":"Penalty is up to 5%.","confidence_score":0.85,'
            '"cited_chunks":[1],"conflict_warnings":[],"amendment_notice":null}'
        )

    from unittest.mock import patch

    with patch("civicsetu.agent.nodes._llm_call", side_effect=fake_llm_call):
        generator_node(state)

    assert "User: Tell me about RERA." in captured["prompt"]
    assert "Assistant: RERA regulates real estate." in captured["prompt"]


# ── Thread isolation ────────────────────────────────────────────────────────────

def test_different_thread_ids_do_not_share_messages():
    """Two sessions with distinct thread_ids must never bleed messages into each other."""
    from langgraph.checkpoint.memory import InMemorySaver
    from civicsetu.agent.graph import build_graph

    graph = build_graph().compile(checkpointer=InMemorySaver())

    def _fake_invoke(graph, state, config):
        """Directly update checkpoint with a fake assistant message to simulate a turn."""
        graph.update_state(config, {"messages": [ChatMessage(role="user", content=state["query"])]})
        graph.update_state(config, {"messages": [ChatMessage(role="assistant", content="Session answer.")]})

    config_a = {"configurable": {"thread_id": "session-A"}}
    config_b = {"configurable": {"thread_id": "session-B"}}

    _fake_invoke(graph, {"query": "Question for A"}, config_a)
    _fake_invoke(graph, {"query": "Question for B"}, config_b)

    state_a = graph.get_state(config_a)
    state_b = graph.get_state(config_b)

    messages_a = state_a.values.get("messages", [])
    messages_b = state_b.values.get("messages", [])

    contents_a = [m.content if hasattr(m, "content") else m["content"] for m in messages_a]
    contents_b = [m.content if hasattr(m, "content") else m["content"] for m in messages_b]

    assert "Question for A" not in contents_b
    assert "Question for B" not in contents_a


# ── turn_reset sentinel ─────────────────────────────────────────────────────────

def test_turn_reset_does_not_touch_session_id_or_jurisdiction():
    """Session-level fields must survive turn_reset so routing stays consistent."""
    from civicsetu.agent.nodes import turn_reset_node

    state = _base_state(
        session_id="abc-123",
        jurisdiction_filter="MAHARASHTRA",
        query="Anything",
    )
    result = turn_reset_node(state)

    assert "session_id" not in result
    assert "jurisdiction_filter" not in result
    assert "query" not in result
