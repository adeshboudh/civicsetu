from __future__ import annotations

import structlog
from langgraph.graph import END, StateGraph

from civicsetu.agent.edges import route_after_classifier, route_after_validator
from civicsetu.agent.nodes import (
    classifier_node,
    generator_node,
    reranker_node,
    validator_node,
    vector_retrieval_node,
    graph_retrieval_node,
)
from civicsetu.agent.state import CivicSetuState

log = structlog.get_logger(__name__)


def _retry_node(state: CivicSetuState) -> dict:
    """Increments retry counter and clears previous retrieval before re-classifying."""
    return {
        "retry_count": state.get("retry_count", 0) + 1,
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "raw_response": None,
        "hallucination_flag": False,
    }


def build_graph() -> StateGraph:
    graph = StateGraph(CivicSetuState)

    # ── Register nodes ─────────────────────────────────────────────────────────
    graph.add_node("classifier",        classifier_node)
    graph.add_node("vector_retrieval",  vector_retrieval_node)
    graph.add_node("graph_retrieval",   graph_retrieval_node)
    graph.add_node("reranker",          reranker_node)
    graph.add_node("generator",         generator_node)
    graph.add_node("validator",         validator_node)
    graph.add_node("retry",             _retry_node)

    # ── Entry point ────────────────────────────────────────────────────────────
    graph.set_entry_point("classifier")

    # ── Edges ──────────────────────────────────────────────────────────────────
    graph.add_conditional_edges(
        "classifier",
        route_after_classifier,
        {
            "vector_retrieval": "vector_retrieval",
            "graph_retrieval":  "graph_retrieval",
        },
    )
    graph.add_edge("vector_retrieval", "reranker")
    graph.add_edge("graph_retrieval",  "reranker")
    graph.add_edge("reranker",         "generator")
    graph.add_edge("generator",        "validator")
    graph.add_conditional_edges(
        "validator",
        route_after_validator,
        {"retry": "retry", "end": END},
    )
    graph.add_edge("retry", "classifier")

    return graph


def get_compiled_graph():
    """Returns the compiled, executable LangGraph. Call once, reuse."""
    return build_graph().compile()
