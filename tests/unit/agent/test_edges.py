from __future__ import annotations

import pytest

from civicsetu.agent.edges import route_after_classifier, route_after_validator
from civicsetu.models.enums import QueryType

# ── route_after_classifier ────────────────────────────────────────────────────

@pytest.mark.parametrize("query_type,expected_node", [
    (QueryType.CONFLICT_DETECTION, "hybrid_retrieval"),
    (QueryType.CROSS_REFERENCE,    "graph_retrieval"),
    (QueryType.PENALTY_LOOKUP,     "graph_retrieval"),
    (QueryType.TEMPORAL,           "graph_retrieval"),
    (QueryType.FACT_LOOKUP,        "vector_retrieval"),
    (None,                         "vector_retrieval"),
])
def test_route_after_classifier(query_type, expected_node):
    state = {"query_type": query_type}
    assert route_after_classifier(state) == expected_node


# ── route_after_validator ─────────────────────────────────────────────────────

def _validator_state(hallucinated, confidence, retry_count, has_chunks=True):
    from tests.conftest import _make_rc
    return {
        "hallucination_flag": hallucinated,
        "confidence_score": confidence,
        "retry_count": retry_count,
        "reranked_chunks": [_make_rc()] if has_chunks else [],
    }


def test_route_to_retry_on_very_low_confidence():
    state = _validator_state(hallucinated=True, confidence=0.1, retry_count=0)
    assert route_after_validator(state) == "retry"


def test_route_to_end_when_confidence_not_low_enough():
    state = _validator_state(hallucinated=True, confidence=0.3, retry_count=1)
    assert route_after_validator(state) == "end"


def test_route_to_end_on_max_retries():
    state = _validator_state(hallucinated=True, confidence=0.3, retry_count=2)
    assert route_after_validator(state) == "end"


def test_route_to_end_on_pass():
    state = _validator_state(hallucinated=False, confidence=0.8, retry_count=0)
    assert route_after_validator(state) == "end"


def test_route_to_end_when_no_chunks():
    # No chunks → retrying won't help, skip retry regardless of hallucination
    state = _validator_state(hallucinated=True, confidence=0.1, retry_count=0, has_chunks=False)
    assert route_after_validator(state) == "end"
