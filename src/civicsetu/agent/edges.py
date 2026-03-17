from __future__ import annotations

from civicsetu.agent.state import CivicSetuState
from civicsetu.models.enums import QueryType


def route_after_classifier(state: CivicSetuState) -> str:
    """
    After classification, all Phase 0 queries go to vector retrieval.
    Phase 1 will route cross_reference + temporal to graph_retrieval.
    """
    query_type = state.get("query_type", QueryType.FACT_LOOKUP)

    # Phase 1 hooks — wired but dormant until graph store is live
    if query_type in (
        QueryType.CROSS_REFERENCE,
        QueryType.TEMPORAL,
        QueryType.PENALTY_LOOKUP,
    ):
        return "graph_retrieval"

    return "vector_retrieval"



def route_after_validator(state: CivicSetuState) -> str:
    """
    After validation:
    - Hallucination detected + retries remaining → retry from classifier
    - Confidence too low + retries remaining → retry from classifier
    - Otherwise → end
    """
    hallucinated = state.get("hallucination_flag", False)
    confidence = state.get("confidence_score", 1.0)
    retry_count = state.get("retry_count", 0)
    has_chunks = len(state.get("reranked_chunks", [])) > 0

    # Don't retry if retrieval returned nothing — retrying won't help
    if not has_chunks:
        return "end"

    if (hallucinated or confidence < 0.50) and retry_count < 2:
        return "retry"

    return "end"
