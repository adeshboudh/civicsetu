# ADR 003 — LangGraph over LangChain Chains

**Date:** March 2026
**Status:** Accepted

---

## Context

CivicSetu requires:
- Conditional routing (different retrieval strategies per query type)
- Retry loops (hallucination detected → re-classify → re-retrieve)
- Parallel retrieval (vector + graph simultaneously in Phase 1)
- Stateful multi-turn sessions (Phase 2)

LangChain sequential chains cannot express conditional branching or retry loops
without imperative Python code that bypasses the framework entirely.

## Decision

Use LangGraph `StateGraph` for all query orchestration.

Each processing step is a **pure function node** with signature:
```python
def node(state: CivicSetuState) -> dict:
    ...
```

Routing is expressed as **conditional edges** — not if/else in application code.

## Consequences

**Positive:**

- Retry loop is a graph edge — not a try/except block
- Phase 1 parallel retrieval = add one node + one edge
- Every node is independently unit-testable
- State is fully inspectable at every step (tracing, debugging)

**Negative:**

- LangGraph has a steeper learning curve than LangChain chains
- Graph compilation step required at startup


## Alternatives Rejected

- **LangChain LCEL chains:** Cannot express retry loops or conditional branching cleanly
- **Raw Python orchestration:** No state management, no tracing, no parallelism primitive
- **CrewAI / AutoGen:** Agent frameworks add unnecessary complexity for a deterministic pipeline