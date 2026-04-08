from __future__ import annotations

import asyncio
import json
import time

import litellm
import structlog

from civicsetu.agent.state import CivicSetuState
from civicsetu.config.settings import get_settings
from civicsetu.models.enums import QueryType, Jurisdiction
from civicsetu.models.schemas import Citation, RetrievedChunk
from civicsetu.prompts.classifier import CLASSIFIER_PROMPT
from civicsetu.prompts.generator import GENERATOR_PROMPT
from civicsetu.retrieval import cached_embed
from civicsetu.stores.relational_store import AsyncSessionLocal
from civicsetu.stores.vector_store import VectorStore

log = structlog.get_logger(__name__)
settings = get_settings()
_ranker = None


# ── LiteLLM fallback chain ─────────────────────────────────────────────────────

FALLBACK_MODELS = [
    settings.primary_model,
    settings.fallback_model_2,
    settings.fallback_model_1,
]


def _get_ranker():
    global _ranker
    if _ranker is not None:
        return _ranker

    from flashrank import Ranker
    ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=".cache/flashrank")

    # Do not cache unittest mocks, otherwise tests that patch Ranker bleed into each other.
    if type(ranker).__module__ != "unittest.mock":
        _ranker = ranker
    return ranker


def turn_reset_node(state: CivicSetuState) -> dict:
    """
    Clear per-turn fields while preserving session-scoped inputs and messages.
    """
    log.info("turn_reset", session_id=state.get("session_id"))
    return {
        "query_type": None,
        "rewritten_query": None,
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "raw_response": None,
        "citations": [],
        "confidence_score": 0.0,
        "conflict_warnings": [],
        "amendment_notice": None,
        "retry_count": 0,
        "hallucination_flag": False,
        "error": None,
    }


def _llm_call(prompt: str, system: str, temperature: float = 0.0) -> str:
    """
    LiteLLM call with automatic fallback chain.
    temperature=0.0 for all legal reasoning — determinism over creativity.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    last_error = None
    for model in FALLBACK_MODELS:
        start = time.perf_counter()
        try:
            # Gemini 3.x models degrade below temperature=1.0
            effective_temp = 1.0 if "gemini-3" in model else temperature
            response = litellm.completion(
                model=model,
                messages=messages,
                temperature=effective_temp,
                max_tokens=1024,
            )
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            content = response.choices[0].message.content.strip()
            usage = getattr(response, "usage", None)
            log.info(
                "llm_call_complete",
                model=model,
                duration_ms=duration_ms,
                prompt_chars=len(prompt),
                output_chars=len(content),
                prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
                completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
            )
            return content
        except Exception as e:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            log.warning("llm_fallback", model=model, duration_ms=duration_ms, error=str(e))
            last_error = e
            continue

    raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")


def _generator_tone_hint(query_type: QueryType | str | None) -> str:
    """
    Return tone guidance for the generator based on classifier output.
    """
    if isinstance(query_type, str):
        query_type = QueryType._value2member_map_.get(query_type)

    tone_hints = {
        QueryType.FACT_LOOKUP: "Tone hint: Give a direct answer and include one helpful analogy.",
        QueryType.PENALTY_LOOKUP: "Tone hint: Lead with the consequence, then explain why it applies.",
        QueryType.CROSS_REFERENCE: "Tone hint: Explain the connection between sections as a narrative.",
        QueryType.CONFLICT_DETECTION: (
            "Tone hint: Explicitly flag the contradiction, explain both sides, "
            "and state which jurisdiction takes precedence when the context supports it."
        ),
        QueryType.TEMPORAL: "Tone hint: Explain what changed, when, and why it matters.",
    }
    return tone_hints.get(query_type, tone_hints[QueryType.FACT_LOOKUP])


# ── Node 1: Classifier ─────────────────────────────────────────────────────────

def classifier_node(state: CivicSetuState) -> dict:
    """
    Classifies query type and rewrites the query for better retrieval.
    Returns: query_type, rewritten_query
    """
    node_start = time.perf_counter()
    # ── Skip classifier for graph-context queries (section drawer flow) ───────
    if state.get("skip_classifier"):
        log.info("classifier_node_skipped", reason="skip_classifier=True")
        log.info("node_timing", node="classifier", duration_ms=round((time.perf_counter() - node_start) * 1000, 2))
        return {
            "query_type": state["query_type"],
            "rewritten_query": state.get("rewritten_query") or state["query"],
        }
    # ─────────────────────────────────────────────────────────────────────────
    query = state["query"]
    log.info("classifier_node", query=query[:80])

    prompt = CLASSIFIER_PROMPT.format(query=query)
    system = (
        "You are a legal query classifier for Indian law. "
        "Respond only with valid JSON. No explanation."
    )

    try:
        raw = _llm_call(prompt, system)
        # Strip markdown code fences if present
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        result = json.loads(raw)

        query_type_str = result.get("query_type", "fact_lookup")
        valid_types = QueryType._value2member_map_
        query_type = QueryType(query_type_str) if query_type_str in valid_types else QueryType.FACT_LOOKUP
        rewritten = result.get("rewritten_query", query)

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        log.warning("classifier_parse_failed", error=str(e), fallback="fact_lookup")
        query_type = QueryType.FACT_LOOKUP
        rewritten = query

    log.info("classified", query_type=query_type.value, rewritten=rewritten[:80])
    log.info("node_timing", node="classifier", duration_ms=round((time.perf_counter() - node_start) * 1000, 2))
    return {"query_type": query_type, "rewritten_query": rewritten}


# ── Node 2: Vector Retriever ───────────────────────────────────────────────────

# ── Node 2: Vector Retriever ───────────────────────────────────────────────────

def vector_retrieval_node(state: CivicSetuState) -> dict:
    """
    Embeds the rewritten query and retrieves top_k chunks from pgvector.
    After similarity search, expands any base section hit into its full
    sub-chunk family (e.g. S11 → S11 + S11(3) ... S11(22)) so the reranker
    can pick the most relevant sub-sections rather than only the first chunk.
    Returns: retrieved_chunks
    """
    import re
    query = state.get("rewritten_query") or state["query"]
    top_k = state.get("top_k", 5)
    jurisdiction = state.get("jurisdiction_filter")
    node_start = time.perf_counter()

    log.info("vector_retrieval_node", query=query[:80], top_k=top_k)

    embed_start = time.perf_counter()
    query_embedding = cached_embed(query)
    log.info("stage_timing", node="vector_retrieval", stage="embedding", duration_ms=round((time.perf_counter() - embed_start) * 1000, 2))

    async def _retrieve():
        async with AsyncSessionLocal() as session:
            results = await VectorStore.similarity_search(
                session=session,
                query_embedding=query_embedding,
                top_k=top_k,
                jurisdiction=jurisdiction,
                active_only=True,
            )

            # Expand base section hits into full sub-chunk families.
            # S11 scores 0.735 but only contains subsection (1) — the
            # remaining 17 sub-chunks score too low to surface individually.
            # Fetching the family gives the reranker the full section content.
            seen_ids: set[str] = {str(r.chunk.chunk_id) for r in results}
            expanded: list[RetrievedChunk] = list(results)

            for rc in results[:1]:
                sid = rc.chunk.section_id
                if not re.search(r'\(', str(sid)):  # base section only — skip sub-chunks
                    jur = Jurisdiction(rc.chunk.jurisdiction)
                    family = await VectorStore.get_section_family(
                        session=session, section_id=sid, jurisdiction=jur
                    )
                    for fc in family:
                        cid = str(fc.chunk.chunk_id)
                        if cid not in seen_ids:
                            seen_ids.add(cid)
                            expanded.append(fc)

            _MAX_VECTOR_EXPANDED = 25
            expanded = expanded[:_MAX_VECTOR_EXPANDED]

            log.info(
                "vector_retrieval_complete",
                base_results=len(results),
                results=len(expanded),
            )
            return expanded

    retrieve_start = time.perf_counter()
    chunks = asyncio.run(_retrieve())
    log.info("stage_timing", node="vector_retrieval", stage="postgres_retrieval", duration_ms=round((time.perf_counter() - retrieve_start) * 1000, 2))
    log.info("node_timing", node="vector_retrieval", duration_ms=round((time.perf_counter() - node_start) * 1000, 2), results=len(chunks))
    return {"retrieved_chunks": chunks}


def graph_retrieval_node(state: CivicSetuState) -> dict:
    """
    Graph-based retrieval for cross_reference and temporal queries.
    Traverses REFERENCES edges in Neo4j then hydrates chunks from pgvector.
    Results are merged with vector results via Annotated[list, operator.add].
    """
    from civicsetu.retrieval.graph_retriever import GraphRetriever

    query = state.get("rewritten_query") or state["query"]
    jurisdiction = state.get("jurisdiction_filter")
    top_k = state.get("top_k", 5)
    node_start = time.perf_counter()

    log.info("graph_retrieval_node", query=query[:80])

    async def _retrieve():
        return await GraphRetriever.retrieve(
            query=query,
            jurisdiction=jurisdiction,
            depth=2,
            source_section_id=state.get("source_section_id"),
        )

    retrieve_start = time.perf_counter()
    chunks = asyncio.run(_retrieve())
    log.info("graph_retrieval_complete", results=len(chunks))
    log.info("stage_timing", node="graph_retrieval", stage="neo4j_postgres_hydration", duration_ms=round((time.perf_counter() - retrieve_start) * 1000, 2))
    
    # Fallback: if graph found nothing (no explicit section in query),
    # run vector retrieval instead
    if not chunks:
        log.info("graph_retrieval_fallback_to_vector", query=query[:80])
        embed_start = time.perf_counter()
        query_embedding = cached_embed(query)
        log.info("stage_timing", node="graph_retrieval", stage="fallback_embedding", duration_ms=round((time.perf_counter() - embed_start) * 1000, 2))

        async def _vector_fallback():
            async with AsyncSessionLocal() as session:
                return await VectorStore.similarity_search(
                    session=session,
                    query_embedding=query_embedding,
                    top_k=top_k,
                    jurisdiction=jurisdiction,
                    active_only=True,
                )

        fallback_start = time.perf_counter()
        chunks = asyncio.run(_vector_fallback())
        log.info("stage_timing", node="graph_retrieval", stage="fallback_vector_search", duration_ms=round((time.perf_counter() - fallback_start) * 1000, 2))
        log.info("graph_fallback_complete", results=len(chunks))

    _MAX_GRAPH_CHUNKS = 25
    if len(chunks) > _MAX_GRAPH_CHUNKS:
        log.warning(
            "graph_retrieval_capped",
            original=len(chunks),
            capped=_MAX_GRAPH_CHUNKS,
        )
        chunks = chunks[:_MAX_GRAPH_CHUNKS]
        
    log.info("node_timing", node="graph_retrieval", duration_ms=round((time.perf_counter() - node_start) * 1000, 2), results=len(chunks))
    return {"retrieved_chunks": chunks}
    
# ── Node 3: Reranker ───────────────────────────────────────────────────────────

def reranker_node(state: CivicSetuState) -> dict:
    """
    Reranks retrieved chunks using FlashRank cross-encoder.
    Deduplicates by chunk_id first, then scores.
    Returns: reranked_chunks (top 5 max)
    """
    from flashrank import RerankRequest

    chunks = state.get("retrieved_chunks", [])
    query = state.get("rewritten_query") or state["query"]
    node_start = time.perf_counter()

    if not chunks:
        log.info("node_timing", node="reranker", duration_ms=round((time.perf_counter() - node_start) * 1000, 2), reranked=0)
        return {"reranked_chunks": []}

    # Deduplicate by chunk_id
    seen = set()
    unique_chunks = []
    for c in chunks:
        cid = str(c.chunk.chunk_id)
        if cid not in seen:
            seen.add(cid)
            unique_chunks.append(c)

    log.info("reranker_node", unique_chunks=len(unique_chunks))

    # Separate pinned (source sections) from rankable
    pinned = [c for c in unique_chunks if c.is_pinned][:2]
    rankable = [c for c in unique_chunks if not c.is_pinned]

    try:
        init_start = time.perf_counter()
        ranker = _get_ranker()
        log.info("stage_timing", node="reranker", stage="ranker_init", duration_ms=round((time.perf_counter() - init_start) * 1000, 2))
        passages = [{"id": i, "text": c.chunk.text} for i, c in enumerate(rankable)]
        request = RerankRequest(query=query, passages=passages)
        inference_start = time.perf_counter()
        results = ranker.rerank(request)
        log.info("stage_timing", node="reranker", stage="ranker_inference", duration_ms=round((time.perf_counter() - inference_start) * 1000, 2), passages=len(passages))

        # Map scores back to chunks
        id_to_chunk = {i: c for i, c in enumerate(rankable)}
        reranked_rankable = []
        for r in results:
            chunk = id_to_chunk[r["id"]]
            chunk.rerank_score = round(float(r["score"]), 4)
            reranked_rankable.append(chunk)

        slots_for_ranked = max(0, 5 - len(pinned))
        reranked = pinned + reranked_rankable[:slots_for_ranked]


    except Exception as e:
        log.warning("reranker_failed", error=str(e), fallback="vector_order")
        slots_for_ranked = max(0, 5 - len(pinned))
        reranked = pinned + rankable[:slots_for_ranked]

    log.info("reranker_complete", reranked=len(reranked), pinned=len(pinned))
    log.info("node_timing", node="reranker", duration_ms=round((time.perf_counter() - node_start) * 1000, 2), unique_chunks=len(unique_chunks), rankable=len(rankable), pinned=len(pinned), reranked=len(reranked))
    return {"reranked_chunks": reranked}


# ── Node 4: Generator ──────────────────────────────────────────────────────────

def generator_node(state: CivicSetuState) -> dict:
    """
    Generates a cited answer from reranked chunks.
    Citations are anchored to cited_chunks indices returned by the LLM —
    only sections the answer actually references are included.
    Returns: raw_response, citations, confidence_score, amendment_notice, conflict_warnings
    """
    query = state["query"]
    chunks: list[RetrievedChunk] = state.get("reranked_chunks", [])
    messages = state.get("messages", [])
    node_start = time.perf_counter()

    if not chunks:
        log.info("node_timing", node="generator", duration_ms=round((time.perf_counter() - node_start) * 1000, 2), context_chunks=0)
        return {
            "raw_response": "Insufficient information found in indexed documents.",
            "citations": [],
            "confidence_score": 0.0,
            "conflict_warnings": [],
            "amendment_notice": None,
        }

    # Build numbered context block — [1], [2], ... so LLM can reference by index
    context_parts = []
    for i, rc in enumerate(chunks, 1):
        c = rc.chunk
        context_parts.append(
            f"[{i}] {c.doc_name} — {c.section_id}: {c.section_title}\n"
            f"Effective: {c.effective_date}\n"
            f"{c.text}\n"
        )
    context = "\n---\n".join(context_parts)

    recent_messages = messages[-6:]
    if recent_messages:
        history_lines = []
        for message in recent_messages:
            role = getattr(message, "role", None)
            content = getattr(message, "content", None)
            if role is None and isinstance(message, dict):
                role = message.get("role")
                content = message.get("content")
            if not role or not content:
                continue
            role_label = "User" if role == "user" else "Assistant"
            history_lines.append(f"{role_label}: {content}")
        conversation_history_block = (
            "Prior conversation (for context only; answer the current question):\n"
            + "\n".join(history_lines)
            + "\n\n"
            if history_lines
            else ""
        )
    else:
        conversation_history_block = ""

    prompt = GENERATOR_PROMPT.format(
        query=query,
        context=context,
        conversation_history_block=conversation_history_block,
    )
    tone_hint = _generator_tone_hint(state.get("query_type"))
    system = (
        "You are CivicSetu, a plain-language guide to Indian RERA laws for homebuyers, builders, and agents. "
        "Your job is to explain what the law means in practice - not recite it. "
        "Use simple language, real-world analogies, and short examples to make rules clear. "
        "After explaining, anchor each point to the specific section it comes from. "
        "Never invent provisions. "
        f"{tone_hint} "
        "Respond with valid JSON only."
    )

    log.info(
        "generator_node",
        query=query[:80],
        context_chunks=len(chunks),
        history_messages=len(recent_messages),
    )

    try:
        llm_start = time.perf_counter()
        raw = _llm_call(prompt, system, temperature=0.0)
        log.info("stage_timing", node="generator", stage="llm", duration_ms=round((time.perf_counter() - llm_start) * 1000, 2))
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        result = json.loads(raw)

        answer = result.get("answer", "")
        confidence = float(result.get("confidence_score", 0.5))
        amendment_notice = result.get("amendment_notice")
        conflict_warnings = result.get("conflict_warnings", [])

        # ── Citation anchoring ────────────────────────────────────────────────
        # Parse cited_chunks indices returned by LLM (1-based)
        raw_indices: list = result.get("cited_chunks", [])

        cited_chunks: list[RetrievedChunk] = []
        if raw_indices and isinstance(raw_indices, list):
            for idx in raw_indices:
                try:
                    i = int(idx)
                    if 1 <= i <= len(chunks):
                        cited_chunks.append(chunks[i - 1])
                except (ValueError, TypeError):
                    continue

        # Fallback: if LLM returned no valid indices, cite all chunks
        # Prevents empty citations on malformed LLM output
        if not cited_chunks:
            log.warning(
                "generator_cited_chunks_empty",
                raw_indices=raw_indices,
                fallback="all_chunks",
            )
            cited_chunks = chunks

        # Deduplicate by (section_id, doc_name) — same section can appear
        # multiple times from different retrieval paths
        seen: set[tuple[str, str]] = set()
        citations = []
        for rc in cited_chunks:
            c = rc.chunk
            key = (c.section_id, c.doc_name)
            if key not in seen:
                seen.add(key)
                citations.append(Citation(
                    section_id=c.section_id,
                    doc_name=c.doc_name,
                    jurisdiction=c.jurisdiction,
                    effective_date=c.effective_date,
                    source_url=c.source_url,
                    chunk_id=c.chunk_id,
                ))

        log.info(
            "generator_citations_anchored",
            total_context=len(chunks),
            cited=len(cited_chunks),
            unique_citations=len(citations),
        )

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        log.warning("generator_parse_failed", error=str(e))
        answer = "Unable to generate a structured response. Please try again."
        confidence = 0.0
        citations = []
        conflict_warnings = []
        amendment_notice = None

    log.info("node_timing", node="generator", duration_ms=round((time.perf_counter() - node_start) * 1000, 2), context_chunks=len(chunks))
    return {
        "raw_response": answer,
        "citations": citations,
        "confidence_score": confidence,
        "conflict_warnings": conflict_warnings,
        "amendment_notice": amendment_notice,
    }



# ── Node 5: Validator ──────────────────────────────────────────────────────────

def validator_node(state: CivicSetuState) -> dict:
    answer = state.get("raw_response", "")
    chunks = state.get("reranked_chunks", [])
    node_start = time.perf_counter()

    if not answer or not chunks:
        log.info("node_timing", node="validator", duration_ms=round((time.perf_counter() - node_start) * 1000, 2), skipped=True)
        return {"hallucination_flag": False, "confidence_score": 0.5}

    updated_score = state.get("confidence_score", 0.5)
    hallucinated = updated_score < 0.2
    log.info("validator_node", confidence=updated_score, hallucinated=hallucinated)
    log.info("node_timing", node="validator", duration_ms=round((time.perf_counter() - node_start) * 1000, 2))
    return {
        "hallucination_flag": hallucinated,
        "confidence_score": updated_score,
    }


def hybrid_retrieval_node(state: CivicSetuState) -> dict:
    """
    Used for conflict_detection queries.
    Runs vector + graph retrieval in parallel, merges results.
    Deduplication happens downstream in reranker_node.
    """
    from civicsetu.retrieval.graph_retriever import GraphRetriever

    query = state.get("rewritten_query") or state["query"]
    top_k = state.get("top_k", 5)
    jurisdiction = state.get("jurisdiction_filter")
    node_start = time.perf_counter()

    log.info("hybrid_retrieval_node", query=query[:80])

    embed_start = time.perf_counter()
    query_embedding = cached_embed(query)
    log.info("stage_timing", node="hybrid_retrieval", stage="embedding", duration_ms=round((time.perf_counter() - embed_start) * 1000, 2))

    async def _retrieve():
        async with AsyncSessionLocal() as session:
            vector_task = VectorStore.similarity_search(
                session=session,
                query_embedding=query_embedding,
                top_k=top_k,
                jurisdiction=jurisdiction,
                active_only=True,
            )
            graph_task = GraphRetriever.retrieve(
                query=query,
                jurisdiction=jurisdiction,
                depth=2,
            )
            vector_chunks, graph_chunks = await asyncio.gather(
                vector_task, graph_task, return_exceptions=True
            )

        v_chunks = vector_chunks if not isinstance(vector_chunks, Exception) else []
        g_chunks = graph_chunks if not isinstance(graph_chunks, Exception) else []

        if isinstance(vector_chunks, Exception):
            log.warning("hybrid_vector_failed", error=str(vector_chunks))
        if isinstance(graph_chunks, Exception):
            log.warning("hybrid_graph_failed", error=str(graph_chunks))

        return v_chunks, g_chunks

    retrieve_start = time.perf_counter()
    v_chunks, g_chunks = asyncio.run(_retrieve())
    log.info("stage_timing", node="hybrid_retrieval", stage="vector_graph_parallel", duration_ms=round((time.perf_counter() - retrieve_start) * 1000, 2))

    log.info(
        "hybrid_retrieval_complete",
        vector_chunks=len(v_chunks),
        graph_chunks=len(g_chunks),
        total=len(v_chunks) + len(g_chunks),
    )
    log.info("node_timing", node="hybrid_retrieval", duration_ms=round((time.perf_counter() - node_start) * 1000, 2), total=len(v_chunks) + len(g_chunks))
    # Return both — reranker deduplicates by chunk_id
    return {"retrieved_chunks": v_chunks + g_chunks}
