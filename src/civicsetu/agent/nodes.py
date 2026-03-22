from __future__ import annotations

import asyncio
import json

import litellm
import structlog

from civicsetu.agent.state import CivicSetuState
from civicsetu.config.settings import get_settings
from civicsetu.ingestion.embedder import Embedder
from civicsetu.models.enums import QueryType, Jurisdiction
from civicsetu.models.schemas import Citation, RetrievedChunk
from civicsetu.prompts.classifier import CLASSIFIER_PROMPT
from civicsetu.prompts.generator import GENERATOR_PROMPT
from civicsetu.prompts.validator import VALIDATOR_PROMPT
from civicsetu.stores.relational_store import AsyncSessionLocal
from civicsetu.stores.vector_store import VectorStore

log = structlog.get_logger(__name__)
settings = get_settings()
_embedder = Embedder()


# ── LiteLLM fallback chain ─────────────────────────────────────────────────────

FALLBACK_MODELS = [
    settings.primary_model,
    settings.fallback_model_1,
    settings.fallback_model_2,
]


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
        try:
            # Gemini 3.x models degrade below temperature=1.0
            effective_temp = 1.0 if "gemini-3" in model else temperature
            response = litellm.completion(
                model=model,
                messages=messages,
                temperature=effective_temp,
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            log.warning("llm_fallback", model=model, error=str(e))
            last_error = e
            continue

    raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")


# ── Node 1: Classifier ─────────────────────────────────────────────────────────

def classifier_node(state: CivicSetuState) -> dict:
    """
    Classifies query type and rewrites the query for better retrieval.
    Returns: query_type, rewritten_query
    """
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

    log.info("vector_retrieval_node", query=query[:80], top_k=top_k)

    query_embedding = _embedder.embed_query(query)

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

    chunks = asyncio.run(_retrieve())
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

    log.info("graph_retrieval_node", query=query[:80])

    async def _retrieve():
        return await GraphRetriever.retrieve(
            query=query,
            jurisdiction=jurisdiction,
            depth=2,
        )

    chunks = asyncio.run(_retrieve())
    log.info("graph_retrieval_complete", results=len(chunks))
    
    # Fallback: if graph found nothing (no explicit section in query),
    # run vector retrieval instead
    if not chunks:
        log.info("graph_retrieval_fallback_to_vector", query=query[:80])
        query_embedding = _embedder.embed_query(query)

        async def _vector_fallback():
            async with AsyncSessionLocal() as session:
                return await VectorStore.similarity_search(
                    session=session,
                    query_embedding=query_embedding,
                    top_k=top_k,
                    jurisdiction=jurisdiction,
                    active_only=True,
                )

        chunks = asyncio.run(_vector_fallback())
        log.info("graph_fallback_complete", results=len(chunks))

    _MAX_GRAPH_CHUNKS = 25
    if len(chunks) > _MAX_GRAPH_CHUNKS:
        log.warning(
            "graph_retrieval_capped",
            original=len(chunks),
            capped=_MAX_GRAPH_CHUNKS,
        )
        chunks = chunks[:_MAX_GRAPH_CHUNKS]
        
    return {"retrieved_chunks": chunks}
    
# ── Node 3: Reranker ───────────────────────────────────────────────────────────

def reranker_node(state: CivicSetuState) -> dict:
    """
    Reranks retrieved chunks using FlashRank cross-encoder.
    Deduplicates by chunk_id first, then scores.
    Returns: reranked_chunks (top 5 max)
    """
    from flashrank import Ranker, RerankRequest

    chunks = state.get("retrieved_chunks", [])
    query = state.get("rewritten_query") or state["query"]

    if not chunks:
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
        ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=".cache/flashrank")
        passages = [{"id": i, "text": c.chunk.text} for i, c in enumerate(rankable)]
        request = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(request)

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

    if not chunks:
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

    prompt = GENERATOR_PROMPT.format(query=query, context=context)
    system = (
        "You are CivicSetu, a legal information assistant for Indian law. "
        "Answer only from the provided context. "
        "Every claim must be traceable to a specific section. "
        "Respond with valid JSON only."
    )

    log.info("generator_node", query=query[:80], context_chunks=len(chunks))

    try:
        raw = _llm_call(prompt, system, temperature=0.0)
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

    if not answer or not chunks:
        return {"hallucination_flag": False, "confidence_score": 0.5}

    # Build context with section metadata — same format as generator_node
    # so the validator can verify citations like "Section 11(1)" actually exist
    context_parts = []
    for i, rc in enumerate(chunks, 1):
        c = rc.chunk
        context_parts.append(
            f"[{i}] {c.doc_name} — {c.section_id}: {c.section_title}\n"
            f"Effective: {c.effective_date}\n"
            f"{c.text}\n"
        )
    context = "\n---\n".join(context_parts)

    query_type = str(state.get("query_type", "fact_lookup"))
    prompt = VALIDATOR_PROMPT.format(
        query_type=query_type, answer=answer, context=context
    )
    system = "You are a factual grounding scorer. Respond with JSON only."

    try:
        raw = _llm_call(prompt, system)
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        result = json.loads(raw)
        updated_score = float(result.get("confidence_score", 0.5))
        hallucinated = updated_score < 0.3
    except Exception as e:
        log.warning("validator_failed", error=str(e))
        updated_score = state.get("confidence_score", 0.5)
        hallucinated = False

    log.info("validator_node", confidence=updated_score, hallucinated=hallucinated)
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

    log.info("hybrid_retrieval_node", query=query[:80])

    query_embedding = _embedder.embed_query(query)

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

    v_chunks, g_chunks = asyncio.run(_retrieve())

    log.info(
        "hybrid_retrieval_complete",
        vector_chunks=len(v_chunks),
        graph_chunks=len(g_chunks),
        total=len(v_chunks) + len(g_chunks),
    )
    # Return both — reranker deduplicates by chunk_id
    return {"retrieved_chunks": v_chunks + g_chunks}
