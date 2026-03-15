from __future__ import annotations

import asyncio
import json

import litellm
import structlog

from civicsetu.agent.state import CivicSetuState
from civicsetu.config.settings import get_settings
from civicsetu.ingestion.embedder import Embedder
from civicsetu.models.enums import QueryType
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

def vector_retrieval_node(state: CivicSetuState) -> dict:
    """
    Embeds the rewritten query and retrieves top_k chunks from pgvector.
    Returns: retrieved_chunks (appended via Annotated[list, operator.add])
    """
    query = state.get("rewritten_query") or state["query"]
    top_k = state.get("top_k", 5)
    jurisdiction = state.get("jurisdiction_filter")

    log.info("vector_retrieval_node", query=query[:80], top_k=top_k)

    query_embedding = _embedder.embed_query(query)

    async def _retrieve():
        async with AsyncSessionLocal() as session:
            return await VectorStore.similarity_search(
                session=session,
                query_embedding=query_embedding,
                top_k=top_k,
                jurisdiction=jurisdiction,
                active_only=True,
            )

    chunks = asyncio.run(_retrieve())
    log.info("vector_retrieval_complete", results=len(chunks))
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

    try:
        ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=".cache/flashrank")
        passages = [{"id": i, "text": c.chunk.text} for i, c in enumerate(unique_chunks)]
        request = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(request)

        # Map scores back to chunks
        reranked = []
        for r in results[:5]:
            chunk = unique_chunks[r["id"]]
            chunk.rerank_score = round(float(r["score"]), 4)
            reranked.append(chunk)

    except Exception as e:
        log.warning("reranker_failed", error=str(e), fallback="vector_order")
        reranked = unique_chunks[:5]

    log.info("reranker_complete", reranked=len(reranked))
    return {"reranked_chunks": reranked}


# ── Node 4: Generator ──────────────────────────────────────────────────────────

def generator_node(state: CivicSetuState) -> dict:
    """
    Generates a cited answer from reranked chunks.
    Returns: raw_response, citations, confidence_score, amendment_notice
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

    # Build context block
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

        # Build citations from chunks referenced in answer
        citations = []
        for rc in chunks:
            c = rc.chunk
            citations.append(Citation(
                section_id=c.section_id,
                doc_name=c.doc_name,
                jurisdiction=c.jurisdiction,
                effective_date=c.effective_date,
                source_url=c.source_url,
                chunk_id=c.chunk_id,
            ))

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
    """
    Checks if the answer is grounded in the retrieved context.
    Sets hallucination_flag=True if answer makes claims not in context.
    """
    answer = state.get("raw_response", "")
    chunks = state.get("reranked_chunks", [])

    if not answer or not chunks:
        return {"hallucination_flag": False}

    context = " ".join(c.chunk.text for c in chunks)
    prompt = VALIDATOR_PROMPT.format(answer=answer, context=context)
    system = "You are a factual grounding checker. Respond with JSON only."

    try:
        raw = _llm_call(prompt, system)
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        result = json.loads(raw)
        hallucinated = result.get("hallucination_detected", False)
        updated_score = result.get("confidence_score", state.get("confidence_score", 0.5))
    except Exception as e:
        log.warning("validator_failed", error=str(e))
        hallucinated = False
        updated_score = state.get("confidence_score", 0.5)

    log.info("validator_node", hallucinated=hallucinated, confidence=updated_score)
    return {
        "hallucination_flag": hallucinated,
        "confidence_score": float(updated_score),
    }
