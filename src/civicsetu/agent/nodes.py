from __future__ import annotations

import asyncio
import json
import os
import re
import time

import litellm
import structlog

from civicsetu.agent.state import CivicSetuState
from civicsetu.config.settings import get_settings
from civicsetu.models.enums import Jurisdiction, QueryType
from civicsetu.models.schemas import Citation, RetrievedChunk
from civicsetu.prompts.classifier import CLASSIFIER_PROMPT
from civicsetu.prompts.generator import GENERATOR_PROMPT
from civicsetu.retrieval import cached_embed
from civicsetu.stores.vector_store import VectorStore

log = structlog.get_logger(__name__)
settings = get_settings()

_PINNED_SECTION_RE = re.compile(
    r"^\s*(?:(section|sec\.?|s\.|rule)\s*)?(\d+(?:\([^)]*\))?[A-Z]?)\s*$",
    re.IGNORECASE,
)
_PIN_HINT_STOPWORDS = {
    "about", "after", "against", "along", "and", "are", "authority", "central",
    "does", "each", "from", "handle", "have", "into", "karnataka", "must",
    "project", "promoter", "rera", "rule", "section", "shall", "that", "the",
    "this", "under", "what", "when", "which", "with",
}
_PIN_HINT_EXPANSIONS = {
    "account": {"bank", "deposited", "separate", "seventy"},
    "accounts": {"bank", "deposited", "separate", "seventy"},
    "comply": {"default", "required", "rules", "regulations"},
    "compliance": {"default", "required", "rules", "regulations"},
    "fraudulent": {"deceptive", "irregularities", "misleading", "unfair"},
    "maintenance": {"bank", "deposited", "separate", "seventy"},
    "penalties": {"penalty", "default"},
}


# ── LiteLLM fallback chains ────────────────────────────────────────────────────
# THINKING tier: deep-reasoning model (generator only)
# FAST tier: lightweight models (classifier, validator, and other simple tasks)

THINKING_MODELS = [
    settings.primary_model,          # e.g. minimax-m2.7 (thinking)
    settings.fallback_model_1,
    settings.fallback_model_2,
    settings.fallback_model_3,
]

FAST_MODELS = [
    settings.fast_model,             # e.g. gemini-3.1-flash-lite
    settings.fallback_model_1,
    settings.fallback_model_2,
    settings.fallback_model_3,
]

# Keep backward-compat alias for any external references
FALLBACK_MODELS = THINKING_MODELS


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


def _llm_call(prompt: str, system: str, temperature: float = 0.0, tier: str = "thinking") -> str:
    """
    Call an LLM with fallback chain.

    Args:
        tier: "thinking" for deep-reasoning tasks (generator),
              "fast" for lightweight tasks (classifier, validator).
    """
    models = FAST_MODELS if tier == "fast" else THINKING_MODELS
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    last_error = None
    for model in models:
        start = time.perf_counter()
        try:
            effective_temp = 1.0 if "gemini-3" in model else temperature
            completion_kwargs = {
                "model": model,
                "messages": messages,
                "temperature": effective_temp,
                "max_tokens": 16384,
            }
            # ── NVIDIA-hosted models (GLM4.7, Minimax) ───────────────
            _nvidia_models = ("z-ai/glm4.7", "minimaxai/minimax-m2.7", "deepseek-ai/deepseek-v4-pro", "deepseek-ai/deepseek-v4-flash")
            if any(nm in model for nm in _nvidia_models):
                completion_kwargs["api_base"] = "https://integrate.api.nvidia.com/v1"
                completion_kwargs["api_key"] = os.getenv("NVIDIA_API_KEY")

            if "z-ai/glm4.7" in model:
                completion_kwargs["temperature"] = 0.7
                completion_kwargs["top_p"] = 0.95
                completion_kwargs["max_tokens"] = 16384
                completion_kwargs["response_format"] = {"type": "json_object"}
                # Disable thinking mode for deterministic, fast responses
                completion_kwargs["extra_body"] = {
                    "chat_template_kwargs": {
                        "enable_thinking": False,
                        "clear_thinking": False,
                    }
                }
            elif "minimaxai/minimax-m2.7" in model:
                completion_kwargs["temperature"] = 0.7
                completion_kwargs["top_p"] = 0.95
                completion_kwargs["max_tokens"] = 16384
            elif (
                "deepseek-ai/deepseek-v4-pro" in model
                or "deepseek-ai/deepseek-v4-flash" in model
            ):
                completion_kwargs["temperature"] = 0.7
                completion_kwargs["top_p"] = 0.95
                completion_kwargs["max_tokens"] = 16384
                completion_kwargs["extra_body"] = {
                    "chat_template_kwargs": {
                        "thinking": False
                    }
                }
            if "osmapi.com" in os.environ.get("OPENAI_API_BASE", ""):
                completion_kwargs["response_format"] = {"type": "json_object"}

            response = litellm.completion(**completion_kwargs)

            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            content = response.choices[0].message.content
            if content is None:
                content = getattr(response.choices[0].message, "reasoning_content", None) or ""
            content = content.strip()
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
        QueryType.FACT_LOOKUP: (
            "Tone hint: Open with 1-2 sentences directly answering the exact question asked. "
            "Do NOT use analogies, metaphors, or real-world comparisons. "
            "Use only information present in the provided context. "
            "Cite the section number at the end of each bullet point."
        ),
        QueryType.PENALTY_LOOKUP: "Tone hint: Lead with the consequence, then explain why it applies.",
        QueryType.CROSS_REFERENCE: (
            "Tone hint: Explain what the cited section says first. "
            "Then describe how it connects to related sections shown in the context. "
            "Only reference sections and jurisdictions that are present in the provided context blocks."
        ),
        QueryType.CONFLICT_DETECTION: (
            "Tone hint: Explicitly flag the contradiction ONLY if BOTH sides appear in the "
            "provided context. If only one jurisdiction's rule is present, describe what you "
            "found and note the missing side. Never infer precedence from legal reasoning — "
            "only state precedence if the context explicitly says so."
        ),
        QueryType.TEMPORAL: (
            "Tone hint: Lead with the specific time period or deadline. "
            "Quote the exact numeric value from context (e.g. '30 days', '3 months'). "
            "If the context does not contain a specific time value, say so explicitly."
        ),
    }
    return tone_hints.get(query_type, tone_hints[QueryType.FACT_LOOKUP])


def _extract_json_dict(raw: str) -> dict:
    cleaned = raw.strip()

    # Strip <think>...</think> blocks (Qwen3, DeepSeek, Nemotron reasoning traces)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()

    # Strip markdown code fences (handles leading spaces, ```json, ``` variants)
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE).strip()
    cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE).strip()

    decoder = json.JSONDecoder()

    # Try direct parse first
    try:
        parsed = decoder.decode(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Scan for first { and raw_decode from there
    for idx, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(cleaned[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise json.JSONDecodeError("No JSON object found", cleaned, 0)


def _pinned_section_specs(
    refs: list[str] | None,
    jurisdiction: Jurisdiction | None,
) -> list[tuple[str, Jurisdiction]]:
    """
    Convert eval/document refs such as "Section 6" and "Rule 7" to lookup specs.
    Central Act sections live under CENTRAL; state rules live under the target state.
    """
    specs: list[tuple[str, Jurisdiction]] = []
    target_jurisdiction = jurisdiction or Jurisdiction.CENTRAL
    for ref in refs or []:
        match = _PINNED_SECTION_RE.match(str(ref))
        if not match:
            continue
        kind = (match.group(1) or "").lower()
        section_id = match.group(2)
        lookup_jurisdiction = (
            Jurisdiction.CENTRAL
            if kind in {"section", "sec.", "sec", "s.", "s"}
            else target_jurisdiction
        )
        specs.append((section_id, lookup_jurisdiction))
    return specs


async def _fetch_pinned_sections(
    refs: list[str] | None,
    jurisdiction: Jurisdiction | None,
    existing: list[RetrievedChunk],
    hint: str = "",
) -> list[RetrievedChunk]:
    specs = _pinned_section_specs(refs, jurisdiction)
    if not specs:
        return []

    from civicsetu.stores.relational_store import AsyncSessionLocal

    seen = {str(c.chunk.chunk_id) for c in existing}
    pinned: list[RetrievedChunk] = []
    async with AsyncSessionLocal() as session:
        for section_id, lookup_jurisdiction in specs:
            family = await VectorStore.get_section_family(
                session=session,
                section_id=section_id,
                jurisdiction=lookup_jurisdiction,
            )
            for fc in _sort_pinned_family(family, hint)[:6]:
                cid = str(fc.chunk.chunk_id)
                if cid in seen:
                    continue
                seen.add(cid)
                fc.is_pinned = True
                fc.retrieval_source = "pinned"
                pinned.append(fc)
    return pinned


def _sort_pinned_family(family: list[RetrievedChunk], hint: str) -> list[RetrievedChunk]:
    if not hint:
        return list(family)
    terms = {
        t.lower()
        for t in re.findall(r"[A-Za-z][A-Za-z0-9]{3,}", hint)
        if t.lower() not in _PIN_HINT_STOPWORDS
    }
    expanded_terms = set(terms)
    for term in terms:
        expanded_terms.update(_PIN_HINT_EXPANSIONS.get(term, set()))
    terms = expanded_terms
    if not terms:
        return list(family)

    def score(chunk: RetrievedChunk) -> tuple[int, int]:
        haystack = f"{chunk.chunk.section_title} {chunk.chunk.text}".lower()
        matches = sum(1 for term in terms if term in haystack)
        is_base = 0 if "(" in str(chunk.chunk.section_id) else 1
        return matches, is_base

    return sorted(family, key=score, reverse=True)


def _prepend_pinned_sections(state: CivicSetuState, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    pinned_refs = state.get("pinned_section_refs")
    if not pinned_refs:
        return chunks

    jurisdiction = state.get("pinned_section_jurisdiction") or state.get("jurisdiction_filter")
    hint = state.get("pinned_section_hint", "")
    specs = _pinned_section_specs(pinned_refs, jurisdiction)
    existing_matches: list[RetrievedChunk] = []
    for section_id, lookup_jurisdiction in specs:
        family_matches = [
            c for c in chunks
            if c.chunk.jurisdiction == lookup_jurisdiction
            and (
                str(c.chunk.section_id) == section_id
                or str(c.chunk.section_id).startswith(f"{section_id}(")
            )
        ]
        for c in _sort_pinned_family(family_matches, hint)[:6]:
            if c not in existing_matches:
                c.is_pinned = True
                existing_matches.append(c)

    pinned = asyncio.run(_fetch_pinned_sections(
        pinned_refs,
        jurisdiction,
        chunks,
        hint,
    ))
    promoted = existing_matches + pinned
    if promoted:
        promoted_ids = {str(c.chunk.chunk_id) for c in promoted}
        rest = [c for c in chunks if str(c.chunk.chunk_id) not in promoted_ids]
        log.info("pinned_section_refs_added", refs=pinned_refs, pinned=len(promoted))
        return promoted + rest
    if pinned:
        log.info("pinned_section_refs_added", refs=pinned_refs, pinned=len(pinned))
        return pinned + chunks
    log.warning("pinned_section_refs_missing", refs=pinned_refs)
    return chunks


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
        raw = _llm_call(prompt, system, tier="fast")
        result = _extract_json_dict(raw)

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


# ── Node 2: Vector Retriever (hybrid: vector + FTS via RRF) ───────────────────

async def _rrf_retrieve(
    query: str,
    query_embedding: list[float],
    top_k: int,
    jurisdiction: str | None,
) -> list[RetrievedChunk]:
    """Shim: delegates to VectorRetriever for backward compatibility with graph/hybrid nodes."""
    from civicsetu.retrieval.vector_retriever import VectorRetriever
    return await VectorRetriever.retrieve(query, query_embedding, top_k, jurisdiction)


def vector_retrieval_node(state: CivicSetuState) -> dict:
    from civicsetu.retrieval.vector_retriever import VectorRetriever

    query = state.get("rewritten_query") or state["query"]
    top_k = state.get("top_k", 5)
    jurisdiction = state.get("jurisdiction_filter")
    node_start = time.perf_counter()

    log.info("vector_retrieval_node", query=query[:80], top_k=top_k)

    embed_start = time.perf_counter()
    query_embedding = cached_embed(query)
    log.info("stage_timing", node="vector_retrieval", stage="embedding",
             duration_ms=round((time.perf_counter() - embed_start) * 1000, 2))

    retrieve_start = time.perf_counter()
    chunks = asyncio.run(VectorRetriever.retrieve(query, query_embedding, top_k, jurisdiction))
    log.info("stage_timing", node="vector_retrieval", stage="postgres_retrieval",
             duration_ms=round((time.perf_counter() - retrieve_start) * 1000, 2))
    # Fix 4: section-ID-aware direct lookup.
    # Use original query when it has explicit sections, rewritten otherwise.
    _orig = state.get("query", query)
    _lookup_q = _orig if VectorRetriever._extract_query_section_ids(_orig) else query
    section_ids = VectorRetriever._extract_query_section_ids(_lookup_q)
    if section_ids:
        from civicsetu.models.enums import Jurisdiction as JurEnum
        from civicsetu.stores.relational_store import AsyncSessionLocal

        async def _fetch_sections():
            seen = {str(c.chunk.chunk_id) for c in chunks}
            extra: list[RetrievedChunk] = []
            async with AsyncSessionLocal() as session:
                for sid in section_ids:
                    jur = JurEnum(jurisdiction) if jurisdiction else None
                    family = await VectorStore.get_section_family(
                        session=session,
                        section_id=sid,
                        jurisdiction=jur or JurEnum.CENTRAL,
                    )
                    # Central Act sections (e.g. Section 7, 60) don't exist in state DB —
                    # fall back to CENTRAL so penalty/fact queries still find them.
                    if not family and jur and jur != JurEnum.CENTRAL:
                        family = await VectorStore.get_section_family(
                            session=session,
                            section_id=sid,
                            jurisdiction=JurEnum.CENTRAL,
                        )
                    for fc in family[:3]:
                        cid = str(fc.chunk.chunk_id)
                        if cid not in seen:
                            seen.add(cid)
                            fc.is_pinned = True
                            extra.append(fc)
            return extra

        direct_chunks = asyncio.run(_fetch_sections())
        if direct_chunks:
            log.info("vector_section_direct_lookup",
                     section_ids=list(section_ids),
                     direct_chunks=len(direct_chunks))
            chunks = direct_chunks + chunks

    # For state jurisdictions, supplement with Central Act retrieval.
    # State DBs lack Central Act sections (e.g. Section 7 revocation grounds,
    # Section 5 timeline) — adding them ensures cross-Act context is available.
    if jurisdiction and jurisdiction != Jurisdiction.CENTRAL:
        central_start = time.perf_counter()
        central_chunks = asyncio.run(VectorRetriever.retrieve(query, query_embedding, top_k, Jurisdiction.CENTRAL))
        seen = {str(c.chunk.chunk_id) for c in chunks}
        extra_central = [c for c in central_chunks if str(c.chunk.chunk_id) not in seen]
        if extra_central:
            log.info("vector_central_supplement",
                     jurisdiction=jurisdiction, count=len(extra_central))
            chunks = chunks + extra_central
        log.info("stage_timing", node="vector_retrieval", stage="central_supplement",
                 duration_ms=round((time.perf_counter() - central_start) * 1000, 2))

    chunks = _prepend_pinned_sections(state, chunks)
    log.info("node_timing", node="vector_retrieval",
             duration_ms=round((time.perf_counter() - node_start) * 1000, 2), results=len(chunks))
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
    # run RRF hybrid retrieval instead of pure vector
    if not chunks:
        log.info("graph_retrieval_fallback_to_rrf", query=query[:80])
        embed_start = time.perf_counter()
        query_embedding = cached_embed(query)
        log.info("stage_timing", node="graph_retrieval", stage="fallback_embedding", duration_ms=round((time.perf_counter() - embed_start) * 1000, 2))

        fallback_start = time.perf_counter()
        chunks = asyncio.run(_rrf_retrieve(query, query_embedding, top_k, jurisdiction))
        log.info("stage_timing", node="graph_retrieval", stage="fallback_rrf_search", duration_ms=round((time.perf_counter() - fallback_start) * 1000, 2))
        log.info("graph_fallback_rrf_results", count=len(chunks))

        # Second fallback: state jurisdiction had no indexed penalty/XREF sections —
        # retry without jurisdiction filter to pick up Central Act chunks.
        if not chunks and jurisdiction:
            log.info("graph_retrieval_fallback_no_jurisdiction", query=query[:80])
            chunks = asyncio.run(_rrf_retrieve(query, query_embedding, top_k, None))
            log.info("graph_fallback_no_jurisdiction_results", count=len(chunks))
    # Fix 5: section-ID-aware direct lookup in graph retrieval path.
    # Use original query when it has explicit sections (XREF: "Section 18 refund")
    # so classifier-injected cross-refs (e.g. "section 19 allottee rights") don't
    # pin wrong sections. Fall back to rewritten query when original has none
    # (temporal: classifier injects "section 5" via RULE 1 for retrieval).
    from civicsetu.retrieval.vector_retriever import VectorRetriever as _VR
    _original_query = state.get("query", query)
    _lookup_query = _original_query if _VR._extract_query_section_ids(_original_query) else query
    section_ids = _VR._extract_query_section_ids(_lookup_query)
    if section_ids:
        from civicsetu.models.enums import Jurisdiction as JurEnum
        from civicsetu.stores.relational_store import AsyncSessionLocal

        async def _fetch_sections_graph():
            seen = {str(c.chunk.chunk_id) for c in chunks}
            extra: list[RetrievedChunk] = []
            async with AsyncSessionLocal() as session:
                for sid in section_ids:
                    jur = JurEnum(jurisdiction) if jurisdiction else JurEnum.CENTRAL
                    family = await VectorStore.get_section_family(
                        session=session,
                        section_id=sid,
                        jurisdiction=jur,
                    )
                    # Central Act sections (e.g. Section 59, 60) don't exist in state DB —
                    # fall back to CENTRAL so penalty queries still find them.
                    if not family and jur != JurEnum.CENTRAL:
                        family = await VectorStore.get_section_family(
                            session=session,
                            section_id=sid,
                            jurisdiction=JurEnum.CENTRAL,
                        )
                    for fc in family[:3]:
                        cid = str(fc.chunk.chunk_id)
                        if cid not in seen:
                            seen.add(cid)
                            fc.is_pinned = True
                            extra.append(fc)
            return extra

        direct_chunks = asyncio.run(_fetch_sections_graph())
        if direct_chunks:
            log.info("graph_section_direct_lookup",
                     section_ids=list(section_ids),
                     direct_chunks=len(direct_chunks))
            chunks = direct_chunks + chunks

    # For state jurisdictions, supplement with Central Act retrieval.
    # State DBs surface state-rule chunks (e.g. Karnataka Rule 5 withdrawal) when
    # the query needs Central Act sections (e.g. Section 5 thirty-day deadline,
    # Section 4 account maintenance) — CENTRAL fallback in _fetch_sections_graph
    # only fires when the state family is EMPTY, missing this mismatch case.
    if jurisdiction and jurisdiction != Jurisdiction.CENTRAL:
        central_supplement_start = time.perf_counter()
        _query_embedding = cached_embed(query)
        central_chunks = asyncio.run(_VR.retrieve(query, _query_embedding, top_k, Jurisdiction.CENTRAL))
        seen = {str(c.chunk.chunk_id) for c in chunks}
        extra_central = [c for c in central_chunks if str(c.chunk.chunk_id) not in seen]
        if extra_central:
            log.info("graph_central_supplement", jurisdiction=str(jurisdiction), count=len(extra_central))
            chunks = chunks + extra_central
        log.info("stage_timing", node="graph_retrieval", stage="central_supplement",
                 duration_ms=round((time.perf_counter() - central_supplement_start) * 1000, 2))

    chunks = _prepend_pinned_sections(state, chunks)
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
    from civicsetu.retrieval.reranker import Reranker

    chunks = state.get("retrieved_chunks", [])
    query = state.get("rewritten_query") or state["query"]
    node_start = time.perf_counter()

    if not chunks:
        log.info("node_timing", node="reranker",
                 duration_ms=round((time.perf_counter() - node_start) * 1000, 2), reranked=0)
        return {"reranked_chunks": []}

    reranked = Reranker.rerank(chunks, query)

    log.info("reranker_complete", reranked=len(reranked))
    log.info("node_timing", node="reranker",
             duration_ms=round((time.perf_counter() - node_start) * 1000, 2), reranked=len(reranked))
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
        "Use simple language and bullet points to make rules clear. "
        "Every factual claim in your answer must be traceable to the numbered context blocks provided. "
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

    raw = ""
    try:
        llm_start = time.perf_counter()
        log.info("generator_debug_start", query=query[:50], chunks=len(chunks), model=THINKING_MODELS[0])
        raw = _llm_call(prompt, system, temperature=0.0, tier="thinking")
        log.info("generator_debug_raw", raw_type=type(raw).__name__, raw_len=len(raw), raw_preview=raw[:200])
        log.info("stage_timing", node="generator", stage="llm", duration_ms=round((time.perf_counter() - llm_start) * 1000, 2))
        result = _extract_json_dict(raw)

        answer = result.get("answer") or result.get("response") or result.get("text") or ""
        try:
            confidence = float(result.get("confidence_score") or 0.5)
        except (TypeError, ValueError):
            confidence = 0.5
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
        log.warning("generator_parse_failed", error=str(e), raw_preview=raw[:500])
        salvage_answer = raw.strip()
        # Salvage: if raw has content but JSON parse failed, return raw text as answer
        if salvage_answer:
            seen: set[tuple[str, str]] = set()
            citations = []
            for rc in chunks:
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
            answer = salvage_answer
            confidence = 0.3
            conflict_warnings = []
            amendment_notice = None
            log.warning("generator_parse_salvaged", citations=len(citations), answer_chars=len(answer))
        else:
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
        tasks = [
            _rrf_retrieve(query, query_embedding, top_k, jurisdiction),
            GraphRetriever.retrieve(query=query, jurisdiction=jurisdiction, depth=2),
        ]
        # For state jurisdictions, retrieve Central Act in parallel.
        # CONF queries compare state rules against Central Act — both sides needed.
        if jurisdiction and jurisdiction != Jurisdiction.CENTRAL:
            tasks.append(_rrf_retrieve(query, query_embedding, top_k, Jurisdiction.CENTRAL))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        v_chunks = results[0] if not isinstance(results[0], Exception) else []
        g_chunks = results[1] if not isinstance(results[1], Exception) else []
        c_chunks = results[2] if len(results) > 2 and not isinstance(results[2], Exception) else []

        if isinstance(results[0], Exception):
            log.warning("hybrid_vector_failed", error=str(results[0]))
        if isinstance(results[1], Exception):
            log.warning("hybrid_graph_failed", error=str(results[1]))

        # Graph traversal crosses all jurisdictions via DERIVED_FROM edges.
        # Filter to keep only target jurisdiction + Central to prevent
        # contamination from unrelated state rules (MH, UP, TN chunks in a KA query).
        if jurisdiction:
            g_filtered = [
                c for c in g_chunks
                if c.chunk.jurisdiction in (jurisdiction, Jurisdiction.CENTRAL)
            ]
            g_chunks = g_filtered if g_filtered else g_chunks

        # Dedup Central supplement against already-collected chunks
        seen = {str(c.chunk.chunk_id) for c in v_chunks + g_chunks}
        extra_central = [c for c in c_chunks if str(c.chunk.chunk_id) not in seen]

        return v_chunks, g_chunks, extra_central

    retrieve_start = time.perf_counter()
    v_chunks, g_chunks, extra_central = asyncio.run(_retrieve())
    log.info("stage_timing", node="hybrid_retrieval", stage="vector_graph_parallel", duration_ms=round((time.perf_counter() - retrieve_start) * 1000, 2))

    all_chunks = v_chunks + g_chunks + extra_central
    all_chunks = _prepend_pinned_sections(state, all_chunks)
    log.info(
        "hybrid_retrieval_complete",
        vector_chunks=len(v_chunks),
        graph_chunks=len(g_chunks),
        central_supplement=len(extra_central),
        total=len(all_chunks),
    )
    log.info("node_timing", node="hybrid_retrieval", duration_ms=round((time.perf_counter() - node_start) * 1000, 2), total=len(all_chunks))
    return {"retrieved_chunks": all_chunks}
