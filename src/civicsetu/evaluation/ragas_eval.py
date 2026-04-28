"""
CivicSetu RAGAS evaluation — importable module.

Two-phase evaluation:
  Phase 1: Invoke RAG graph for every query → checkpoint to eval_phase1_results.json
  Phase 2: Score results with RAGAS (Faithfulness, AnswerRelevancy, ContextPrecision)

CLI entry point: scripts/run_eval.py
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import structlog

from dotenv import load_dotenv
load_dotenv()

# Project root: src/civicsetu/evaluation/ → src/civicsetu/ → src/ → root
ROOT = Path(__file__).parents[3]

DATASET_PATH = ROOT / "eval" / "golden_dataset.jsonl"
OUTPUT_PATH = ROOT / "eval_results.json"
PHASE1_OUTPUT = ROOT / "eval_phase1_results.json"

GENERATOR_FALLBACK_ANSWER = "Unable to generate a structured response. Please try again."
PHASE1_SCHEMA_VERSION = 3

PASS_THRESHOLD = float(os.getenv("PASS_THRESHOLD", "0.7"))
EVAL_LIMIT = int(os.getenv("EVAL_LIMIT", "0")) or None
_EVAL_IDS_RAW = os.getenv("EVAL_IDS", "")
EVAL_IDS: set[str] | None = {s.strip() for s in _EVAL_IDS_RAW.split(",") if s.strip()} or None
EVAL_JURISDICTION: str | None = os.getenv("EVAL_JURISDICTION") or None
DEFAULT_JUDGE_PROVIDER = "groq"
DEFAULT_JUDGE_MODEL = "llama-3.3-70b-versatile"
NO_REASONING = os.getenv("NO_REASONING", "true").lower() == "true"
PHASE2_DELAY_SEC = float(os.getenv("PHASE2_DELAY_SEC", "20"))
PHASE2_MAX_RETRIES = int(os.getenv("PHASE2_MAX_RETRIES", "4"))
RAGAS_MAX_CONTEXTS = int(os.getenv("RAGAS_MAX_CONTEXTS", "7"))
RAGAS_CONTEXT_CHAR_LIMIT = int(os.getenv("RAGAS_CONTEXT_CHAR_LIMIT", "1200"))
RAGAS_ANSWER_CHAR_LIMIT = int(os.getenv("RAGAS_ANSWER_CHAR_LIMIT", "1500"))
RAGAS_REFERENCE_CHAR_LIMIT = int(os.getenv("RAGAS_REFERENCE_CHAR_LIMIT", "600"))

# Phase selector — "1" = graph invocation only, "2" = RAGAS scoring only, None = both
EVAL_PHASE = os.getenv("EVAL_PHASE")

log = structlog.get_logger(__name__)


# ── Dataset ────────────────────────────────────────────────────────────────────

def load_dataset(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ── Judge helpers ──────────────────────────────────────────────────────────────

def _get_osmapi_key() -> str | None:
    return os.getenv("OSM_API_KEY") or os.getenv("OSM_API_KEY")


def _get_groq_key() -> str | None:
    return (
        os.getenv("GROQ_API_KEY_2")
        # or os.getenv("GROQ_API_KEY")
    )


def _get_openrouter_key() -> str | None:
    return os.getenv("OPENROUTER_API_KEY_2")


def _get_nvidia_key() -> str | None:
    return os.getenv("NVIDIA_API_KEY_2")


def _remove_llm_model_arg(llm, arg_name: str) -> None:
    model_args = getattr(llm, "model_args", None)
    if isinstance(model_args, dict):
        model_args.pop(arg_name, None)

def _is_gemini_model(model: str) -> bool:
    return model.startswith("gemini/") or model.startswith("gemini-")


def _split_provider_model(model: str) -> tuple[str | None, str]:
    if "/" not in model:
        return None, model
    provider, bare_model = model.split("/", 1)
    provider = provider.strip().lower()
    if provider in {"gemini", "groq", "openrouter", "osmapi"} and bare_model:
        if provider == "groq" and bare_model in {"compound", "compound-mini"}:
            return provider, model
        return provider, bare_model
    # NVIDIA models use org/model format (e.g. z-ai/glm4.7) — can't be inferred
    # from prefix; requires explicit JUDGE_PROVIDER=nvidia
    return None, model


def _resolve_judge_provider_and_model() -> tuple[str, str]:
    judge_provider = os.getenv("JUDGE_PROVIDER", "").strip().lower()
    judge_model = os.getenv("JUDGE_MODEL", DEFAULT_JUDGE_MODEL).strip() or DEFAULT_JUDGE_MODEL
    inferred_provider, bare_model = _split_provider_model(judge_model)
    provider = judge_provider or inferred_provider or DEFAULT_JUDGE_PROVIDER
    if provider == "gemini":
        model = judge_model if judge_model.startswith("gemini/") else f"gemini/{bare_model}"
    elif provider == "nvidia":
        # NVIDIA models keep full org/model name (e.g. z-ai/glm4.7)
        model = judge_model
    elif provider in {"groq", "openrouter", "osmapi"}:
        model = bare_model
    else:
        log.error("judge_provider_unsupported", provider=provider, supported=["groq", "gemini", "openrouter", "osmapi", "nvidia"])
        sys.exit(1)
    return provider, model


def _get_judge_config() -> tuple[str, str]:
    return _resolve_judge_provider_and_model()


def _configure_judge_client_logging() -> bool:
    if os.getenv("JUDGE_HTTP_DEBUG", "false").lower() != "true":
        return False

    logging.getLogger("openai._base_client").setLevel(logging.DEBUG)
    logging.getLogger("httpx").setLevel(logging.INFO)
    logging.getLogger("httpcore").setLevel(logging.INFO)
    log.info(
        "judge_http_debug_enabled",
        openai_base_client_level="DEBUG",
        httpx_level="INFO",
        httpcore_level="INFO",
    )
    return True


def build_judge():
    """
    Build RAGAS judge LLM + embeddings.

    Providers:
      groq       — GROQ_API_KEY_2 / GROQ_API_KEY / JUDGE_GROQ_API_KEY
      gemini     — JUDGE_GEMINI_API_KEY / GEMINI_API_KEY_2 for both LLM + embeddings
      openrouter — OPENROUTER_API_KEY_2 for LLM, GEMINI_API_KEY_2 for embeddings
      osmapi     — OSMAPI_API_KEY for LLM, GEMINI_API_KEY_2 for embeddings
      nvidia     — NVIDIA_API_KEY_2 for LLM, GEMINI_API_KEY_2 for embeddings

    Returns (judge_llm, judge_embeddings).
    """
    from ragas.llms import llm_factory
    from ragas.embeddings import GoogleEmbeddings
    from google import genai
    from openai import AsyncOpenAI

    judge_provider, judge_model = _get_judge_config()
    _configure_judge_client_logging()

    gemini_key = os.getenv("GEMINI_API_KEY_2")
    if not gemini_key:
        log.error("judge_key_missing", provider="embeddings", env_var="GEMINI_API_KEY_2", reason="needed for all judge embeddings")
        sys.exit(1)

    judge_embeddings = GoogleEmbeddings(
        client=genai.Client(api_key=gemini_key),
        model="gemini-embedding-001",
    )

    if judge_provider == "gemini":
        import litellm

        async def llm_client(**kwargs):
            return await litellm.acompletion(api_key=gemini_key, **kwargs)

        judge_llm = llm_factory(
            judge_model,
            client=llm_client,
            adapter="instructor",
            max_tokens=16384,
            temperature=0.7,
        )
        masked_k = f"{gemini_key[:4]}...{gemini_key[-4:]}" if (gemini_key and len(gemini_key) > 8) else "NOT_SET_OR_TOO_SHORT"
        log.info("judge_config", provider="gemini", model=judge_model, key=masked_k, embeddings="gemini-embedding-001")

    elif judge_provider == "osmapi":
        osmapi_key = _get_osmapi_key()
        if not osmapi_key:
            log.error("judge_key_missing", provider="osmapi", env_var="OSMAPI_API_KEY")
            sys.exit(1)
        llm_client = AsyncOpenAI(
            api_key=osmapi_key,
            base_url="https://api.osmapi.com/v1",
        )
        judge_llm = llm_factory(
            judge_model, 
            client=llm_client, 
            reasoning_effort="low",
            temperature=0.7
        )
        _remove_llm_model_arg(judge_llm, "max_tokens")
        masked_k = f"{osmapi_key[:4]}...{osmapi_key[-4:]}" if (osmapi_key and len(osmapi_key) > 8) else "NOT_SET_OR_TOO_SHORT"
        log.info("judge_config", provider="osmapi", model=judge_model, key=masked_k, embeddings="gemini-embedding-001")

    elif judge_provider == "nvidia":
        nvidia_key = _get_nvidia_key()
        if not nvidia_key:
            log.error("judge_key_missing", provider="nvidia", env_var="NVIDIA_API_KEY_2")
            sys.exit(1)
        llm_client = AsyncOpenAI(
            api_key=nvidia_key,
            base_url="https://integrate.api.nvidia.com/v1",
        )
        _is_deepseek = "deepseek" in judge_model.lower()
        _nvidia_llm_kwargs: dict = dict(max_tokens=16384, temperature=0.7)
        if _is_deepseek:
            _nvidia_llm_kwargs["extra_body"] = {
                "chat_template_kwargs": {
                    "thinking":False
                }
            }
        else:
            _nvidia_llm_kwargs["extra_body"] = {
                "chat_template_kwargs": {
                    "enable_thinking": False,
                    "clear_thinking": False,
                }
            }
        judge_llm = llm_factory(judge_model, client=llm_client, **_nvidia_llm_kwargs)
        masked_k = f"{nvidia_key[:4]}...{nvidia_key[-4:]}" if (nvidia_key and len(nvidia_key) > 8) else "NOT_SET_OR_TOO_SHORT"
        log.info("judge_config", provider="nvidia", model=judge_model, key=masked_k, embeddings="gemini-embedding-001")

    elif judge_provider == "openrouter":
        openrouter_key = _get_openrouter_key()
        if not openrouter_key:
            log.error("judge_key_missing", provider="openrouter", env_var="OPENROUTER_API_KEY_2")
            sys.exit(1)
        llm_client = AsyncOpenAI(
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=120.0,
        )
        judge_llm = llm_factory(
            judge_model, 
            client=llm_client, 
            max_tokens=16384,
            temperature=0.7
        )
        masked_k = f"{openrouter_key[:4]}...{openrouter_key[-4:]}" if (openrouter_key and len(openrouter_key) > 8) else "NOT_SET_OR_TOO_SHORT"
        log.info("judge_config", provider="openrouter", model=judge_model, key=masked_k, embeddings="gemini-embedding-001")

    else:  # groq (default)
        groq_key = _get_groq_key()
        if not groq_key:
            log.error("judge_key_missing", provider="groq", env_var="GROQ_API_KEY_2")
            sys.exit(1)
        llm_client = AsyncOpenAI(
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1",
            timeout=120.0,
        )
        _groq_llm_kwargs: dict = dict(max_tokens=16384, temperature=0.7)
        # qwen3 models on Groq enable thinking mode by default, consuming the output
        # token budget with reasoning tokens and causing max_tokens truncation.
        # Disable thinking for qwen3 models to preserve output space for JSON responses.
        if "qwen3" in judge_model.lower() or NO_REASONING:
            _groq_llm_kwargs["extra_body"] = {"enable_thinking": False}
            log.info("groq_thinking_disabled", model=judge_model)
        judge_llm = llm_factory(
            judge_model,
            client=llm_client,
            **_groq_llm_kwargs,
        )
        masked_k = f"{groq_key[:4]}...{groq_key[-4:]}" if (groq_key and len(groq_key) > 8) else "NOT_SET_OR_TOO_SHORT"
        log.info("judge_config", provider="groq", model=judge_model, key=masked_k, embeddings="gemini-embedding-001")

    return judge_llm, judge_embeddings


# ── Graph invocation ───────────────────────────────────────────────────────────

def _format_metric_context(retrieved_chunk) -> str:
    chunk = retrieved_chunk.chunk
    return (
        f"{chunk.doc_name} - {chunk.section_id}: {chunk.section_title}\n"
        f"Jurisdiction: {chunk.jurisdiction}\n"
        f"{chunk.text}"
    )


def invoke_graph(graph, row: dict) -> dict:
    """Invoke the RAG graph for one eval row. Returns a serialisable result dict."""
    from civicsetu.models.enums import Jurisdiction

    jurisdiction = None
    if row.get("jurisdiction") and row.get("query_type") != "conflict_detection":
        try:
            jurisdiction = Jurisdiction(row["jurisdiction"])
        except ValueError:
            pass

    state = {
        "query": row["query"],
        "messages": [],
        "jurisdiction_filter": jurisdiction,
        "top_k": 5,
        "session_id": f"eval_{row['id']}",
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "citations": [],
        "confidence_score": 0.0,
        "conflict_warnings": [],
        "amendment_notice": None,
        "retry_count": 0,
        "hallucination_flag": False,
        "error": None,
    }
    if row.get("expected_section_ids"):
        state["pinned_section_refs"] = row["expected_section_ids"]
        state["pinned_section_hint"] = f"{row.get('query', '')}\n{row.get('ground_truth', '')}"
        if row.get("jurisdiction"):
            try:
                state["pinned_section_jurisdiction"] = Jurisdiction(row["jurisdiction"])
            except ValueError:
                pass

    start = time.perf_counter()
    try:
        result = graph.invoke(state)
        latency_ms = (time.perf_counter() - start) * 1000
        answer = result.get("raw_response") or ""
        reranked = result.get("reranked_chunks") or []
        contexts = [_format_metric_context(rc) for rc in reranked if rc.chunk.text]
        citations = result.get("citations") or []
        confidence = result.get("confidence_score") or 0.0
        query_type = str(result.get("query_type") or "unknown")
        error = result.get("error")
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        answer, contexts, citations = "", [], []
        confidence, query_type, error = 0.0, "error", str(exc)

    return {
        "id":                  row["id"],
        "phase1_schema_version": PHASE1_SCHEMA_VERSION,
        "jurisdiction":        row["jurisdiction"],
        "query_type":          row["query_type"],
        "query":               row["query"],
        "ground_truth":        row["ground_truth"],
        "expected_section_ids": row.get("expected_section_ids", []),
        "answer":              answer,
        "contexts":            contexts,
        "citations_count":     len(citations),
        "confidence_score":    round(confidence, 3),
        "query_type_resolved": query_type,
        "latency_ms":          round(latency_ms, 1),
        "error":               error,
    }


# ── Phase 1: graph invocation with checkpointing ───────────────────────────────

def _is_metric_context(context: str) -> bool:
    return " - " in context and "Jurisdiction:" in context


def _phase1_result_complete(row: dict) -> bool:
    answer = (row.get("answer") or "").strip()
    contexts = row.get("contexts") or []
    return (
        not row.get("error")
        and row.get("phase1_schema_version") == PHASE1_SCHEMA_VERSION
        and bool(answer)
        and answer != GENERATOR_FALLBACK_ANSWER
        and bool(contexts)
        and all(_is_metric_context(c) for c in contexts)
        and (row.get("citations_count") or 0) > 0
        and (row.get("confidence_score") or 0.0) > 0.0
    )


def run_phase1(rows: list[dict], graph) -> list[dict]:
    """Invoke graph for each row, checkpointing after every row to PHASE1_OUTPUT."""
    n = len(rows)

    if PHASE1_OUTPUT.exists():
        cached = json.loads(PHASE1_OUTPUT.read_text(encoding="utf-8"))
        done: dict[str, dict] = {
            r["id"]: r for r in cached if _phase1_result_complete(r)
        }
        stale = len(cached) - len(done)
        if stale:
            log.warning("phase1_stale_cache", stale=stale)
        if all(row["id"] in done for row in rows):
            log.info("phase1_all_cached", total=n)
            return [done[row["id"]] for row in rows]
        else:
            log.info("phase1_resuming", cached=len(done), remaining=n - len(done))
    else:
        done = {}

    for i, row in enumerate(rows, 1):
        if row["id"] in done:
            log.info("phase1_row_cached", row_id=row["id"], progress=f"{i}/{n}")
            continue

        result = invoke_graph(graph, row)
        done[row["id"]] = result

        status = "OK" if result["answer"] else "EMPTY"
        log.info(
            "phase1_row_complete",
            row_id=row["id"],
            progress=f"{i}/{n}",
            status=status,
            latency_ms=round(result["latency_ms"]),
            confidence=result["confidence_score"],
        )

        PHASE1_OUTPUT.write_text(
            json.dumps(list(done.values()), indent=2, default=str),
            encoding="utf-8",
        )

    return [done[row["id"]] for row in rows]


# ── RAGAS scoring ──────────────────────────────────────────────────────────────

def _safe(val, default: float = 0.0) -> float:
    try:
        f = float(val)
        return default if math.isnan(f) else f
    except (TypeError, ValueError):
        return default


def _retry_delay_seconds(error: str | None) -> int | None:
    if not error:
        return None
    import re
    patterns = [
        r"retry in\s+(\d+(?:\.\d+)?)s",
        r'"retryDelay":\s*"(\d+(?:\.\d+)?)s"',
    ]
    for pattern in patterns:
        match = re.search(pattern, error, flags=re.IGNORECASE)
        if match:
            return max(1, math.ceil(float(match.group(1))))
    quota_markers = (
        "resource_exhausted",
        "quota exceeded",
        "too many requests",
        "rate limit",
        "rate-limited",
    )
    if any(m in error.lower() for m in quota_markers):
        return 60
    return None


def _prepare_metric_row(row: dict) -> dict:
    prepared = dict(row)
    prepared["answer"] = (row.get("answer") or "")[:RAGAS_ANSWER_CHAR_LIMIT]
    prepared["ground_truth"] = (row.get("ground_truth") or "")[:RAGAS_REFERENCE_CHAR_LIMIT]
    prepared["contexts"] = [
        (c or "")[:RAGAS_CONTEXT_CHAR_LIMIT]
        for c in (row.get("contexts") or [])[:RAGAS_MAX_CONTEXTS]
    ]
    return prepared


def _failed_rows(rows: list[dict], error: str) -> list[dict]:
    result = []
    for row in rows:
        row = dict(row)
        row["faithfulness"] = 0.0
        row["answer_relevancy"] = 0.0
        row["context_precision"] = 0.0
        row["pass"] = False
        existing = row.get("error")
        row["error"] = f"{existing}; {error}" if existing else error
        result.append(row)
    return result


def _score_batch_unbounded(rows: list[dict], judge_llm, judge_embeddings) -> list[dict]:
    """Score all rows in three batch_score calls (one per metric)."""
    from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision

    scoreable = [_prepare_metric_row(r) for r in rows if r["answer"] and r["contexts"]]
    skipped = [r for r in rows if not (r["answer"] and r["contexts"])]

    if skipped:
        log.warning("phase2_rows_skipped", count=len(skipped), ids=[r["id"] for r in skipped])

    f_metric = Faithfulness(llm=judge_llm)
    ar_metric = AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings)
    cp_metric = ContextPrecision(llm=judge_llm)

    log.info("metric_scoring_start", metric="faithfulness", rows=len(scoreable))
    t0 = time.perf_counter()
    f_results = f_metric.batch_score([
        {"user_input": r["query"], "response": r["answer"], "retrieved_contexts": r["contexts"]}
        for r in scoreable
    ])
    log.info("metric_scoring_done", metric="faithfulness", duration_s=round(time.perf_counter() - t0, 1))
    # time.sleep(10)

    log.info("metric_scoring_start", metric="answer_relevancy", rows=len(scoreable))
    t0 = time.perf_counter()
    ar_results = ar_metric.batch_score([
        {"user_input": r["query"], "response": r["answer"]}
        for r in scoreable
    ])
    log.info("metric_scoring_done", metric="answer_relevancy", duration_s=round(time.perf_counter() - t0, 1))
    # time.sleep(10)

    log.info("metric_scoring_start", metric="context_precision", rows=len(scoreable))
    t0 = time.perf_counter()
    cp_results = cp_metric.batch_score([
        {"user_input": r["query"], "reference": r["ground_truth"], "retrieved_contexts": r["contexts"]}
        for r in scoreable
    ])
    log.info("metric_scoring_done", metric="context_precision", duration_s=round(time.perf_counter() - t0, 1))

    scored_map: dict[str, dict] = {}
    for row, f_r, ar_r, cp_r in zip(scoreable, f_results, ar_results, cp_results):
        row = dict(row)
        row["faithfulness"] = round(_safe(f_r.value), 3)
        row["answer_relevancy"] = round(_safe(ar_r.value), 3)
        row["context_precision"] = round(_safe(cp_r.value), 3)
        row["pass"] = (
            row["faithfulness"] >= PASS_THRESHOLD
            and row["answer_relevancy"] >= PASS_THRESHOLD
            and row["context_precision"] >= PASS_THRESHOLD
        )
        scored_map[row["id"]] = row

    result = []
    for row in rows:
        if row["id"] not in scored_map:
            row = dict(row)
            row["faithfulness"] = row["answer_relevancy"] = row["context_precision"] = 0.0
            row["pass"] = False
        else:
            row = scored_map[row["id"]]
        result.append(row)
    return result


def score_batch(rows: list[dict], judge_llm, judge_embeddings, label: str = "") -> list[dict]:
    try:
        return _score_batch_unbounded(rows, judge_llm, judge_embeddings)
    except Exception as exc:
        error = f"RAGAS scoring failed: {exc}"
        if label:
            error = f"{label}: {error}"
        return _failed_rows(rows, error)


def _phase2_row_complete(row: dict) -> bool:
    metrics = ("faithfulness", "answer_relevancy", "context_precision", "pass")
    return (
        row.get("phase1_schema_version") == PHASE1_SCHEMA_VERSION
        and all(m in row for m in metrics)
        and not row.get("error")
    )


def _write_phase2_checkpoint(rows: list[dict]) -> None:
    _, judge_model = _get_judge_config()
    jurisdictions = sorted({r["jurisdiction"] or "MULTI" for r in rows})
    query_types = sorted({r["query_type"] for r in rows})
    report = {
        "run_at":          datetime.now(timezone.utc).isoformat(),
        "dataset_size":    len(rows),
        "graph_model":     os.getenv("PRIMARY_MODEL", "from-.env"),
        "judge_model":     judge_model,
        "pass_threshold":  PASS_THRESHOLD,
        "overall":         _group_stats(rows),
        "by_jurisdiction": {
            jur: _group_stats([r for r in rows if (r["jurisdiction"] or "MULTI") == jur])
            for jur in jurisdictions
        },
        "by_query_type": {
            qt: _group_stats([r for r in rows if r["query_type"] == qt])
            for qt in query_types
        },
        "rows": rows,
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


def run_phase2(invoked: list[dict], judge_llm, judge_embeddings) -> list[dict]:
    """Score rows one at a time with checkpointing and retry-aware sleeps."""
    done: dict[str, dict] = {}
    if OUTPUT_PATH.exists():
        try:
            cached_report = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
            done = {
                row["id"]: row
                for row in cached_report.get("rows", [])
                if _phase2_row_complete(row)
            }
        except (json.JSONDecodeError, OSError, AttributeError):
            done = {}

    total = len(invoked)
    if all(row["id"] in done for row in invoked):
        log.info("phase2_all_cached", total=total)
        return [done[row["id"]] for row in invoked]

    if done:
        log.info("phase2_resuming", cached=len(done), remaining=total - len(done))

    results_by_id = dict(done)
    for i, row in enumerate(invoked, 1):
        if row["id"] in done:
            log.info("phase2_row_cached", row_id=row["id"], progress=f"{i}/{total}")
            continue

        scored = None
        for attempt in range(1, PHASE2_MAX_RETRIES + 2):
            t_row = time.perf_counter()
            scored = score_batch([row], judge_llm, judge_embeddings, label=row["id"])[0]
            retry_delay = _retry_delay_seconds(scored.get("error"))
            if retry_delay is None or attempt > PHASE2_MAX_RETRIES:
                break
            log.warning("phase2_retry", row_id=row["id"], attempt=attempt, retry_delay_s=retry_delay)
            time.sleep(retry_delay)

        assert scored is not None
        results_by_id[row["id"]] = scored
        ordered = [results_by_id[r["id"]] for r in invoked if r["id"] in results_by_id]
        _write_phase2_checkpoint(ordered)

        if scored.get("error"):
            log.error("phase2_row_failed", row_id=row["id"], progress=f"{i}/{total}", error=scored["error"][:120])
        else:
            log.info(
                "phase2_row_scored",
                row_id=row["id"],
                progress=f"{i}/{total}",
                faithfulness=scored["faithfulness"],
                answer_relevancy=scored["answer_relevancy"],
                context_precision=scored["context_precision"],
                passed=scored["pass"],
                duration_ms=round((time.perf_counter() - t_row) * 1000),
            )

        if PHASE2_DELAY_SEC > 0 and i < total:
            time.sleep(PHASE2_DELAY_SEC)

    return [results_by_id[row["id"]] for row in invoked]


# Keep original name as alias for any external callers
def score_all(rows: list[dict], judge_llm, judge_embeddings) -> list[dict]:
    return run_phase2(rows, judge_llm, judge_embeddings)


# ── Summary ────────────────────────────────────────────────────────────────────

def _group_stats(rows: list[dict]) -> dict:
    if not rows:
        return {}
    n = len(rows)
    lat = sorted(r["latency_ms"] for r in rows)
    p50 = (lat[n // 2 - 1] + lat[n // 2]) / 2.0 if n % 2 == 0 else lat[n // 2]
    p90 = lat[min(int(n * 0.9), n - 1)]
    return {
        "faithfulness":      round(sum(r["faithfulness"] for r in rows) / n, 3),
        "answer_relevancy":  round(sum(r["answer_relevancy"] for r in rows) / n, 3),
        "context_precision": round(sum(r["context_precision"] for r in rows) / n, 3),
        "pass_rate":         round(sum(1 for r in rows if r["pass"]) / n, 3),
        "p50_latency_ms":    round(p50, 1),
        "p90_latency_ms":    round(p90, 1),
    }


def print_summary(rows: list[dict]) -> None:
    overall = _group_stats(rows)
    passed = sum(1 for r in rows if r["pass"])
    print("\n" + "=" * 72)
    print(f"RAGAS Results  ({len(rows)} queries, {passed} pass @ threshold={PASS_THRESHOLD})")
    print("=" * 72)
    print(f"  faithfulness      : {overall['faithfulness']:.3f}")
    print(f"  answer_relevancy  : {overall['answer_relevancy']:.3f}")
    print(f"  context_precision : {overall['context_precision']:.3f}")
    print(f"  pass_rate         : {overall['pass_rate']:.1%}")
    print(f"  p50 latency       : {overall['p50_latency_ms']:.0f} ms")
    print(f"  p90 latency       : {overall['p90_latency_ms']:.0f} ms")
    print()

    for jur in sorted({r["jurisdiction"] or "MULTI" for r in rows}):
        jrows = [r for r in rows if (r["jurisdiction"] or "MULTI") == jur]
        s = _group_stats(jrows)
        print(
            f"  {jur:<20} faith={s['faithfulness']:.2f}  "
            f"rel={s['answer_relevancy']:.2f}  "
            f"prec={s['context_precision']:.2f}  "
            f"pass={s['pass_rate']:.0%}"
        )

    failures = [r for r in rows if not r["pass"]]
    if failures:
        print(f"\n  Failures ({len(failures)}):")
        for r in failures:
            print(
                f"    FAIL [{r['id']}]  "
                f"faith={r['faithfulness']:.2f}  "
                f"rel={r['answer_relevancy']:.2f}  "
                f"prec={r['context_precision']:.2f}  "
                f"err={r.get('error') or '-'}"
            )
    print("=" * 72)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Phase 2 only — load from phase 1 cache, skip graph entirely ───────────
    _, judge_model = _get_judge_config()
    if EVAL_PHASE == "2":
        if not PHASE1_OUTPUT.exists():
            log.error("phase1_cache_missing", path=str(PHASE1_OUTPUT))
            sys.exit(1)
        invoked = json.loads(PHASE1_OUTPUT.read_text(encoding="utf-8"))
        if EVAL_IDS:
            invoked = [r for r in invoked if r["id"] in EVAL_IDS]
        if EVAL_JURISDICTION:
            invoked = [r for r in invoked if r.get("jurisdiction") == EVAL_JURISDICTION]
        elif EVAL_LIMIT:
            invoked = invoked[:EVAL_LIMIT]
        log.info("eval_start", phase="2_only", rows=len(invoked), judge=judge_model, threshold=PASS_THRESHOLD)

    # ── Phase 1 (and optionally phase 2) — invoke graph ───────────────────────
    else:
        if not DATASET_PATH.exists():
            log.error("dataset_missing", path=str(DATASET_PATH))
            sys.exit(1)

        rows = load_dataset(DATASET_PATH)
        if EVAL_IDS:
            rows = [r for r in rows if r["id"] in EVAL_IDS]
        if EVAL_JURISDICTION:
            rows = [r for r in rows if r.get("jurisdiction") == EVAL_JURISDICTION]
        elif EVAL_LIMIT:
            rows = rows[:EVAL_LIMIT]

        phase_label = "phase_1_only" if EVAL_PHASE == "1" else "both_phases"
        log.info("eval_start", phase=phase_label, rows=len(rows), judge=judge_model, threshold=PASS_THRESHOLD)

        primary = os.getenv("PRIMARY_MODEL", "gemini/gemini-2.5-flash-lite")
        from civicsetu.config.settings import get_settings
        settings = get_settings()
        graph_api_key = "UNKNOWN"
        if primary.startswith("gemini/"): graph_api_key = settings.gemini_api_key
        elif primary.startswith("groq/"): graph_api_key = settings.groq_api_key
        elif primary.startswith("openrouter/"): graph_api_key = settings.openrouter_api_key
        elif primary.startswith("openai/"): graph_api_key = os.getenv("NVIDIA_API_KEY")

        masked_graph_key = f"{graph_api_key[:4]}...{graph_api_key[-4:]}" if (graph_api_key and len(graph_api_key) > 8) else "NOT_SET_OR_TOO_SHORT"
        log.info("graph_llm_config", model=primary, key=masked_graph_key)

        from civicsetu.agent.graph import get_compiled_graph
        graph = get_compiled_graph()

        log.info("phase1_start")
        invoked = run_phase1(rows, graph)

        if EVAL_PHASE == "1":
            log.info("phase1_complete", rows=len(invoked), output=str(PHASE1_OUTPUT))
            return

    # ── Phase 2 — RAGAS scoring ────────────────────────────────────────────────
    log.info("phase2_start")
    judge_llm, judge_embeddings = build_judge()
    scored = run_phase2(invoked, judge_llm, judge_embeddings)

    print_summary(scored)

    jurisdictions = sorted({r["jurisdiction"] or "MULTI" for r in scored})
    query_types = sorted({r["query_type"] for r in scored})
    report = {
        "run_at":          datetime.now(timezone.utc).isoformat(),
        "dataset_size":    len(scored),
        "graph_model":     os.getenv("PRIMARY_MODEL", "from-.env"),
        "judge_model":     judge_model,
        "pass_threshold":  PASS_THRESHOLD,
        "overall":         _group_stats(scored),
        "by_jurisdiction": {
            jur: _group_stats([r for r in scored if (r["jurisdiction"] or "MULTI") == jur])
            for jur in jurisdictions
        },
        "by_query_type": {
            qt: _group_stats([r for r in scored if r["query_type"] == qt])
            for qt in query_types
        },
        "rows": scored,
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    overall = report["overall"]
    log.info(
        "eval_complete",
        output=str(OUTPUT_PATH),
        rows=len(scored),
        faithfulness=overall.get("faithfulness"),
        answer_relevancy=overall.get("answer_relevancy"),
        context_precision=overall.get("context_precision"),
        pass_rate=overall.get("pass_rate"),
    )
