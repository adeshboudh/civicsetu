"""
CivicSetu RAGAS evaluation — two checkpointed phases.

  Phase 1: Invoke RAG graph for every query → save to eval_phase1_results.json
  Phase 2: Score results with RAGAS (Faithfulness, AnswerRelevancy, ContextPrecision)

Usage:
    uv run python scripts/run_eval.py
    EVAL_LIMIT=5 uv run python scripts/run_eval.py          # quick smoke-test

Graph LLM override — routes graph through osmapi before civicsetu is imported:
    EVAL_PRIMARY_MODEL=qwen3.5-122b-a10b uv run python scripts/run_eval.py
    EVAL_PRIMARY_MODEL=qwen3.5-397b-a17b uv run python scripts/run_eval.py

    When set, the script configures LiteLLM to use osmapi (OSMAPI_API_KEY + base URL)
    as the openai-compatible provider for all graph calls. No fallbacks — single model.

Judge (RAGAS scorer) model:
    JUDGE_MODEL=qwen3.5-397b-a17b uv run python scripts/run_eval.py

Resume after phase 2 failure (phase 1 cached in eval_phase1_results.json):
    uv run python scripts/run_eval.py
    # Phase 1 prints "all N rows loaded from cache" and is skipped

Force re-run phase 1:
    rm eval_phase1_results.json && uv run python scripts/run_eval.py

Disable no_reasoning (not recommended for Qwen3 thinking models):
    NO_REASONING=false uv run python scripts/run_eval.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import io
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv
load_dotenv()

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

ROOT         = Path(__file__).parent.parent
DATASET_PATH = ROOT / "eval" / "golden_dataset.jsonl"
OUTPUT_PATH  = ROOT / "eval_results.json"
PHASE1_OUTPUT = ROOT / "eval_phase1_results.json"
GENERATOR_FALLBACK_ANSWER = "Unable to generate a structured response. Please try again."

PASS_THRESHOLD      = float(os.getenv("PASS_THRESHOLD", "0.7"))
EVAL_LIMIT          = int(os.getenv("EVAL_LIMIT", "0")) or None
JUDGE_MODEL         = os.getenv("JUDGE_MODEL", "qwen3.5-122b-a10b")
NO_REASONING        = os.getenv("NO_REASONING", "true").lower() == "true"
PHASE2_DELAY_SEC    = float(os.getenv("PHASE2_DELAY_SEC", "5"))
PHASE2_MAX_RETRIES  = int(os.getenv("PHASE2_MAX_RETRIES", "4"))
RAGAS_MAX_CONTEXTS  = int(os.getenv("RAGAS_MAX_CONTEXTS", "3"))
RAGAS_CONTEXT_CHAR_LIMIT = int(os.getenv("RAGAS_CONTEXT_CHAR_LIMIT", "800"))
RAGAS_ANSWER_CHAR_LIMIT = int(os.getenv("RAGAS_ANSWER_CHAR_LIMIT", "800"))
RAGAS_REFERENCE_CHAR_LIMIT = int(os.getenv("RAGAS_REFERENCE_CHAR_LIMIT", "600"))

# Graph LLM override — osmapi model name only (e.g. "qwen3.5-122b-a10b")
# When set, all graph calls route through osmapi via LiteLLM's openai-compat provider.
EVAL_PRIMARY_MODEL = os.getenv("EVAL_PRIMARY_MODEL")

# Phase selector — "1" = graph invocation only, "2" = RAGAS scoring only, None = both
EVAL_PHASE = os.getenv("EVAL_PHASE")


# ── Dataset ────────────────────────────────────────────────────────────────────

def _get_osmapi_key() -> str | None:
    return os.getenv("OSMAPI_API_KEY") or os.getenv("OSM_API_KEY")


def load_dataset(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ── Judge setup ────────────────────────────────────────────────────────────────

class _NoReasoningTransport(httpx.AsyncBaseTransport):
    def __init__(self, wrapped: httpx.AsyncBaseTransport) -> None:
        self._wrapped = wrapped

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        ct = request.headers.get("content-type", "")
        if "application/json" in ct and request.content:
            import json as _json
            body = _json.loads(request.content)
            body.setdefault("no_reasoning", True)
            content = _json.dumps(body).encode()
            headers = request.headers.copy()
            headers.pop("content-length", None)
            request = httpx.Request(
                request.method,
                request.url,
                headers=headers,
                content=content,
                extensions=request.extensions,
            )
        return await self._wrapped.handle_async_request(request)


def _is_gemini_model(model: str) -> bool:
    return model.startswith("gemini/") or model.startswith("gemini-")


def build_judge():
    """
    Two judge backends:
      Gemini  — JUDGE_MODEL starts with "gemini/" or "gemini-"
                Uses GEMINI_API_KEY_2 for both LLM and embeddings.
                Set via:  make eval-p2 JUDGE_MODEL=gemini/gemini-2.5-flash-lite
      osmapi  — default; any other model name
                Uses OSMAPI_API_KEY for LLM, GEMINI_API_KEY_2 for embeddings.
    """
    from ragas.llms import llm_factory
    from ragas.embeddings import GoogleEmbeddings
    from google import genai

    gemini_key = os.getenv("GEMINI_API_KEY_2")
    if not gemini_key:
        print("ERROR: GEMINI_API_KEY_2 not set in .env (needed for embeddings + Gemini judge)", file=sys.stderr)
        sys.exit(1)

    judge_embeddings = GoogleEmbeddings(
        client=genai.Client(api_key=gemini_key),
        model="gemini-embedding-001",
    )

    if _is_gemini_model(JUDGE_MODEL):
        # LiteLLM uses GEMINI_API_KEY env var for gemini/ models
        os.environ["GEMINI_API_KEY"] = gemini_key
        model = JUDGE_MODEL if "/" in JUDGE_MODEL else f"gemini/{JUDGE_MODEL}"
        judge_llm = llm_factory(model, max_tokens=8192)
        print(f"  Judge LLM  : Gemini / {model}")
        print(f"  Embeddings : Google gemini-embedding-001")
    else:
        from openai import AsyncOpenAI
        osmapi_key = _get_osmapi_key()
        if not osmapi_key:
            print("ERROR: OSMAPI_API_KEY not set in .env", file=sys.stderr)
            sys.exit(1)
        if NO_REASONING:
            _base = httpx.AsyncHTTPTransport()
            _http_client = httpx.AsyncClient(transport=_NoReasoningTransport(_base))
            llm_client = AsyncOpenAI(
                api_key=osmapi_key,
                base_url="https://api.osmapi.com/v1",
                timeout=120.0,
                http_client=_http_client,
            )
        else:
            llm_client = AsyncOpenAI(
                api_key=osmapi_key,
                base_url="https://api.osmapi.com/v1",
                timeout=120.0,
            )
        judge_llm = llm_factory(JUDGE_MODEL, client=llm_client, max_tokens=8192)
        print(f"  Judge LLM  : osmapi / {JUDGE_MODEL}")
        print(f"  Embeddings : Google gemini-embedding-001")
        print(f"  no_reasoning : {NO_REASONING}")

    return judge_llm, judge_embeddings


# ── Graph invocation ───────────────────────────────────────────────────────────

def invoke_graph(graph, row: dict) -> dict:
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

    start = time.perf_counter()
    try:
        result     = graph.invoke(state)
        latency_ms = (time.perf_counter() - start) * 1000
        answer     = result.get("raw_response") or ""
        reranked   = result.get("reranked_chunks") or []
        contexts   = [_format_metric_context(rc) for rc in reranked if rc.chunk.text]
        citations  = result.get("citations") or []
        confidence = result.get("confidence_score") or 0.0
        query_type = str(result.get("query_type") or "unknown")
        error      = result.get("error")
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        answer, contexts, citations = "", [], []
        confidence, query_type, error = 0.0, "error", str(exc)

    return {
        "id":                  row["id"],
        "jurisdiction":        row["jurisdiction"],
        "query_type":          row["query_type"],
        "query":               row["query"],
        "ground_truth":        row["ground_truth"],
        "answer":              answer,
        "contexts":            contexts,
        "citations_count":     len(citations),
        "confidence_score":    round(confidence, 3),
        "query_type_resolved": query_type,
        "latency_ms":          round(latency_ms, 1),
        "error":               error,
    }


def _format_metric_context(retrieved_chunk) -> str:
    chunk = retrieved_chunk.chunk
    return (
        f"{chunk.doc_name} - {chunk.section_id}: {chunk.section_title}\n"
        f"Jurisdiction: {chunk.jurisdiction}\n"
        f"{chunk.text}"
    )


# ── Phase 1: graph invocation with checkpointing ───────────────────────────────

def run_phase1(rows: list[dict], graph) -> list[dict]:
    """Invoke graph for each row, checkpointing after every row to PHASE1_OUTPUT."""
    n = len(rows)

    if PHASE1_OUTPUT.exists():
        cached = json.loads(PHASE1_OUTPUT.read_text(encoding="utf-8"))
        done: dict[str, dict] = {
            r["id"]: r
            for r in cached
            if _phase1_result_complete(r)
        }
        stale = len(cached) - len(done)
        if stale:
            print(f"Phase 1: ignoring {stale} stale cached row(s)")
        if all(row["id"] in done for row in rows):
            print(f"Phase 1: all {n} rows loaded from cache")
            return [done[row["id"]] for row in rows]
        else:
            print(f"Phase 1: resuming — {len(done)} rows cached, {n - len(done)} remaining")
    else:
        done = {}

    for i, row in enumerate(rows, 1):
        if row["id"] in done:
            print(f"  [{i:02}/{n}] {row['id']} — cached")
            continue

        result = invoke_graph(graph, row)
        done[row["id"]] = result

        status = "OK" if result["answer"] else "EMPTY"
        conf = result["confidence_score"]
        latency = result["latency_ms"]
        print(f"  [{i:02}/{n}] {row['id']} … {status} ({latency:.0f}ms conf={conf})")

        # Checkpoint immediately after each row
        PHASE1_OUTPUT.write_text(
            json.dumps(list(done.values()), indent=2, default=str),
            encoding="utf-8",
        )

    # Return in original row order
    return [done[row["id"]] for row in rows]


def _phase1_result_complete(row: dict) -> bool:
    answer = (row.get("answer") or "").strip()
    contexts = row.get("contexts") or []
    return (
        not row.get("error")
        and bool(answer)
        and answer != GENERATOR_FALLBACK_ANSWER
        and bool(contexts)
        and all(_is_metric_context(context) for context in contexts)
        and (row.get("citations_count") or 0) > 0
        and (row.get("confidence_score") or 0.0) > 0.0
    )


def _is_metric_context(context: str) -> bool:
    return " - " in context and "Jurisdiction:" in context


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
    lowered = error.lower()
    if any(marker in lowered for marker in quota_markers):
        return 60
    return None


def _prepare_metric_row(row: dict) -> dict:
    prepared = dict(row)
    prepared["answer"] = (row.get("answer") or "")[:RAGAS_ANSWER_CHAR_LIMIT]
    prepared["ground_truth"] = (row.get("ground_truth") or "")[:RAGAS_REFERENCE_CHAR_LIMIT]
    prepared["contexts"] = [
        (context or "")[:RAGAS_CONTEXT_CHAR_LIMIT]
        for context in (row.get("contexts") or [])[:RAGAS_MAX_CONTEXTS]
    ]
    return prepared


def _score_batch_unbounded(rows: list[dict], judge_llm, judge_embeddings) -> list[dict]:
    """Score all rows at once — 3 batch_score calls total (one per metric)."""
    from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision

    scoreable = [_prepare_metric_row(r) for r in rows if r["answer"] and r["contexts"]]
    skipped   = [r for r in rows if not (r["answer"] and r["contexts"])]

    if skipped:
        print(f"  Skipping {len(skipped)} rows (no answer/context): "
              f"{[r['id'] for r in skipped]}")

    f_metric  = Faithfulness(llm=judge_llm)
    ar_metric = AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings)
    cp_metric = ContextPrecision(llm=judge_llm)

    print(f"  faithfulness ({len(scoreable)} rows) ...", end=" ", flush=True)
    t0 = time.perf_counter()
    f_results = f_metric.batch_score([
        {"user_input": r["query"], "response": r["answer"], "retrieved_contexts": r["contexts"]}
        for r in scoreable
    ])
    print(f"done ({time.perf_counter() - t0:.1f}s)")

    print(f"  answer_relevancy ...", end=" ", flush=True)
    t0 = time.perf_counter()
    ar_results = ar_metric.batch_score([
        {"user_input": r["query"], "response": r["answer"]}
        for r in scoreable
    ])
    print(f"done ({time.perf_counter() - t0:.1f}s)")

    print(f"  context_precision ...", end=" ", flush=True)
    t0 = time.perf_counter()
    cp_results = cp_metric.batch_score([
        {"user_input": r["query"], "reference": r["ground_truth"], "retrieved_contexts": r["contexts"]}
        for r in scoreable
    ])
    print(f"done ({time.perf_counter() - t0:.1f}s)")

    scored_map: dict[str, dict] = {}
    for row, f_r, ar_r, cp_r in zip(scoreable, f_results, ar_results, cp_results):
        row = dict(row)
        row["faithfulness"]      = round(_safe(f_r.value), 3)
        row["answer_relevancy"]  = round(_safe(ar_r.value), 3)
        row["context_precision"] = round(_safe(cp_r.value), 3)
        row["pass"] = (
            row["faithfulness"]      >= PASS_THRESHOLD
            and row["answer_relevancy"]  >= PASS_THRESHOLD
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


def _failed_rows(rows: list[dict], error: str) -> list[dict]:
    result = []
    for row in rows:
        row = dict(row)
        row["faithfulness"] = 0.0
        row["answer_relevancy"] = 0.0
        row["context_precision"] = 0.0
        row["pass"] = False
        existing_error = row.get("error")
        row["error"] = f"{existing_error}; {error}" if existing_error else error
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
    return all(metric in row for metric in metrics) and not row.get("error")


def _write_phase2_checkpoint(rows: list[dict]) -> None:
    jurisdictions = sorted({r["jurisdiction"] or "MULTI" for r in rows})
    query_types   = sorted({r["query_type"] for r in rows})
    report = {
        "run_at":          datetime.now(timezone.utc).isoformat(),
        "dataset_size":    len(rows),
        "graph_model":     EVAL_PRIMARY_MODEL or os.getenv("PRIMARY_MODEL", "from-.env"),
        "judge_model":     JUDGE_MODEL,
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
    """Score rows sequentially with checkpointing and retry-aware sleeps."""
    done: dict[str, dict] = {}
    if OUTPUT_PATH.exists():
        try:
            cached_report = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
            cached_rows = cached_report.get("rows", [])
            done = {
                row["id"]: row
                for row in cached_rows
                if _phase2_row_complete(row)
            }
        except (json.JSONDecodeError, OSError, AttributeError):
            done = {}

    total = len(invoked)
    if all(row["id"] in done for row in invoked):
        print(f"Phase 2: all {total} rows loaded from cache")
        return [done[row["id"]] for row in invoked]

    if done:
        print(f"Phase 2: resuming - {len(done)} rows cached, {total - len(done)} remaining")

    results_by_id = dict(done)
    for i, row in enumerate(invoked, 1):
        if row["id"] in done:
            print(f"  [{i:02}/{total}] {row['id']} - cached")
            continue

        scored = None
        for attempt in range(1, PHASE2_MAX_RETRIES + 2):
            scored = score_batch([row], judge_llm, judge_embeddings, label=row["id"])[0]
            retry_delay = _retry_delay_seconds(scored.get("error"))
            if retry_delay is None or attempt > PHASE2_MAX_RETRIES:
                break
            print(f"  [{i:02}/{total}] {row['id']} - retrying in {retry_delay}s (attempt {attempt})")
            time.sleep(retry_delay)

        assert scored is not None
        results_by_id[row["id"]] = scored
        ordered_rows = [results_by_id[r["id"]] for r in invoked if r["id"] in results_by_id]
        _write_phase2_checkpoint(ordered_rows)

        err = scored.get("error")
        if err:
            print(f"  [{i:02}/{total}] {row['id']} - FAIL")
        else:
            print(
                f"  [{i:02}/{total}] {row['id']} - "
                f"faith={scored['faithfulness']:.2f} rel={scored['answer_relevancy']:.2f} "
                f"prec={scored['context_precision']:.2f} {'PASS' if scored['pass'] else 'fail'}"
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
    passed  = sum(1 for r in rows if r["pass"])
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
        print(f"  {jur:<20} faith={s['faithfulness']:.2f}  "
              f"rel={s['answer_relevancy']:.2f}  "
              f"prec={s['context_precision']:.2f}  "
              f"pass={s['pass_rate']:.0%}")

    failures = [r for r in rows if not r["pass"]]
    if failures:
        print(f"\n  Failures ({len(failures)}):")
        for r in failures:
            print(f"    FAIL [{r['id']}]  "
                  f"faith={r['faithfulness']:.2f}  "
                  f"rel={r['answer_relevancy']:.2f}  "
                  f"prec={r['context_precision']:.2f}  "
                  f"err={r.get('error') or '-'}")
    print("=" * 72)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Phase 2 only — load from phase 1 cache, skip graph entirely ───────────
    if EVAL_PHASE == "2":
        if not PHASE1_OUTPUT.exists():
            print(f"ERROR: {PHASE1_OUTPUT} not found — run phase 1 first (make eval-p1)", file=sys.stderr)
            sys.exit(1)
        invoked = json.loads(PHASE1_OUTPUT.read_text(encoding="utf-8"))
        if EVAL_LIMIT:
            invoked = invoked[:EVAL_LIMIT]
        print(f"CivicSetu RAGAS Eval — phase 2 only | {len(invoked)} rows | judge={JUDGE_MODEL} | threshold={PASS_THRESHOLD}")

    # ── Phase 1 (and optionally phase 2) — invoke graph ───────────────────────
    else:
        if not DATASET_PATH.exists():
            print(f"ERROR: dataset not found at {DATASET_PATH}", file=sys.stderr)
            sys.exit(1)

        rows = load_dataset(DATASET_PATH)
        if EVAL_LIMIT:
            rows = rows[:EVAL_LIMIT]

        phase_label = "phase 1 only" if EVAL_PHASE == "1" else "both phases"
        print(f"CivicSetu RAGAS Eval — {len(rows)} queries | {phase_label} | judge={JUDGE_MODEL} | threshold={PASS_THRESHOLD}")

        # ── EVAL_PRIMARY_MODEL override block — do not touch ──────────────────
        # Apply graph LLM overrides BEFORE importing civicsetu.
        # nodes.py builds FALLBACK_MODELS at module import time from settings,
        # so env vars must be set before the first import of civicsetu.agent.graph.
        if EVAL_PRIMARY_MODEL:
            osmapi_key = _get_osmapi_key()
            if not osmapi_key:
                print("ERROR: OSMAPI_API_KEY not set in .env", file=sys.stderr)
                sys.exit(1)
            litellm_model = f"openai/{EVAL_PRIMARY_MODEL}"
            os.environ["OPENAI_API_KEY"]   = osmapi_key
            os.environ["OPENAI_API_BASE"]  = "https://api.osmapi.com/v1"
            os.environ["PRIMARY_MODEL"]    = litellm_model
            os.environ["FALLBACK_MODEL_1"] = litellm_model
            os.environ["FALLBACK_MODEL_2"] = litellm_model
            os.environ["FALLBACK_MODEL_3"] = litellm_model
            print(f"  Graph LLM : osmapi / {EVAL_PRIMARY_MODEL}")
        else:
            primary = os.getenv("PRIMARY_MODEL", "gemini/gemini-2.5-flash-lite")
            print(f"  Graph LLM : {primary}  (from .env / default RAG chain)")

        from civicsetu.agent.graph import get_compiled_graph
        graph = get_compiled_graph()

        print(f"\nPhase 1 — Graph invocation...")
        invoked = run_phase1(rows, graph)

        if EVAL_PHASE == "1":
            print(f"\nPhase 1 complete — {len(invoked)} rows saved to {PHASE1_OUTPUT}")
            return

    # ── Phase 2 — RAGAS scoring ────────────────────────────────────────────────
    print(f"\nPhase 2 — RAGAS scoring...")
    judge_llm, judge_embeddings = build_judge()
    scored = run_phase2(invoked, judge_llm, judge_embeddings)

    print_summary(scored)

    jurisdictions = sorted({r["jurisdiction"] or "MULTI" for r in scored})
    query_types   = sorted({r["query_type"] for r in scored})
    report = {
        "run_at":          datetime.now(timezone.utc).isoformat(),
        "dataset_size":    len(scored),
        "graph_model":     EVAL_PRIMARY_MODEL or os.getenv("PRIMARY_MODEL", "from-.env"),
        "judge_model":     JUDGE_MODEL,
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
    print(f"\nFull results → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
