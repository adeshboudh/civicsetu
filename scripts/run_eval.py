"""
CivicSetu RAGAS evaluation — single pass, no phases.

  1. Load golden dataset
  2. Invoke RAG graph for every query (collect answers + contexts)
  3. Score all rows at once with RAGAS (3 batch_score calls total)
  4. Print summary + save eval_results.json

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

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

ROOT         = Path(__file__).parent.parent
DATASET_PATH = ROOT / "eval" / "golden_dataset.jsonl"
OUTPUT_PATH  = ROOT / "eval_results.json"

PASS_THRESHOLD      = float(os.getenv("PASS_THRESHOLD", "0.7"))
EVAL_LIMIT          = int(os.getenv("EVAL_LIMIT", "0")) or None
JUDGE_MODEL         = os.getenv("JUDGE_MODEL", "qwen3.5-122b-a10b")

# Graph LLM override — osmapi model name only (e.g. "qwen3.5-122b-a10b")
# When set, all graph calls route through osmapi via LiteLLM's openai-compat provider.
EVAL_PRIMARY_MODEL = os.getenv("EVAL_PRIMARY_MODEL")


# ── Dataset ────────────────────────────────────────────────────────────────────

def load_dataset(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ── Judge setup ────────────────────────────────────────────────────────────────

def build_judge():
    """
    LLM  : osmapi (OSMAPI_API_KEY) — OpenAI-compatible endpoint.
    Embed: Google GenAI (GEMINI_API_KEY_2) — needed for AnswerRelevancy.
    """
    from dotenv import load_dotenv
    load_dotenv()

    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.embeddings import GoogleEmbeddings
    from google import genai

    osmapi_key = os.getenv("OSMAPI_API_KEY")
    if not osmapi_key:
        print("ERROR: OSMAPI_API_KEY not set in .env", file=sys.stderr)
        sys.exit(1)

    gemini_key = os.getenv("GEMINI_API_KEY_2")
    if not gemini_key:
        print("ERROR: GEMINI_API_KEY_2 not set in .env (needed for embeddings)", file=sys.stderr)
        sys.exit(1)

    llm_client = AsyncOpenAI(
        api_key=osmapi_key,
        base_url="https://api.osmapi.com/v1",
        timeout=120.0,
    )
    judge_llm        = llm_factory(JUDGE_MODEL, client=llm_client, max_tokens=8192)
    judge_embeddings = GoogleEmbeddings(
        client=genai.Client(api_key=gemini_key),
        model="gemini-embedding-001",
    )

    print(f"  Judge LLM  : osmapi / {JUDGE_MODEL}")
    print(f"  Embeddings : Google gemini-embedding-001")
    return judge_llm, judge_embeddings


# ── Graph invocation ───────────────────────────────────────────────────────────

def invoke_graph(graph, row: dict) -> dict:
    from civicsetu.models.enums import Jurisdiction

    jurisdiction = None
    if row.get("jurisdiction"):
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
        contexts   = [rc.chunk.text for rc in reranked if rc.chunk.text]
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


# ── RAGAS scoring ──────────────────────────────────────────────────────────────

def _safe(val, default: float = 0.0) -> float:
    try:
        f = float(val)
        return default if math.isnan(f) else f
    except (TypeError, ValueError):
        return default


def score_all(rows: list[dict], judge_llm, judge_embeddings) -> list[dict]:
    """Score all rows at once — 3 batch_score calls total (one per metric)."""
    from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision

    scoreable = [r for r in rows if r["answer"] and r["contexts"]]
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
    if not DATASET_PATH.exists():
        print(f"ERROR: dataset not found at {DATASET_PATH}", file=sys.stderr)
        sys.exit(1)

    rows = load_dataset(DATASET_PATH)
    if EVAL_LIMIT:
        rows = rows[:EVAL_LIMIT]

    print(f"CivicSetu RAGAS Eval — {len(rows)} queries | judge={JUDGE_MODEL} | threshold={PASS_THRESHOLD}")

    # ── Step 1: collect ────────────────────────────────────────────────────────
    # Apply graph LLM overrides BEFORE importing civicsetu.
    # nodes.py builds FALLBACK_MODELS at module import time from settings,
    # so env vars must be set before the first import of civicsetu.agent.graph.
    if EVAL_PRIMARY_MODEL:
        # Route graph through osmapi via LiteLLM's openai-compatible provider.
        # Must be set before civicsetu.agent.graph is imported — nodes.py builds
        # FALLBACK_MODELS at module level from settings.
        osmapi_key = os.getenv("OSMAPI_API_KEY")
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
        from dotenv import load_dotenv
        load_dotenv()
        primary = os.getenv("PRIMARY_MODEL", "gemini/gemini-2.5-flash-lite")
        print(f"  Graph LLM : {primary}  (from .env)")

    from civicsetu.agent.graph import get_compiled_graph
    graph = get_compiled_graph()

    print(f"\nStep 1/2 — invoking graph ({len(rows)} queries)...")
    invoked: list[dict] = []
    for i, row in enumerate(rows, 1):
        print(f"  [{i:02}/{len(rows)}] {row['id']} ...", end=" ", flush=True)
        result = invoke_graph(graph, row)
        invoked.append(result)
        status = "OK" if result["answer"] else "EMPTY"
        print(f"{status}  ({result['latency_ms']:.0f}ms  conf={result['confidence_score']})")

    # ── Step 2: score ──────────────────────────────────────────────────────────
    print(f"\nStep 2/2 — RAGAS scoring (3 batch calls)...")
    judge_llm, judge_embeddings = build_judge()
    scored = score_all(invoked, judge_llm, judge_embeddings)

    # ── Save + print ───────────────────────────────────────────────────────────
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
