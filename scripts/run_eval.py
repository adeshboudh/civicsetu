# scripts/run_eval.py
"""
RAGAS offline benchmark for CivicSetu.

Two independent phases so you never re-invoke the graph just to re-score:

    Phase 1 — invoke graph for all queries, save raw results:
        make eval-collect
        # or: uv run python scripts/run_eval.py --phase 1

    Phase 2 — RAGAS scoring on saved results (reads eval_phase1_results.json):
        make eval-score
        # or: uv run python scripts/run_eval.py --phase 2

    Both phases in sequence:
        make eval
        # or: uv run python scripts/run_eval.py

Env-var tuning:
    BATCH_SIZE=3 BATCH_DELAY_SEC=60 EVAL_LIMIT=3 make eval-score

Judge provider (Phase 2):
    # Gemini free tier (15 RPM) — default; sleeps BATCH_DELAY_SEC between each metric
    JUDGE_PROVIDER=gemini BATCH_SIZE=1 BATCH_DELAY_SEC=60 make eval-score

    # Dual Gemini keys — parallel 2-worker mode (~30 RPM effective, default delay=30s)
    # Set GEMINI_API_KEY_2 + GEMINI_API_KEY_3 in .env; script auto-detects second key.
    BATCH_SIZE=2 make eval-score

    # OpenRouter free tier — more generous RPM; set OPENROUTER_API_KEY in .env
    JUDGE_PROVIDER=openrouter JUDGE_MODEL=stepfun/step-3.5-flash:free make eval-score
    JUDGE_PROVIDER=openrouter make eval-score   # uses stepfun/step-3.5-flash:free by default
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ── Rate-limit constants (override via env vars) ───────────────────────────────
BATCH_SIZE      = int(os.getenv("BATCH_SIZE", "2"))   # 1 row/batch keeps burst ≤5 calls
BATCH_DELAY_SEC = int(os.getenv("BATCH_DELAY_SEC", "60"))
PASS_THRESHOLD  = float(os.getenv("PASS_THRESHOLD", "0.7"))
EVAL_LIMIT      = int(os.getenv("EVAL_LIMIT", "0")) or None   # 0 = no limit

# Judge provider: osmapi (OSMAPI_API_KEY) — active
# Embeddings:     google-genai (GEMINI_API_KEY_2) — still needed for AnswerRelevancy
# Previous providers (commented out in build_judge): gemini, openrouter
JUDGE_PROVIDER  = os.getenv("JUDGE_PROVIDER", "osmapi")
JUDGE_MODEL     = os.getenv("JUDGE_MODEL", "qwen3.5-122b-a10b")

ROOT            = Path(__file__).parent.parent
DATASET_PATH    = ROOT / "eval" / "golden_dataset.jsonl"
PHASE1_OUT      = ROOT / "eval_phase1_results.json"
PHASE2_OUT      = ROOT / "eval_results.json"


# ── Logging helper ─────────────────────────────────────────────────────────────

def _log(msg: str, label: str = "") -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    prefix = f"[{ts}]" + (f" [{label}]" if label else "")
    print(f"{prefix} {msg}", flush=True)


def _sleep_log(seconds: int, label: str = "") -> None:
    resume = datetime.fromtimestamp(time.time() + seconds).strftime("%H:%M:%S")
    _log(f"sleeping {seconds}s — resuming at {resume}", label)
    time.sleep(seconds)


# ── RAGAS judge (RAGAS 0.4.x native API via instructor) ───────────────────────

def build_judge(gemini_key: str):
    """
    Build RAGAS 0.4.x judge LLM + embeddings.

    LLM: osmapi (OSMAPI_API_KEY) — OpenAI-compatible endpoint, free credits.
    Embeddings: Google GenAI (GEMINI_API_KEY_2) — AnswerRelevancy needs semantic similarity.

    # ── Previously used providers (commented out) ──────────────────────────
    # JUDGE_PROVIDER=gemini:
    #     llm_client = AsyncOpenAI(
    #         api_key=gemini_key,
    #         base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    #         timeout=120.0,
    #     )
    #     print(f"  Judge: Gemini / {JUDGE_MODEL}")
    #
    # JUDGE_PROVIDER=openrouter:
    #     openrouter_key = os.getenv("OPENROUTER_API_KEY")
    #     llm_client = AsyncOpenAI(
    #         api_key=openrouter_key,
    #         base_url="https://openrouter.ai/api/v1",
    #         timeout=120.0,
    #     )
    #     print(f"  Judge: OpenRouter / {JUDGE_MODEL}")
    # ───────────────────────────────────────────────────────────────────────
    """
    from google import genai
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.embeddings import GoogleEmbeddings

    osmapi_key = os.getenv("OSMAPI_API_KEY")
    if not osmapi_key:
        print(
            "ERROR: OSMAPI_API_KEY is not set.\n"
            "Add it to your .env: OSMAPI_API_KEY=<your-key>",
            file=sys.stderr,
        )
        sys.exit(1)

    llm_client = AsyncOpenAI(
        api_key=osmapi_key,
        base_url="https://api.osmapi.com/v1",
        timeout=120.0,  # 2-min cap per call — prevents infinite hangs
    )
    print(f"  Judge: osmapi / {JUDGE_MODEL}")

    # max_tokens=8192: RERA answers produce many NLI statements; the RAGAS
    # default of 1024 causes IncompleteOutputException on complex legal answers.
    judge_llm = llm_factory(JUDGE_MODEL, client=llm_client, max_tokens=8192)

    # Embeddings: google-genai (AnswerRelevancy needs semantic similarity;
    # osmapi has no embedding endpoint).
    genai_client = genai.Client(api_key=gemini_key)
    judge_embeddings = GoogleEmbeddings(client=genai_client, model="gemini-embedding-001")

    return judge_llm, judge_embeddings


def build_judge_pool() -> list[tuple]:
    """
    Build a single judge (llm, embeddings) pair using GEMINI_API_KEY_2.
    Always single-worker sequential mode to avoid auth conflicts.
    """
    from dotenv import load_dotenv
    load_dotenv()

    key1 = os.getenv("GEMINI_API_KEY_2")
    if not key1:
        print(
            "ERROR: GEMINI_API_KEY_2 is not set (required for embeddings).\n"
            "Add it to your .env: GEMINI_API_KEY_2=<your-gemini-key>",
            file=sys.stderr,
        )
        sys.exit(1)

    return [build_judge(key1)]


def score_batch_in_thread(batch: list[dict], judge_llm, judge_embeddings, label: str = "") -> list[dict]:
    """Thread-safe wrapper: gives each worker thread its own asyncio event loop."""
    ids = [r["id"] for r in batch]
    _log(f"starting {ids}", label)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return score_batch(batch, judge_llm, judge_embeddings, label=label)
    finally:
        loop.close()


# ── Dataset helpers ────────────────────────────────────────────────────────────

def load_dataset(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


# ── Phase 1: graph invocation ──────────────────────────────────────────────────

def invoke_graph(graph, row: dict) -> dict:
    """Run one golden row through the graph and return enriched result dict."""
    from civicsetu.models.enums import Jurisdiction

    jurisdiction = None
    if row.get("jurisdiction"):
        try:
            jurisdiction = Jurisdiction(row["jurisdiction"])
        except ValueError:
            jurisdiction = None

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
        result = graph.invoke(state)
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
        "id":                 row["id"],
        "jurisdiction":       row["jurisdiction"],
        "query_type":         row["query_type"],
        "query":              row["query"],
        "ground_truth":       row["ground_truth"],
        "answer":             answer,
        "contexts":           contexts,
        "citations_count":    len(citations),
        "confidence_score":   round(confidence, 3),
        "query_type_resolved": query_type,
        "latency_ms":         round(latency_ms, 1),
        "error":              error,
    }


def run_phase1(rows: list[dict]) -> list[dict]:
    from civicsetu.agent.graph import get_compiled_graph

    print(f"\nPhase 1: Invoking graph for {len(rows)} queries...")
    graph = get_compiled_graph()
    invoked: list[dict] = []
    for i, row in enumerate(rows, 1):
        print(f"  [{i:02}/{len(rows)}] {row['id']} ...", end=" ", flush=True)
        result = invoke_graph(graph, row)
        invoked.append(result)
        status = "OK" if result["answer"] else "EMPTY"
        print(f"{status}  ({result['latency_ms']:.0f}ms, conf={result['confidence_score']})")

    PHASE1_OUT.write_text(json.dumps(invoked, indent=2, default=str), encoding="utf-8")
    print(f"\nPhase 1 complete — results saved to {PHASE1_OUT}")
    return invoked


# ── Phase 2: RAGAS scoring ─────────────────────────────────────────────────────

def _safe_metric(val, default: float = 0.0) -> float:
    try:
        f = float(val)
        return default if math.isnan(f) else f
    except (TypeError, ValueError):
        return default


def score_batch(batch: list[dict], judge_llm, judge_embeddings, label: str = "") -> list[dict]:
    from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision

    scoreable = [r for r in batch if r["answer"] and r["contexts"]]
    skipped   = [r for r in batch if not (r["answer"] and r["contexts"])]

    ids = [r["id"] for r in scoreable]
    scored = []
    if scoreable:
        # Collections metrics use their own .batch_score() API, not ragas.evaluate().
        # evaluate() only works with ragas.metrics.Metric subclasses; these inherit
        # from ragas.metrics.collections.base.BaseMetric which is a different hierarchy.
        f_metric  = Faithfulness(llm=judge_llm)
        ar_metric = AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings)
        cp_metric = ContextPrecision(llm=judge_llm)

        # Sleep BATCH_DELAY_SEC between each metric so the per-minute quota resets.
        # abatch_score fires all rows concurrently (asyncio.gather), so even with
        # BATCH_SIZE=1 we get ~2-5 calls per metric. Sleeping 60 s between metrics
        # keeps us well under Gemini free tier's 15 RPM cap.
        _log(f"faithfulness    → scoring {ids}", label)
        t0 = time.perf_counter()
        f_results  = f_metric.batch_score([
            {"user_input": r["query"], "response": r["answer"], "retrieved_contexts": r["contexts"]}
            for r in scoreable
        ])
        _log(f"faithfulness    ✓ done ({time.perf_counter() - t0:.1f}s)", label)
        _sleep_log(BATCH_DELAY_SEC, label)

        _log(f"answer_relevancy → scoring {ids}", label)
        t0 = time.perf_counter()
        ar_results = ar_metric.batch_score([
            {"user_input": r["query"], "response": r["answer"]}
            for r in scoreable
        ])
        _log(f"answer_relevancy ✓ done ({time.perf_counter() - t0:.1f}s)", label)
        _sleep_log(BATCH_DELAY_SEC, label)

        _log(f"context_precision → scoring {ids}", label)
        t0 = time.perf_counter()
        cp_results = cp_metric.batch_score([
            {"user_input": r["query"], "reference": r["ground_truth"], "retrieved_contexts": r["contexts"]}
            for r in scoreable
        ])
        _log(f"context_precision ✓ done ({time.perf_counter() - t0:.1f}s)", label)

        for row, f_r, ar_r, cp_r in zip(scoreable, f_results, ar_results, cp_results):
            row = dict(row)
            row["faithfulness"]      = round(_safe_metric(f_r.value), 3)
            row["answer_relevancy"]  = round(_safe_metric(ar_r.value), 3)
            row["context_precision"] = round(_safe_metric(cp_r.value), 3)
            row["pass"] = (
                row["faithfulness"]      >= PASS_THRESHOLD
                and row["answer_relevancy"]  >= PASS_THRESHOLD
                and row["context_precision"] >= PASS_THRESHOLD
            )
            scored.append(row)

    for row in skipped:
        row = dict(row)
        row["faithfulness"] = row["answer_relevancy"] = row["context_precision"] = 0.0
        row["pass"] = False
        scored.append(row)

    return scored


def compute_group_stats(rows: list[dict], key: str) -> dict:
    if not rows:
        return {}
    latencies = sorted(r["latency_ms"] for r in rows)
    n = len(latencies)
    p50 = (latencies[n // 2 - 1] + latencies[n // 2]) / 2.0 if n % 2 == 0 else latencies[n // 2]
    p90 = latencies[min(int(n * 0.9), n - 1)]
    p99 = latencies[min(int(n * 0.99), n - 1)]
    return {
        "faithfulness":      round(sum(r["faithfulness"] for r in rows) / n, 3),
        "answer_relevancy":  round(sum(r["answer_relevancy"] for r in rows) / n, 3),
        "context_precision": round(sum(r["context_precision"] for r in rows) / n, 3),
        "pass_rate":         round(sum(1 for r in rows if r["pass"]) / n, 3),
        "p50_latency_ms":    round(p50, 1),
        "p90_latency_ms":    round(p90, 1),
        "p99_latency_ms":    round(p99, 1),
    }


def print_summary(all_rows: list[dict]) -> None:
    overall = compute_group_stats(all_rows, "overall")
    passed  = sum(1 for r in all_rows if r["pass"])
    print("\n" + "=" * 72)
    print(f"RAGAS Evaluation Results  ({len(all_rows)} queries, {passed} pass)")
    print("=" * 72)
    print(f"  faithfulness      : {overall['faithfulness']:.3f}")
    print(f"  answer_relevancy  : {overall['answer_relevancy']:.3f}")
    print(f"  context_precision : {overall['context_precision']:.3f}")
    print(f"  pass_rate         : {overall['pass_rate']:.1%}")
    print(f"  p50 latency       : {overall['p50_latency_ms']:.0f} ms")
    print(f"  p90 latency       : {overall['p90_latency_ms']:.0f} ms")
    print()
    for jur in sorted({r["jurisdiction"] or "MULTI" for r in all_rows}):
        rows  = [r for r in all_rows if (r["jurisdiction"] or "MULTI") == jur]
        stats = compute_group_stats(rows, jur)
        print(f"    {jur:<20} faith={stats['faithfulness']:.2f}  "
              f"rel={stats['answer_relevancy']:.2f}  "
              f"prec={stats['context_precision']:.2f}  "
              f"pass={stats['pass_rate']:.0%}  p50={stats['p50_latency_ms']:.0f}ms")
    failures = [r for r in all_rows if not r["pass"]]
    if failures:
        print(f"\n  Failures ({len(failures)}):")
        for r in failures:
            print(f"    FAIL [{r['id']}]  "
                  f"faith={r['faithfulness']:.2f}  rel={r['answer_relevancy']:.2f}  "
                  f"prec={r['context_precision']:.2f}  err={r.get('error') or '-'}")
    print("=" * 72)


def run_phase2(invoked: list[dict]) -> list[dict]:
    print(f"\nPhase 2: RAGAS scoring {len(invoked)} rows in batches of {BATCH_SIZE}...")
    judges = build_judge_pool()
    num_workers = len(judges)
    all_scored: list[dict] = []

    batches = [invoked[i:i + BATCH_SIZE] for i in range(0, len(invoked), BATCH_SIZE)]

    if num_workers == 1:
        judge_llm, judge_embeddings = judges[0]
        for batch_num, batch in enumerate(batches, 1):
            ids = [r["id"] for r in batch]
            _log(f"Batch {batch_num}/{len(batches)}: {ids}")
            scored = score_batch(batch, judge_llm, judge_embeddings, label=f"B{batch_num}")
            all_scored.extend(scored)
            _log(f"Batch {batch_num}/{len(batches)} complete")
            if batch_num < len(batches):
                _sleep_log(BATCH_DELAY_SEC)
    else:
        # Two workers: interleave batches across keys (worker 0 = even, worker 1 = odd).
        # Each thread runs its own asyncio event loop to avoid cross-thread conflicts.
        futures: dict = {}
        with ThreadPoolExecutor(max_workers=2) as executor:
            for i, batch in enumerate(batches):
                judge_llm, judge_emb = judges[i % 2]
                ids = [r["id"] for r in batch]
                worker_label = f"W{i % 2 + 1}-B{i + 1}"
                _log(f"Submitting batch {i + 1}/{len(batches)}: {ids} → worker {i % 2 + 1}")
                f = executor.submit(score_batch_in_thread, batch, judge_llm, judge_emb, worker_label)
                futures[f] = i

            results: dict[int, list[dict]] = {}
            for f in as_completed(futures):
                idx = futures[f]
                try:
                    results[idx] = f.result(timeout=300)  # 5-min cap per batch
                except Exception as exc:
                    _log(f"Batch {idx + 1} FAILED ({type(exc).__name__}: {exc}) — skipping with zeros")
                    results[idx] = []
                else:
                    _log(f"Batch {idx + 1} complete")

        for i in sorted(results):
            all_scored.extend(results[i])

    print_summary(all_scored)

    jurisdictions = sorted({r["jurisdiction"] or "MULTI" for r in all_scored})
    query_types   = sorted({r["query_type"] for r in all_scored})
    report = {
        "run_at":          datetime.now(timezone.utc).isoformat(),
        "dataset_size":    len(all_scored),
        "batch_size":      BATCH_SIZE,
        "batch_delay_sec": BATCH_DELAY_SEC,
        "pass_threshold":  PASS_THRESHOLD,
        "overall":         compute_group_stats(all_scored, "overall"),
        "by_jurisdiction": {
            jur: compute_group_stats([r for r in all_scored if (r["jurisdiction"] or "MULTI") == jur], jur)
            for jur in jurisdictions
        },
        "by_query_type": {
            qt: compute_group_stats([r for r in all_scored if r["query_type"] == qt], qt)
            for qt in query_types
        },
        "rows": all_scored,
    }
    PHASE2_OUT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nFull results → {PHASE2_OUT}")
    return all_scored


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="CivicSetu RAGAS benchmark")
    parser.add_argument(
        "--phase", type=int, choices=[1, 2],
        help="1 = collect (graph invocations only), 2 = score (RAGAS only). Default: both."
    )
    args = parser.parse_args()

    if not DATASET_PATH.exists():
        print(f"ERROR: dataset not found at {DATASET_PATH}", file=sys.stderr)
        sys.exit(1)

    rows = load_dataset(DATASET_PATH)
    if EVAL_LIMIT:
        rows = rows[:EVAL_LIMIT]

    print(f"CivicSetu RAGAS Eval — {len(rows)} queries  |  "
          f"batch_size={BATCH_SIZE}  delay={BATCH_DELAY_SEC}s  threshold={PASS_THRESHOLD}")

    if args.phase == 1:
        run_phase1(rows)

    elif args.phase == 2:
        if not PHASE1_OUT.exists():
            print(f"ERROR: {PHASE1_OUT} not found. Run Phase 1 first: make eval-collect", file=sys.stderr)
            sys.exit(1)
        invoked = json.loads(PHASE1_OUT.read_text(encoding="utf-8"))
        if EVAL_LIMIT:
            invoked = invoked[:EVAL_LIMIT]
        all_scored = run_phase2(invoked)
        if sum(1 for r in all_scored if not r["pass"]):
            sys.exit(1)

    else:
        # Both phases
        invoked    = run_phase1(rows)
        all_scored = run_phase2(invoked)
        if sum(1 for r in all_scored if not r["pass"]):
            sys.exit(1)


if __name__ == "__main__":
    main()
