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
"""
from __future__ import annotations

import argparse
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

# ── Rate-limit constants (override via env vars) ───────────────────────────────
BATCH_SIZE      = int(os.getenv("BATCH_SIZE", "3"))
BATCH_DELAY_SEC = int(os.getenv("BATCH_DELAY_SEC", "60"))
PASS_THRESHOLD  = float(os.getenv("PASS_THRESHOLD", "0.7"))
EVAL_LIMIT      = int(os.getenv("EVAL_LIMIT", "0")) or None   # 0 = no limit
JUDGE_MODEL     = os.getenv("JUDGE_MODEL", "gemini-2.0-flash-lite")

ROOT            = Path(__file__).parent.parent
DATASET_PATH    = ROOT / "eval" / "golden_dataset.jsonl"
PHASE1_OUT      = ROOT / "eval_phase1_results.json"
PHASE2_OUT      = ROOT / "eval_results.json"


# ── RAGAS judge (RAGAS 0.4.x native API via instructor) ───────────────────────

def build_judge():
    """Build RAGAS 0.4.x judge using LiteLLM + instructor."""
    import instructor
    from litellm import completion as litellm_completion
    from ragas.llms import llm_factory
    from ragas.embeddings import GoogleEmbeddings
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY_2")
    if not api_key:
        print(
            "ERROR: GEMINI_API_KEY_2 is not set.\n"
            "Add it to your .env: GEMINI_API_KEY_2=<your-key>",
            file=sys.stderr,
        )
        sys.exit(1)

    # Route judge calls through LiteLLM using the dedicated GEMINI_API_KEY_2.
    # LiteLLM reads GEMINI_API_KEY for the gemini/ provider.
    os.environ["GEMINI_API_KEY"] = api_key
    instructor_client = instructor.from_litellm(litellm_completion)
    judge_llm = llm_factory(f"gemini/{JUDGE_MODEL}", client=instructor_client)
    judge_embeddings = GoogleEmbeddings(
        google_api_key=api_key,
        model="models/embedding-001",
    )
    return judge_llm, judge_embeddings


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


def score_batch(batch: list[dict], judge_llm, judge_embeddings) -> list[dict]:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision

    scoreable = [r for r in batch if r["answer"] and r["contexts"]]
    skipped   = [r for r in batch if not (r["answer"] and r["contexts"])]

    scored = []
    if scoreable:
        ds = Dataset.from_list([
            {
                "question":     r["query"],
                "answer":       r["answer"],
                "contexts":     r["contexts"],
                "ground_truth": r["ground_truth"],
            }
            for r in scoreable
        ])
        result = evaluate(
            ds,
            metrics=[
                Faithfulness(llm=judge_llm),
                AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings),
                ContextPrecision(llm=judge_llm),
            ],
            raise_exceptions=False,
            column_map={
                "user_input":        "question",
                "response":          "answer",
                "retrieved_contexts": "contexts",
                "reference":         "ground_truth",
            },
        )
        result_df = result.to_pandas()
        for row, (_, scores) in zip(scoreable, result_df.iterrows()):
            row = dict(row)
            row["faithfulness"]      = round(_safe_metric(scores.get("faithfulness")), 3)
            row["answer_relevancy"]  = round(_safe_metric(scores.get("answer_relevancy")), 3)
            row["context_precision"] = round(_safe_metric(scores.get("context_precision")), 3)
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
    judge_llm, judge_embeddings = build_judge()
    all_scored: list[dict] = []

    batches = [invoked[i:i + BATCH_SIZE] for i in range(0, len(invoked), BATCH_SIZE)]
    for batch_num, batch in enumerate(batches, 1):
        ids = [r["id"] for r in batch]
        print(f"  Batch {batch_num}/{len(batches)}: {ids} ...", end=" ", flush=True)
        scored = score_batch(batch, judge_llm, judge_embeddings)
        all_scored.extend(scored)
        print("done")
        if batch_num < len(batches):
            print(f"  Sleeping {BATCH_DELAY_SEC}s before next batch...")
            time.sleep(BATCH_DELAY_SEC)

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
