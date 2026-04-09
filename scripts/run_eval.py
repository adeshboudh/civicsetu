# scripts/run_eval.py
"""
RAGAS offline benchmark for CivicSetu.

Calls graph.invoke() directly (no HTTP server required) to capture
reranked_chunks for context fields, then scores with RAGAS in batches
to stay within free-tier API rate limits.

Run:
    uv run python scripts/run_eval.py

Tune rate limits:
    BATCH_SIZE=3 BATCH_DELAY_SEC=60 uv run python scripts/run_eval.py

Limit rows for smoke testing:
    EVAL_LIMIT=3 BATCH_SIZE=3 BATCH_DELAY_SEC=5 uv run python scripts/run_eval.py
"""
from __future__ import annotations

import json
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
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "3"))
BATCH_DELAY_SEC = int(os.getenv("BATCH_DELAY_SEC", "60"))
PASS_THRESHOLD = float(os.getenv("PASS_THRESHOLD", "0.7"))
EVAL_LIMIT = int(os.getenv("EVAL_LIMIT", "0")) or None   # 0 = no limit
# Judge model: set JUDGE_MODEL to override. Must be a google-generativeai model name
# (not LiteLLM prefix format). e.g. "gemini-2.0-flash-lite" for gemini-3.1-flash-lite-preview
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-2.0-flash-lite")

# ── Imports ────────────────────────────────────────────────────────────────────
from civicsetu.agent.graph import get_compiled_graph
from civicsetu.models.enums import Jurisdiction

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


def build_judge() -> tuple[LangchainLLMWrapper, LangchainEmbeddingsWrapper]:
    """Build RAGAS judge LLM and embeddings from GEMINI_API_KEY_2 (separate key
    to avoid rate-limit conflicts with the RAG system's own GEMINI_API_KEY)."""
    api_key = os.environ["GEMINI_API_KEY_2"]
    llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model=JUDGE_MODEL,
            google_api_key=api_key,
            temperature=1.0,  # Gemini requires temperature >= 1.0
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key,
        )
    )
    return llm, embeddings


def load_dataset(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def invoke_graph(graph, row: dict) -> dict:
    """Run one golden row through the graph and return enriched result dict."""
    jurisdiction = None
    if row.get("jurisdiction"):
        try:
            jurisdiction = Jurisdiction(row["jurisdiction"])
        except ValueError:
            jurisdiction = None

    state = {
        "query": row["query"],
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

        answer = result.get("raw_response") or ""
        reranked = result.get("reranked_chunks") or []
        contexts = [rc.chunk.text for rc in reranked if rc.chunk.text]
        citations = result.get("citations") or []
        confidence = result.get("confidence_score") or 0.0
        query_type = str(result.get("query_type") or "unknown")
        error = result.get("error")

    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        answer = ""
        contexts = []
        citations = []
        confidence = 0.0
        query_type = "error"
        error = str(exc)

    return {
        "id": row["id"],
        "jurisdiction": row["jurisdiction"],
        "query_type": row["query_type"],
        "query": row["query"],
        "ground_truth": row["ground_truth"],
        "answer": answer,
        "contexts": contexts,
        "citations_count": len(citations),
        "confidence_score": round(confidence, 3),
        "query_type_resolved": query_type,
        "latency_ms": round(latency_ms, 1),
        "error": error,
    }


def score_batch(
    batch: list[dict],
    judge_llm: LangchainLLMWrapper,
    judge_embeddings: LangchainEmbeddingsWrapper,
) -> list[dict]:
    """Run RAGAS on a batch and return rows annotated with metric scores."""
    # Filter out errored rows — RAGAS needs non-empty answer and contexts
    scoreable = [r for r in batch if r["answer"] and r["contexts"]]
    skipped = [r for r in batch if not (r["answer"] and r["contexts"])]

    scored = []
    if scoreable:
        ds = Dataset.from_list(
            [
                {
                    "question": r["query"],
                    "answer": r["answer"],
                    "contexts": r["contexts"],
                    "ground_truth": r["ground_truth"],
                }
                for r in scoreable
            ]
        )
        result = evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=judge_llm,
            embeddings=judge_embeddings,
            raise_exceptions=False,
        )
        result_df = result.to_pandas()

        for row, (_, scores) in zip(scoreable, result_df.iterrows()):
            row = dict(row)
            row["faithfulness"] = round(float(scores.get("faithfulness", 0.0)), 3)
            row["answer_relevancy"] = round(float(scores.get("answer_relevancy", 0.0)), 3)
            row["context_precision"] = round(float(scores.get("context_precision", 0.0)), 3)
            row["pass"] = (
                row["faithfulness"] >= PASS_THRESHOLD
                and row["answer_relevancy"] >= PASS_THRESHOLD
                and row["context_precision"] >= PASS_THRESHOLD
            )
            scored.append(row)

    for row in skipped:
        row = dict(row)
        row["faithfulness"] = 0.0
        row["answer_relevancy"] = 0.0
        row["context_precision"] = 0.0
        row["pass"] = False
        scored.append(row)

    return scored


def compute_group_stats(rows: list[dict], key: str) -> dict:
    """Aggregate RAGAS metrics and latency for a group of rows."""
    if not rows:
        return {}
    latencies = [r["latency_ms"] for r in rows]
    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)
    p50 = latencies_sorted[n // 2]
    p90 = latencies_sorted[min(int(n * 0.9), n - 1)]
    p99 = latencies_sorted[min(int(n * 0.99), n - 1)]
    return {
        "faithfulness": round(sum(r["faithfulness"] for r in rows) / n, 3),
        "answer_relevancy": round(sum(r["answer_relevancy"] for r in rows) / n, 3),
        "context_precision": round(sum(r["context_precision"] for r in rows) / n, 3),
        "pass_rate": round(sum(1 for r in rows if r["pass"]) / n, 3),
        "p50_latency_ms": round(p50, 1),
        "p90_latency_ms": round(p90, 1),
        "p99_latency_ms": round(p99, 1),
    }


def print_summary(all_rows: list[dict]) -> None:
    overall = compute_group_stats(all_rows, "overall")
    passed = sum(1 for r in all_rows if r["pass"])
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
    # Per-jurisdiction
    jurisdictions = sorted({r["jurisdiction"] or "MULTI" for r in all_rows})
    print("  By jurisdiction:")
    for jur in jurisdictions:
        rows = [r for r in all_rows if (r["jurisdiction"] or "MULTI") == jur]
        stats = compute_group_stats(rows, jur)
        print(f"    {jur:<20} faith={stats['faithfulness']:.2f}  "
              f"rel={stats['answer_relevancy']:.2f}  "
              f"prec={stats['context_precision']:.2f}  "
              f"pass={stats['pass_rate']:.0%}  p50={stats['p50_latency_ms']:.0f}ms")
    print()
    # Failures
    failures = [r for r in all_rows if not r["pass"]]
    if failures:
        print(f"  Failures ({len(failures)}):")
        for r in failures:
            print(f"    FAIL [{r['id']}]  "
                  f"faith={r['faithfulness']:.2f}  "
                  f"rel={r['answer_relevancy']:.2f}  "
                  f"prec={r['context_precision']:.2f}  "
                  f"err={r.get('error') or '-'}")
    print("=" * 72)


def main() -> None:
    dataset_path = Path(__file__).parent.parent / "eval" / "golden_dataset.jsonl"
    if not dataset_path.exists():
        print(f"ERROR: dataset not found at {dataset_path}", file=sys.stderr)
        sys.exit(1)

    rows = load_dataset(dataset_path)
    if EVAL_LIMIT:
        rows = rows[:EVAL_LIMIT]
    print(f"CivicSetu RAGAS Eval — {len(rows)} queries, batch_size={BATCH_SIZE}, delay={BATCH_DELAY_SEC}s")

    # Phase 1: run all queries through graph (no rate-limit needed here)
    print("\nPhase 1: Invoking graph for all queries...")
    graph = get_compiled_graph()
    invoked: list[dict] = []
    for i, row in enumerate(rows, 1):
        print(f"  [{i:02}/{len(rows)}] {row['id']} ...", end=" ", flush=True)
        result = invoke_graph(graph, row)
        invoked.append(result)
        status = "OK" if result["answer"] else "EMPTY"
        print(f"{status}  ({result['latency_ms']:.0f}ms, conf={result['confidence_score']})")

    # Phase 2: RAGAS scoring in batches
    print("\nPhase 2: RAGAS scoring in batches...")
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

    # Phase 3: write results
    print_summary(all_scored)

    # Build structured report
    jurisdictions = sorted({r["jurisdiction"] or "MULTI" for r in all_scored})
    query_types = sorted({r["query_type"] for r in all_scored})

    report = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "dataset_size": len(all_scored),
        "batch_size": BATCH_SIZE,
        "batch_delay_sec": BATCH_DELAY_SEC,
        "pass_threshold": PASS_THRESHOLD,
        "overall": compute_group_stats(all_scored, "overall"),
        "by_jurisdiction": {
            jur: compute_group_stats(
                [r for r in all_scored if (r["jurisdiction"] or "MULTI") == jur], jur
            )
            for jur in jurisdictions
        },
        "by_query_type": {
            qt: compute_group_stats(
                [r for r in all_scored if r["query_type"] == qt], qt
            )
            for qt in query_types
        },
        "rows": all_scored,
    }

    out = Path("eval_results.json")
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nFull results → {out}")

    # Exit 1 if any failures
    failures = sum(1 for r in all_scored if not r["pass"])
    if failures:
        print(f"{failures} row(s) below pass threshold ({PASS_THRESHOLD})")
        sys.exit(1)


if __name__ == "__main__":
    main()
