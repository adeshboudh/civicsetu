# scripts/test_e2e_queries.py
"""
End-to-end query tests across all 4 states.
Tests vector retrieval, graph retrieval, and multi-jurisdiction paths.

Run:
    uv run python scripts/test_e2e_queries.py
"""
from __future__ import annotations
import sys, json, time
import io
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import structlog
from civicsetu.agent.graph import get_compiled_graph
from civicsetu.models.enums import Jurisdiction

log = structlog.get_logger(__name__)

# ── Test cases ────────────────────────────────────────────────────────────────
# Format: (label, query, jurisdiction_filter, expected_query_type, must_contain_section)
TEST_CASES = [

    # ── Fact lookup — vector path ─────────────────────────────────────────────
    (
        "FACT_CENTRAL_01",
        "What are the obligations of a promoter under RERA?",
        None,
        "fact_lookup",
        "4",          # RERA Act Section 4 — promoter obligations
    ),
    (
        "FACT_MH_01",
        "What documents must a promoter submit for project registration in Maharashtra?",
        Jurisdiction.MAHARASHTRA,
        "fact_lookup",
        None,
    ),
    (
        "FACT_UP_01",
        "What is the timeline for refund of amount paid by allottee in Uttar Pradesh?",
        Jurisdiction.UTTAR_PRADESH,
        "fact_lookup",
        None,
    ),
    (
        "FACT_KA_01",
        "What are the grounds for revocation of project registration in Karnataka?",
        Jurisdiction.KARNATAKA,
        "fact_lookup",
        None,
    ),
    (
        "FACT_TN_01",
        "What information must a real estate agent provide to buyers in Tamil Nadu?",
        Jurisdiction.TAMIL_NADU,
        "fact_lookup",
        None,
    ),

    # ── Cross-reference — graph path ──────────────────────────────────────────
    (
        "XREF_01",
        "What does section 18 of RERA say about refund obligations?",
        None,
        "cross_reference",
        "18",
    ),
    (
        "XREF_02",
        "Which state rules implement section 9 of the RERA Act on agent registration?",
        None,
        "cross_reference",
        "9",
    ),
    (
        "XREF_MH_01",
        "Which MahaRERA rule derives from section 4 of the central RERA Act?",
        Jurisdiction.MAHARASHTRA,
        "cross_reference",
        None,
    ),

    # ── Penalty lookup — graph path ───────────────────────────────────────────
    (
        "PENALTY_01",
        "What is the penalty for non-registration of a real estate project under RERA?",
        None,
        "penalty_lookup",
        None,
    ),
    (
        "PENALTY_02",
        "What are the penalties for a promoter who fails to register under RERA?",
        None,
        "penalty_lookup",
        None,
    ),

    # ── Multi-jurisdiction comparison ─────────────────────────────────────────
    (
        "MULTI_01",
        "How does Karnataka handle extension of project registration compared to the central RERA Act?",
        None,
        "cross_reference",
        None,
    ),
    (
        "MULTI_02",
        "What is the rate of interest payable on refunds under Tamil Nadu RERA rules?",
        Jurisdiction.TAMIL_NADU,
        "fact_lookup",
        None,
    ),
]


def run_test(graph, label, query, jurisdiction_filter, expected_type, must_contain):
    start = time.time()
    state = {
        "query": query,
        "jurisdiction_filter": jurisdiction_filter,
        "top_k": 5,
        "session_id": f"e2e_{label}",
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

    try:
        result = graph.invoke(state)
        elapsed = time.time() - start

        query_type = result.get("query_type", "UNKNOWN")
        confidence = result.get("confidence_score", 0.0)
        answer = result.get("raw_response", "") or ""
        citations = result.get("citations", [])
        error = result.get("error")

        # Checks
        type_ok = expected_type in str(query_type).lower() if expected_type else True
        citation_ok = len(citations) > 0
        section_ok = True
        if must_contain:
            section_ok = any(
                must_contain == str(c.section_id) for c in citations
            ) or must_contain in answer

        status = "PASS" if (citation_ok and section_ok and not error) else "FAIL"

        return {
            "label": label,
            "status": status,
            "query_type": str(query_type),
            "type_match": type_ok,
            "confidence": round(confidence, 3),
            "citations": len(citations),
            "section_check": f"{must_contain}={'OK' if section_ok else 'MISSING'}" if must_contain else "N/A",
            "elapsed_s": round(elapsed, 2),
            "error": error,
            "answer_preview": answer[:120].replace("\n", " ") if answer else "",
        }

    except Exception as e:
        return {
            "label": label,
            "status": "ERROR",
            "query_type": "N/A",
            "type_match": False,
            "confidence": 0.0,
            "citations": 0,
            "section_check": "N/A",
            "elapsed_s": round(time.time() - start, 2),
            "error": str(e),
            "answer_preview": "",
        }


def main():
    print("Compiling LangGraph...")
    graph = get_compiled_graph()
    print(f"Running {len(TEST_CASES)} test cases...\n")

    results = []
    for label, query, jfilter, etype, must_section in TEST_CASES:
        print(f"  [{label}] ...", end=" ", flush=True)
        r = run_test(graph, label, query, jfilter, etype, must_section)
        results.append(r)
        status_icon = "PASS" if r["status"] == "PASS" else "FAIL"
        print(f"{status_icon}  ({r['elapsed_s']}s, conf={r['confidence']}, cit={r['citations']})")

    # ── Summary ───────────────────────────────────────────────────────────────
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    errors = sum(1 for r in results if r["status"] == "ERROR")
    avg_conf = sum(r["confidence"] for r in results) / len(results)
    avg_time = sum(r["elapsed_s"] for r in results) / len(results)

    print("\n" + "=" * 70)
    print(f"E2E Test Results: {passed} PASS  {failed} FAIL  {errors} ERROR")
    print(f"Avg confidence : {avg_conf:.3f}")
    print(f"Avg latency    : {avg_time:.2f}s")
    print("=" * 70)

    # Print failures detail
    for r in results:
        if r["status"] != "PASS":
            print(f"\n{r['status']}: {r['label']}")
            print(f"  query_type   : {r['query_type']}")
            print(f"  section_check: {r['section_check']}")
            print(f"  citations    : {r['citations']}")
            print(f"  error        : {r['error']}")
            print(f"  answer       : {r['answer_preview']}")

    # Save full results
    out = Path("e2e_results.json")
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nFull results → {out}")

    if failed + errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
