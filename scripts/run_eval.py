"""
CivicSetu RAGAS evaluation — CLI entry point.

  Phase 1: Invoke RAG graph for every query → save to eval_phase1_results.json
  Phase 2: Score results with RAGAS (Faithfulness, AnswerRelevancy, ContextPrecision)

Usage:
    uv run python scripts/run_eval.py
    EVAL_LIMIT=5 uv run python scripts/run_eval.py          # quick smoke-test

Graph LLM:
    Uses the normal app routing from settings.py / .env for phase 1 generation.
    PRIMARY_MODEL, FALLBACK_MODEL_1, FALLBACK_MODEL_2, and FALLBACK_MODEL_3
    are read by civicsetu.agent.nodes when the graph is imported.

Judge (RAGAS scorer) model:
    # Default judge is Groq llama-3.3-70b-versatile via GROQ_API_KEY_2
    uv run python scripts/run_eval.py
    JUDGE_PROVIDER=groq JUDGE_MODEL=llama-3.3-70b-versatile uv run python scripts/run_eval.py
    JUDGE_PROVIDER=gemini JUDGE_MODEL=gemini/gemini-2.5-flash-lite uv run python scripts/run_eval.py
    JUDGE_PROVIDER=openrouter JUDGE_MODEL=nvidia/nemotron-3-super-120b-a12b:free uv run python scripts/run_eval.py
    JUDGE_PROVIDER=osmapi JUDGE_MODEL=qwen3.5-397b-a17b uv run python scripts/run_eval.py
    JUDGE_PROVIDER=nvidia JUDGE_MODEL=z-ai/glm4.7 uv run python scripts/run_eval.py
    JUDGE_GEMINI_API_KEY=<key> JUDGE_PROVIDER=gemini JUDGE_MODEL=gemini/gemma-4-31b-it uv run python scripts/run_eval.py

Resume after phase 2 failure (phase 1 cached in eval_phase1_results.json):
    uv run python scripts/run_eval.py
    # Phase 1 prints "all N rows loaded from cache" and is skipped

Force re-run phase 1:
    rm eval_phase1_results.json && uv run python scripts/run_eval.py

Run only one phase:
    EVAL_PHASE=1 uv run python scripts/run_eval.py   # graph invocation only
    EVAL_PHASE=2 uv run python scripts/run_eval.py   # RAGAS scoring only (requires phase 1 cache)

Disable no_reasoning (not recommended for Qwen3 thinking models):
    NO_REASONING=false uv run python scripts/run_eval.py

All logic lives in: src/civicsetu/evaluation/ragas_eval.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# Ensure src/ is on path when running the script directly (outside of uv run)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from civicsetu.evaluation.ragas_eval import main  # noqa: E402

if __name__ == "__main__":
    main()
