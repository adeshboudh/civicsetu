from __future__ import annotations

import asyncio
import json
import importlib.util
import time
from pathlib import Path
from types import SimpleNamespace
import types

import pytest
import httpx


def _load_run_eval_module():
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "run_eval.py"
    spec = importlib.util.spec_from_file_location("test_run_eval_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _SlowMetric:
    def __init__(self, *args, **kwargs):
        pass

    async def ascore(self, **kwargs):
        await asyncio.sleep(0.05)
        return SimpleNamespace(value=0.9)

    def batch_score(self, inputs):
        time.sleep(0.05)
        return [SimpleNamespace(value=0.9) for _ in inputs]


class _FailingMetric:
    def __init__(self, *args, **kwargs):
        pass

    def batch_score(self, inputs):
        raise RuntimeError("judge unavailable")


class _CaptureTransport(httpx.AsyncBaseTransport):
    def __init__(self):
        self.body = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.body = json.loads(request.content)
        return httpx.Response(200, request=request)


def test_no_reasoning_transport_adds_flag_without_request_copy():
    run_eval = _load_run_eval_module()
    wrapped = _CaptureTransport()
    transport = run_eval._NoReasoningTransport(wrapped)
    request = httpx.Request(
        "POST",
        "https://example.test/v1/chat/completions",
        headers={"content-type": "application/json"},
        content=json.dumps({"model": "x"}).encode(),
    )

    asyncio.run(transport.handle_async_request(request))

    assert wrapped.body == {"model": "x", "no_reasoning": True}


def test_score_batch_runs_without_timeout(monkeypatch):
    """score_batch has no timeout — slow metrics run to completion."""
    run_eval = _load_run_eval_module()

    assert not hasattr(run_eval, "METRIC_TIMEOUT_SEC"), (
        "METRIC_TIMEOUT_SEC should not exist — timeout was removed to allow unbounded RAGAS scoring"
    )


def test_score_batch_returns_failed_rows_on_metric_error(monkeypatch):
    run_eval = _load_run_eval_module()

    import ragas.metrics.collections as collections

    monkeypatch.setattr(collections, "Faithfulness", _FailingMetric)
    monkeypatch.setattr(collections, "AnswerRelevancy", _FailingMetric)
    monkeypatch.setattr(collections, "ContextPrecision", _FailingMetric)

    rows = [
        {
            "id": "CASE-001",
            "query": "What is section 3?",
            "answer": "A test answer",
            "contexts": ["A test context"],
            "ground_truth": "A test reference",
            "latency_ms": 10.0,
            "jurisdiction": "CENTRAL",
            "query_type": "fact",
            "error": None,
        }
    ]

    scored = run_eval.score_batch(rows, judge_llm=object(), judge_embeddings=object(), label="T")

    assert scored[0]["faithfulness"] == 0.0
    assert scored[0]["answer_relevancy"] == 0.0
    assert scored[0]["context_precision"] == 0.0
    assert scored[0]["pass"] is False
    assert "judge unavailable" in scored[0]["error"].lower()


def test_run_phase1_retries_cached_fallback_rows(monkeypatch):
    run_eval = _load_run_eval_module()
    phase1_path = Path(__file__).resolve().parents[2] / "eval_phase1_results.test.json"
    monkeypatch.setattr(run_eval, "PHASE1_OUTPUT", phase1_path)
    phase1_path.unlink(missing_ok=True)

    row = {
        "id": "CASE-001",
        "jurisdiction": "CENTRAL",
        "query_type": "fact_lookup",
        "query": "What are promoter duties?",
        "ground_truth": "Promoters have statutory duties.",
    }
    phase1_path.write_text(
        json.dumps(
            [
                {
                    **row,
                    "answer": "Unable to generate a structured response. Please try again.",
                    "contexts": ["Section text"],
                    "citations_count": 0,
                    "confidence_score": 0.0,
                    "query_type_resolved": "fact_lookup",
                    "latency_ms": 10.0,
                    "error": None,
                }
            ]
        ),
        encoding="utf-8",
    )

    fresh = {
        **row,
        "answer": "A real generated answer.",
        "contexts": ["Section text"],
        "citations_count": 1,
        "confidence_score": 0.8,
        "query_type_resolved": "fact_lookup",
        "latency_ms": 20.0,
        "error": None,
    }
    calls = []

    def fake_invoke_graph(graph, input_row):
        calls.append(input_row["id"])
        return fresh

    monkeypatch.setattr(run_eval, "invoke_graph", fake_invoke_graph)

    try:
        results = run_eval.run_phase1([row], graph=object())

        assert calls == ["CASE-001"]
        assert results == [fresh]
    finally:
        phase1_path.unlink(missing_ok=True)


def test_get_osmapi_key_accepts_legacy_osm_api_key(monkeypatch):
    run_eval = _load_run_eval_module()
    monkeypatch.delenv("OSMAPI_API_KEY", raising=False)
    monkeypatch.setenv("OSM_API_KEY", "legacy-key")

    assert run_eval._get_osmapi_key() == "legacy-key"


def test_reasoning_is_disabled_by_default(monkeypatch):
    """NO_REASONING defaults True — prevents Qwen3 thinking tokens from corrupting RAGAS JSON."""
    monkeypatch.delenv("NO_REASONING", raising=False)
    run_eval = _load_run_eval_module()

    assert run_eval.NO_REASONING is True


def test_build_judge_does_not_pass_reasoning_effort(monkeypatch):
    """reasoning_effort must NOT be passed — osmapi rejects it for non-o-series models."""
    monkeypatch.setenv("OSM_API_KEY", "legacy-key")
    monkeypatch.setenv("GEMINI_API_KEY_2", "gemini-key")
    run_eval = _load_run_eval_module()

    captured = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            captured["openai_kwargs"] = kwargs

    def fake_llm_factory(model, **kwargs):
        captured["llm_factory_model"] = model
        captured["llm_factory_kwargs"] = kwargs
        return "judge-llm"

    class FakeGoogleEmbeddings:
        def __init__(self, **kwargs):
            captured["embeddings_kwargs"] = kwargs

    class FakeGenAIClient:
        def __init__(self, **kwargs):
            captured["genai_kwargs"] = kwargs

    import openai
    import ragas.llms
    import ragas.embeddings
    import google

    monkeypatch.setattr(openai, "AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.setattr(ragas.llms, "llm_factory", fake_llm_factory)
    monkeypatch.setattr(ragas.embeddings, "GoogleEmbeddings", FakeGoogleEmbeddings)
    monkeypatch.setattr(google, "genai", types.SimpleNamespace(Client=FakeGenAIClient), raising=False)

    judge_llm, judge_embeddings = run_eval.build_judge()

    assert judge_llm == "judge-llm"
    assert isinstance(judge_embeddings, FakeGoogleEmbeddings)
    assert captured["llm_factory_model"] == run_eval.JUDGE_MODEL
    assert "reasoning_effort" not in captured["llm_factory_kwargs"]


def test_retry_delay_seconds_extracts_provider_hint():
    run_eval = _load_run_eval_module()

    error = "quota exceeded. Please retry in 46.62650982s."

    assert run_eval._retry_delay_seconds(error) == 47


def test_retry_delay_seconds_has_reasonable_floor_for_quota_errors():
    run_eval = _load_run_eval_module()

    error = "RESOURCE_EXHAUSTED quota exceeded for input_token_count"

    assert run_eval._retry_delay_seconds(error) == 60


def test_prepare_metric_row_truncates_long_fields(monkeypatch):
    run_eval = _load_run_eval_module()
    monkeypatch.setattr(run_eval, "RAGAS_MAX_CONTEXTS", 2, raising=False)
    monkeypatch.setattr(run_eval, "RAGAS_CONTEXT_CHAR_LIMIT", 5, raising=False)
    monkeypatch.setattr(run_eval, "RAGAS_ANSWER_CHAR_LIMIT", 6, raising=False)
    monkeypatch.setattr(run_eval, "RAGAS_REFERENCE_CHAR_LIMIT", 7, raising=False)

    row = {
        "query": "What is section 3?",
        "answer": "ABCDEFGHIJK",
        "contexts": ["123456", "abcdef", "zzzzzz"],
        "ground_truth": "reference-text",
    }

    prepared = run_eval._prepare_metric_row(row)

    assert prepared["answer"] == "ABCDEF"
    assert prepared["contexts"] == ["12345", "abcde"]
    assert prepared["ground_truth"] == "referen"


def test_run_phase2_reuses_completed_checkpoint_rows(monkeypatch):
    run_eval = _load_run_eval_module()
    output_path = Path(__file__).resolve().parents[2] / "eval_results.test.json"
    monkeypatch.setattr(run_eval, "OUTPUT_PATH", output_path)
    output_path.unlink(missing_ok=True)

    row = {
        "id": "CASE-001",
        "jurisdiction": "CENTRAL",
        "query_type": "fact_lookup",
        "query": "What are promoter duties?",
        "ground_truth": "Promoters have statutory duties.",
        "answer": "A real generated answer.",
        "contexts": ["Section text"],
        "citations_count": 1,
        "confidence_score": 0.8,
        "query_type_resolved": "fact_lookup",
        "latency_ms": 20.0,
        "error": None,
    }
    scored = {
        **row,
        "faithfulness": 0.9,
        "answer_relevancy": 0.8,
        "context_precision": 0.7,
        "pass": True,
    }

    output_path.write_text(
        json.dumps({"rows": [scored]}, indent=2),
        encoding="utf-8",
    )

    def fail_build_judge():
        raise AssertionError("judge should not be built when checkpoint is reusable")

    monkeypatch.setattr(run_eval, "build_judge", fail_build_judge)

    try:
        results = run_eval.run_phase2([row], judge_llm=None, judge_embeddings=None)
        assert results == [scored]
    finally:
        output_path.unlink(missing_ok=True)
