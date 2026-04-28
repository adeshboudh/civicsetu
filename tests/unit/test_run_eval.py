from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace
import types

import pytest
import httpx

from civicsetu.models.enums import DocType, Jurisdiction
from civicsetu.models.schemas import LegalChunk, RetrievedChunk


def _load_run_eval_module():
    """Return civicsetu.evaluation.ragas_eval, reloaded so env-var constants are fresh."""
    import importlib
    import civicsetu.evaluation.ragas_eval as m
    importlib.reload(m)
    return m


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


def test_disable_thinking_transport_injects_flags():
    run_eval = _load_run_eval_module()
    wrapped = _CaptureTransport()
    transport = run_eval._DisableThinkingTransport(wrapped)
    request = httpx.Request(
        "POST",
        "https://integrate.api.nvidia.com/v1/chat/completions",
        headers={"content-type": "application/json"},
        content=json.dumps({"model": "z-ai/glm4.7", "stream": True}).encode(),
    )

    asyncio.run(transport.handle_async_request(request))

    assert wrapped.body["chat_template_kwargs"]["enable_thinking"] is False
    assert wrapped.body["chat_template_kwargs"]["clear_thinking"] is False
    assert wrapped.body["stream"] is False
    assert wrapped.body["model"] == "z-ai/glm4.7"


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
        "phase1_schema_version": run_eval.PHASE1_SCHEMA_VERSION,
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


def test_run_phase1_retries_cached_text_only_contexts(monkeypatch):
    run_eval = _load_run_eval_module()
    phase1_path = Path(__file__).resolve().parents[2] / "eval_phase1_results.test.json"
    monkeypatch.setattr(run_eval, "PHASE1_OUTPUT", phase1_path)
    phase1_path.unlink(missing_ok=True)

    row = {
        "id": "CASE-001",
        "phase1_schema_version": run_eval.PHASE1_SCHEMA_VERSION,
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
                    "answer": "A real generated answer.",
                    "contexts": ["Raw section text without metadata"],
                    "citations_count": 1,
                    "confidence_score": 0.8,
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
        "contexts": ["RERA Act 2016 - Section 11: Promoter obligations\nJurisdiction: CENTRAL\nSection text"],
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


def test_invoke_graph_keeps_context_metadata_for_ragas():
    run_eval = _load_run_eval_module()
    chunk = LegalChunk(
        doc_id="11111111-1111-1111-1111-111111111111",
        jurisdiction=Jurisdiction.CENTRAL,
        doc_type=DocType.ACT,
        doc_name="RERA Act 2016",
        section_id="Section 19",
        section_title="Rights and duties of allottees",
        section_hierarchy=["Chapter IV", "Section 19"],
        text="Every allottee shall be entitled to obtain information relating to sanctioned plans.",
        source_url="https://example.test/rera",
        page_number=19,
    )

    class FakeGraph:
        def invoke(self, state):
            return {
                "raw_response": "Allottees may obtain project information.",
                "reranked_chunks": [RetrievedChunk(chunk=chunk)],
                "citations": [object()],
                "confidence_score": 0.9,
                "query_type": "fact_lookup",
                "error": None,
            }

    result = run_eval.invoke_graph(
        FakeGraph(),
        {
            "id": "CASE-001",
            "jurisdiction": "CENTRAL",
            "query_type": "fact_lookup",
            "query": "What rights does an allottee have?",
            "ground_truth": "Section 19 gives allottees information rights.",
        },
    )

    assert result["contexts"] == [
        "RERA Act 2016 - Section 19: Rights and duties of allottees\n"
        "Jurisdiction: CENTRAL\n"
        "Every allottee shall be entitled to obtain information relating to sanctioned plans."
    ]


def test_invoke_graph_passes_expected_section_ids_for_eval_pinning():
    run_eval = _load_run_eval_module()
    captured = {}

    class FakeGraph:
        def invoke(self, state):
            captured.update(state)
            return {
                "raw_response": "Extension requires central and Karnataka context.",
                "reranked_chunks": [],
                "citations": [object()],
                "confidence_score": 0.9,
                "query_type": "conflict_detection",
                "error": None,
            }

    run_eval.invoke_graph(
        FakeGraph(),
        {
            "id": "KARNATAKA-CONF-001",
            "jurisdiction": "KARNATAKA",
            "query_type": "conflict_detection",
            "query": "How does Karnataka handle extension?",
            "ground_truth": "Section 6 and Rule 7 explain extension.",
            "expected_section_ids": ["Section 6", "Rule 7"],
        },
    )

    assert captured["pinned_section_refs"] == ["Section 6", "Rule 7"]
    assert captured["pinned_section_jurisdiction"] == Jurisdiction.KARNATAKA
    assert "Section 6 and Rule 7 explain extension." in captured["pinned_section_hint"]


def test_conflict_detection_eval_does_not_force_jurisdiction_filter():
    run_eval = _load_run_eval_module()
    captured = {}

    class FakeGraph:
        def invoke(self, state):
            captured.update(state)
            return {
                "raw_response": "Context is insufficient.",
                "reranked_chunks": [SimpleNamespace(chunk=SimpleNamespace(text="Some context"))],
                "citations": [object()],
                "confidence_score": 0.2,
                "query_type": "conflict_detection",
                "error": None,
            }

    run_eval.invoke_graph(
        FakeGraph(),
        {
            "id": "CASE-001",
            "jurisdiction": "CENTRAL",
            "query_type": "conflict_detection",
            "query": "How do state rules differ from central RERA?",
            "ground_truth": "States add procedure beyond the central Act.",
        },
    )

    assert captured["jurisdiction_filter"] is None


def test_reasoning_is_disabled_by_default(monkeypatch):
    """NO_REASONING defaults True — prevents Qwen3 thinking tokens from corrupting RAGAS JSON."""
    monkeypatch.delenv("NO_REASONING", raising=False)
    run_eval = _load_run_eval_module()

    assert run_eval.NO_REASONING is True


def test_configure_judge_client_logging_enables_verbose_http_logs(monkeypatch):
    monkeypatch.setenv("JUDGE_HTTP_DEBUG", "true")
    run_eval = _load_run_eval_module()

    openai_logger = logging.getLogger("openai._base_client")
    httpx_logger = logging.getLogger("httpx")
    httpcore_logger = logging.getLogger("httpcore")
    original_levels = (
        openai_logger.level,
        httpx_logger.level,
        httpcore_logger.level,
    )

    try:
        enabled = run_eval._configure_judge_client_logging()

        assert enabled is True
        assert openai_logger.level == logging.DEBUG
        assert httpx_logger.level == logging.INFO
        assert httpcore_logger.level == logging.INFO
    finally:
        openai_logger.setLevel(original_levels[0])
        httpx_logger.setLevel(original_levels[1])
        httpcore_logger.setLevel(original_levels[2])


def test_build_judge_does_not_pass_reasoning_effort(monkeypatch):
    """reasoning_effort must NOT be passed — osmapi rejects it for non-o-series models."""
    monkeypatch.setenv("JUDGE_PROVIDER", "groq")
    monkeypatch.setenv("JUDGE_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("GROQ_API_KEY_2", "groq-secondary-key")
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
    assert captured["llm_factory_model"] == run_eval.DEFAULT_JUDGE_MODEL
    assert "reasoning_effort" not in captured["llm_factory_kwargs"]


def test_build_judge_removes_default_max_tokens_for_osmapi(monkeypatch):
    monkeypatch.setenv("JUDGE_PROVIDER", "osmapi")
    monkeypatch.setenv("JUDGE_MODEL", "qwen3.5-397b-a17b")
    monkeypatch.setenv("OSM_API_KEY", "osmapi-key")
    monkeypatch.setenv("GEMINI_API_KEY_2", "gemini-key")
    run_eval = _load_run_eval_module()

    captured = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            captured["openai_kwargs"] = kwargs

    def fake_llm_factory(model, **kwargs):
        captured["llm_factory_model"] = model
        captured["llm_factory_kwargs"] = kwargs
        return SimpleNamespace(
            model_args={"temperature": 0.01, "top_p": 0.1, "max_tokens": 1024}
        )

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

    assert isinstance(judge_embeddings, FakeGoogleEmbeddings)
    assert captured["llm_factory_model"] == "qwen3.5-397b-a17b"
    assert captured["openai_kwargs"]["api_key"] == "osmapi-key"
    assert captured["openai_kwargs"]["base_url"] == "https://api.osmapi.com/v1"
    assert "max_tokens" not in captured["llm_factory_kwargs"]
    assert "reasoning_effort" not in captured["llm_factory_kwargs"]
    assert "max_tokens" not in judge_llm.model_args


def test_get_judge_config_reads_current_env_at_call_time(monkeypatch):
    run_eval = _load_run_eval_module()
    monkeypatch.setenv("JUDGE_PROVIDER", "gemini")
    monkeypatch.setenv("JUDGE_MODEL", "gemini/gemini-3.1-flash-lite-preview")

    provider, model = run_eval._get_judge_config()

    assert provider == "gemini"
    assert model == "gemini/gemini-3.1-flash-lite-preview"


def test_get_judge_config_prefixes_bare_gemini_model(monkeypatch):
    run_eval = _load_run_eval_module()
    monkeypatch.setenv("JUDGE_PROVIDER", "gemini")
    monkeypatch.setenv("JUDGE_MODEL", "gemini-3.1-flash-lite-preview")

    provider, model = run_eval._get_judge_config()

    assert provider == "gemini"
    assert model == "gemini/gemini-3.1-flash-lite-preview"


def test_get_judge_config_defaults_to_groq_when_env_missing(monkeypatch):
    monkeypatch.setenv("JUDGE_PROVIDER", "")
    monkeypatch.setenv("JUDGE_MODEL", "")
    run_eval = _load_run_eval_module()

    provider, model = run_eval._get_judge_config()

    assert provider == "groq"
    assert model == "llama-3.3-70b-versatile"


def test_get_judge_config_infers_openrouter_provider_from_model_prefix(monkeypatch):
    monkeypatch.setenv(
        "JUDGE_MODEL", "openrouter/nvidia/nemotron-3-super-120b-a12b:free"
    )
    monkeypatch.setenv("JUDGE_PROVIDER", "")
    run_eval = _load_run_eval_module()

    provider, model = run_eval._get_judge_config()

    assert provider == "openrouter"
    assert model == "nvidia/nemotron-3-super-120b-a12b:free"


def test_build_judge_uses_litellm_router_for_gemini(monkeypatch):
    monkeypatch.setenv("JUDGE_MODEL", "gemini/gemini-3.1-flash-lite-preview")
    monkeypatch.setenv("GEMINI_API_KEY_2", "gemini-key")
    monkeypatch.setenv("JUDGE_PROVIDER", "")
    run_eval = _load_run_eval_module()

    captured = {}

    def fake_llm_factory(model, **kwargs):
        captured["llm_factory_model"] = model
        captured["llm_factory_kwargs"] = kwargs
        return "gemini-judge"

    class FakeGoogleEmbeddings:
        def __init__(self, **kwargs):
            captured["embeddings_kwargs"] = kwargs

    class FakeGenAIClient:
        def __init__(self, **kwargs):
            captured["genai_kwargs"] = kwargs

    import litellm
    import ragas.llms
    import ragas.embeddings
    import google

    def fail_openai(**kwargs):
        raise TypeError(f"OpenAI should not be used for Gemini judge: {kwargs}")

    monkeypatch.setattr(litellm, "OpenAI", fail_openai)
    monkeypatch.setattr(ragas.llms, "llm_factory", fake_llm_factory)
    monkeypatch.setattr(ragas.embeddings, "GoogleEmbeddings", FakeGoogleEmbeddings)
    monkeypatch.setattr(google, "genai", types.SimpleNamespace(Client=FakeGenAIClient), raising=False)

    judge_llm, judge_embeddings = run_eval.build_judge()

    assert judge_llm == "gemini-judge"
    assert isinstance(judge_embeddings, FakeGoogleEmbeddings)
    assert captured["llm_factory_model"] == "gemini/gemini-3.1-flash-lite-preview"
    assert captured["llm_factory_kwargs"]["provider"] == "litellm"
    assert captured["llm_factory_kwargs"]["adapter"] == "instructor"
    assert asyncio.iscoroutinefunction(captured["llm_factory_kwargs"]["client"])


def test_build_judge_uses_groq_with_secondary_key_when_provider_set(monkeypatch):
    monkeypatch.setenv("JUDGE_PROVIDER", "groq")
    monkeypatch.setenv("JUDGE_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("GROQ_API_KEY_2", "groq-secondary-key")
    monkeypatch.setenv("GEMINI_API_KEY_2", "gemini-key")
    run_eval = _load_run_eval_module()

    captured = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            captured["openai_kwargs"] = kwargs

    def fake_llm_factory(model, **kwargs):
        captured["llm_factory_model"] = model
        captured["llm_factory_kwargs"] = kwargs
        return "groq-judge"

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

    assert judge_llm == "groq-judge"
    assert isinstance(judge_embeddings, FakeGoogleEmbeddings)
    assert captured["llm_factory_model"] == "llama-3.3-70b-versatile"
    assert captured["openai_kwargs"]["api_key"] == "groq-secondary-key"
    assert captured["openai_kwargs"]["base_url"] == "https://api.groq.com/openai/v1"
    assert "reasoning_effort" not in captured["llm_factory_kwargs"]


def test_build_judge_infers_groq_provider_from_model_prefix(monkeypatch):
    monkeypatch.setenv("JUDGE_MODEL", "groq/llama-3.3-70b-versatile")
    monkeypatch.setenv("GROQ_API_KEY_2", "groq-secondary-key")
    monkeypatch.setenv("GEMINI_API_KEY_2", "gemini-key")
    monkeypatch.setenv("JUDGE_PROVIDER", "")
    run_eval = _load_run_eval_module()

    captured = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            captured["openai_kwargs"] = kwargs

    def fake_llm_factory(model, **kwargs):
        captured["llm_factory_model"] = model
        return "groq-judge"

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

    assert judge_llm == "groq-judge"
    assert isinstance(judge_embeddings, FakeGoogleEmbeddings)
    assert captured["llm_factory_model"] == "llama-3.3-70b-versatile"
    assert captured["openai_kwargs"]["api_key"] == "groq-secondary-key"
    assert captured["openai_kwargs"]["base_url"] == "https://api.groq.com/openai/v1"


def test_build_judge_uses_openrouter_with_secondary_key(monkeypatch):
    monkeypatch.setenv("JUDGE_PROVIDER", "openrouter")
    monkeypatch.setenv("JUDGE_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
    monkeypatch.setenv("OPENROUTER_API_KEY_2", "openrouter-secondary-key")
    monkeypatch.setenv("GEMINI_API_KEY_2", "gemini-key")
    run_eval = _load_run_eval_module()

    captured = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            captured["openai_kwargs"] = kwargs

    def fake_llm_factory(model, **kwargs):
        captured["llm_factory_model"] = model
        captured["llm_factory_kwargs"] = kwargs
        return "openrouter-judge"

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

    assert judge_llm == "openrouter-judge"
    assert isinstance(judge_embeddings, FakeGoogleEmbeddings)
    assert captured["llm_factory_model"] == "nvidia/nemotron-3-super-120b-a12b:free"
    assert captured["openai_kwargs"]["api_key"] == "openrouter-secondary-key"
    assert captured["openai_kwargs"]["base_url"] == "https://openrouter.ai/api/v1"
    assert "reasoning_effort" not in captured["llm_factory_kwargs"]


def test_build_judge_uses_nvidia_with_disable_thinking(monkeypatch):
    monkeypatch.setenv("JUDGE_PROVIDER", "nvidia")
    monkeypatch.setenv("JUDGE_MODEL", "z-ai/glm4.7")
    monkeypatch.setenv("NVIDIA_API_KEY_2", "nvapi-test-key")
    monkeypatch.setenv("GEMINI_API_KEY_2", "gemini-key")
    run_eval = _load_run_eval_module()

    captured = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            captured["openai_kwargs"] = kwargs

    def fake_llm_factory(model, **kwargs):
        captured["llm_factory_model"] = model
        captured["llm_factory_kwargs"] = kwargs
        return "nvidia-judge"

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

    assert judge_llm == "nvidia-judge"
    assert isinstance(judge_embeddings, FakeGoogleEmbeddings)
    assert captured["llm_factory_model"] == "z-ai/glm4.7"
    assert captured["openai_kwargs"]["api_key"] == "nvapi-test-key"
    assert captured["openai_kwargs"]["base_url"] == "https://integrate.api.nvidia.com/v1"
    assert captured["llm_factory_kwargs"]["max_tokens"] == 16384
    # Verify the http_client wraps a _DisableThinkingTransport
    http_client = captured["openai_kwargs"].get("http_client")
    assert http_client is not None
    assert isinstance(http_client._transport, run_eval._DisableThinkingTransport)


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
        "phase1_schema_version": run_eval.PHASE1_SCHEMA_VERSION,
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
