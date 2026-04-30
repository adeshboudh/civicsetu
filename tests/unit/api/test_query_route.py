from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from civicsetu.models.enums import Jurisdiction, QueryType
from civicsetu.models.schemas import Citation
from tests.conftest import _base_state


@pytest.fixture(autouse=True)
def reset_graph_topology_cache():
    from civicsetu.api.routes import graph as graph_routes

    graph_routes._topo_cache["data"] = None
    graph_routes._topo_cache["ts"] = 0.0


def _make_citation(section_id: str = "18") -> Citation:
    return Citation(
        section_id=section_id,
        doc_name="RERA Act 2016",
        jurisdiction=Jurisdiction.CENTRAL,
        effective_date=date(2016, 5, 1),
        source_url="https://example.com/rera.pdf",
        chunk_id=uuid.uuid4(),
    )


@pytest.fixture
def client():
    with patch("civicsetu.agent.graph.get_compiled_graph") as mock_graph:
        mock_compiled = MagicMock()
        mock_graph.return_value = mock_compiled
        fake_checkpointer = AsyncMock()

        @asynccontextmanager
        async def fake_checkpointer_context():
            yield fake_checkpointer

        with patch("civicsetu.api.main.create_checkpointer", return_value=fake_checkpointer_context()):
            from civicsetu.api.main import create_app
            app = create_app()
            app.state.graph = mock_compiled

            with TestClient(app) as c:
                yield c, mock_compiled


# ── POST /api/v1/query ────────────────────────────────────────────────────────

def test_app_startup_warms_reranker_from_retrieval_module():
    fake_checkpointer = AsyncMock()

    @asynccontextmanager
    async def fake_checkpointer_context():
        yield fake_checkpointer

    with patch("civicsetu.api.main.create_checkpointer", return_value=fake_checkpointer_context()), patch(
        "civicsetu.agent.graph.get_compiled_graph", return_value=MagicMock()
    ), patch("civicsetu.api.main.get_driver", new=AsyncMock()), patch(
        "civicsetu.api.main.close_driver", new=AsyncMock()
    ), patch("civicsetu.retrieval.warm_embedding_model"), patch(
        "civicsetu.retrieval.reranker._get_ranker"
    ) as mock_get_ranker:
        from civicsetu.api.main import create_app

        app = create_app()

        with TestClient(app):
            pass

    mock_get_ranker.assert_called_once()


def test_app_startup_on_non_windows_does_not_shadow_asyncio():
    fake_checkpointer = AsyncMock()

    @asynccontextmanager
    async def fake_checkpointer_context():
        yield fake_checkpointer

    with patch("civicsetu.api.main.sys.platform", "linux"), patch(
        "civicsetu.api.main.create_checkpointer", return_value=fake_checkpointer_context()
    ), patch("civicsetu.agent.graph.get_compiled_graph", return_value=MagicMock()), patch(
        "civicsetu.api.main.get_driver", new=AsyncMock()
    ), patch("civicsetu.api.main.close_driver", new=AsyncMock()), patch(
        "civicsetu.retrieval.warm_embedding_model"
    ) as mock_warm_embedding_model, patch(
        "civicsetu.retrieval.reranker._get_ranker"
    ) as mock_get_ranker:
        from civicsetu.api.main import create_app

        app = create_app()

        with TestClient(app):
            pass

    mock_warm_embedding_model.assert_called_once()
    mock_get_ranker.assert_called_once()


def test_root_returns_ok_when_frontend_not_served(client):
    test_client, _ = client

    response = test_client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_graph_topology_returns_empty_payload_when_neo4j_auth_fails(client):
    test_client, _ = client

    with patch(
        "civicsetu.api.routes.graph.GraphStore.get_topology",
        new=AsyncMock(side_effect=RuntimeError("Neo.ClientError.Security.Unauthorized")),
    ), patch(
        "civicsetu.api.routes.graph.GraphStore.graph_stats",
        new=AsyncMock(side_effect=RuntimeError("Neo.ClientError.Security.Unauthorized")),
    ):
        response = test_client.get("/api/v1/graph/topology")

    assert response.status_code == 200
    assert response.json() == {
        "nodes": [],
        "edges": [],
        "stats": {
            "docs": 0,
            "sections": 0,
            "refs": 0,
            "has_sec": 0,
            "derived_from": 0,
        },
    }


def test_query_returns_200_with_citations(client):
    test_client, mock_graph = client
    mock_graph.invoke.return_value = {
        "raw_response": "Under Section 18, the promoter must...",
        "citations": [_make_citation("18")],
        "confidence_score": 0.9,
        "query_type": QueryType.CROSS_REFERENCE,
        "conflict_warnings": [],
        "amendment_notice": None,
    }

    response = test_client.post("/api/v1/query", json={"query": "What does Section 18 say?"})
    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    assert len(body["citations"]) == 1
    assert body["citations"][0]["section_id"] == "18"
    assert "session_id" in body


def test_query_returns_insufficient_when_no_citations(client):
    test_client, mock_graph = client
    mock_graph.invoke.return_value = {
        "raw_response": "I don't know",
        "citations": [],
        "confidence_score": 0.2,
        "query_type": QueryType.FACT_LOOKUP,
        "conflict_warnings": [],
        "amendment_notice": None,
    }

    response = test_client.post("/api/v1/query", json={"query": "Some obscure question here"})
    assert response.status_code == 200
    body = response.json()
    assert "Insufficient" in body["answer"]


def test_query_rejects_short_query(client):
    test_client, _ = client
    response = test_client.post("/api/v1/query", json={"query": "hi"})
    assert response.status_code == 422


def test_query_rejects_top_k_out_of_range(client):
    test_client, _ = client
    response = test_client.post(
        "/api/v1/query",
        json={"query": "What does Section 18 say?", "top_k": 25},
    )
    assert response.status_code == 422


def test_query_response_always_has_disclaimer(client):
    test_client, mock_graph = client
    mock_graph.invoke.return_value = {
        "raw_response": "Under Section 18...",
        "citations": [_make_citation()],
        "confidence_score": 0.8,
        "query_type": QueryType.FACT_LOOKUP,
        "conflict_warnings": [],
        "amendment_notice": None,
    }

    response = test_client.post("/api/v1/query", json={"query": "What does Section 18 say?"})
    body = response.json()
    assert "disclaimer" in body
    assert len(body["disclaimer"]) > 0


def test_query_reuses_provided_session_id(client):
    test_client, mock_graph = client
    mock_graph.invoke.return_value = {
        "raw_response": "Under Section 18...",
        "citations": [_make_citation()],
        "confidence_score": 0.8,
        "query_type": QueryType.FACT_LOOKUP,
        "conflict_warnings": [],
        "amendment_notice": None,
    }

    response = test_client.post(
        "/api/v1/query",
        json={"query": "What does Section 18 say?", "session_id": "my-session-abc"},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["session_id"] == "my-session-abc"
    _, config = mock_graph.invoke.call_args.args
    assert config["configurable"]["thread_id"] == "my-session-abc"


def test_graph_topology_returns_nodes_and_edges(client):
    test_client, _ = client

    with patch("civicsetu.api.routes.graph.GraphStore.get_topology", new=AsyncMock(return_value=(
        [
            {
                "chunk_id": "chunk-18",
                "section_id": "18",
                "title": "Return of amount and compensation",
                "jurisdiction": "CENTRAL",
                "doc_name": "RERA Act 2016",
                "is_active": True,
                "connection_count": 3,
            }
        ],
        [
            {
                "source": "chunk-18",
                "target": "chunk-19",
                "edge_type": "REFERENCES",
            }
        ],
    ))), patch(
        "civicsetu.api.routes.graph.GraphStore.graph_stats",
        new=AsyncMock(return_value={"sections": 10, "refs": 4, "derived_from": 2}),
    ):
        response = test_client.get("/api/v1/graph/topology")

    assert response.status_code == 200
    body = response.json()
    assert body["nodes"][0]["section_id"] == "18"
    assert body["edges"][0]["edge_type"] == "REFERENCES"
    assert body["stats"]["sections"] == 10


def test_graph_topology_uses_cache(client):
    test_client, _ = client
    with patch("civicsetu.api.routes.graph.GraphStore.get_topology", new=AsyncMock(return_value=([], []))) as mock_topology, patch(
        "civicsetu.api.routes.graph.GraphStore.graph_stats",
        new=AsyncMock(return_value={"sections": 0}),
    ):
        first = test_client.get("/api/v1/graph/topology")
        second = test_client.get("/api/v1/graph/topology")

    assert first.status_code == 200
    assert second.status_code == 200
    assert mock_topology.await_count == 1


def test_graph_section_content_returns_stitched_chunks_and_connections(client):
    test_client, _ = client
    fake_rows = [
        {
            "chunk_id": "chunk-18-a",
            "section_title": "Return of amount and compensation",
            "text": "First chunk.",
            "page_number": 1,
            "doc_name": "RERA Act 2016",
            "jurisdiction": "CENTRAL",
            "effective_date": date(2016, 5, 1),
            "source_url": "https://example.com/rera.pdf",
        },
        {
            "chunk_id": "chunk-18-b",
            "section_title": "Return of amount and compensation",
            "text": "Second chunk.",
            "page_number": 2,
            "doc_name": "RERA Act 2016",
            "jurisdiction": "CENTRAL",
            "effective_date": date(2016, 5, 1),
            "source_url": "https://example.com/rera.pdf",
        },
    ]

    mock_result = MagicMock()
    mock_result.mappings.return_value.all.return_value = fake_rows
    mock_db = AsyncMock()
    mock_db.execute.return_value = mock_result

    @asynccontextmanager
    async def fake_session():
        yield mock_db

    with patch("civicsetu.api.routes.graph.AsyncSessionLocal", side_effect=fake_session), patch(
        "civicsetu.api.routes.graph.GraphStore.get_referenced_sections",
        new=AsyncMock(return_value=[{"section_id": "19", "title": "Rights", "jurisdiction": "CENTRAL"}]),
    ), patch(
        "civicsetu.api.routes.graph.GraphStore.get_sections_referencing",
        new=AsyncMock(return_value=[]),
    ), patch(
        "civicsetu.api.routes.graph.GraphStore.get_derived_act_sections",
        new=AsyncMock(return_value=[]),
    ), patch(
        "civicsetu.api.routes.graph.GraphStore.get_deriving_rule_sections",
        new=AsyncMock(return_value=[]),
    ):
        response = test_client.get("/api/v1/graph/section/18?jurisdiction=CENTRAL")

    assert response.status_code == 200
    body = response.json()
    assert [chunk["page_number"] for chunk in body["chunks"]] == [1, 2]
    assert body["connected_sections"][0]["edge_type"] == "REFERENCES_OUT"
    assert body["source_url"] == "https://example.com/rera.pdf"


def test_graph_section_content_resolves_graph_chunk_id_case_insensitive_status(client):
    test_client, _ = client

    node_result = MagicMock()
    node_result.mappings.return_value.first.return_value = {
        "section_id": "4",
        "jurisdiction": "CENTRAL",
    }
    chunks_result = MagicMock()
    chunks_result.mappings.return_value.all.return_value = [
        {
            "chunk_id": "postgres-chunk-4",
            "section_id": "4",
            "section_title": "Prior registration of real estate project",
            "text": "Section text.",
            "page_number": 1,
            "doc_name": "RERA Act 2016",
            "jurisdiction": "CENTRAL",
            "effective_date": date(2016, 5, 1),
            "source_url": "https://example.com/rera.pdf",
        }
    ]
    mock_db = AsyncMock()
    mock_db.execute.side_effect = [node_result, chunks_result]

    @asynccontextmanager
    async def fake_session():
        yield mock_db

    with patch("civicsetu.api.routes.graph.AsyncSessionLocal", side_effect=fake_session), patch(
        "civicsetu.api.routes.graph.GraphStore.get_referenced_sections",
        new=AsyncMock(return_value=[]),
    ), patch(
        "civicsetu.api.routes.graph.GraphStore.get_sections_referencing",
        new=AsyncMock(return_value=[]),
    ), patch(
        "civicsetu.api.routes.graph.GraphStore.get_derived_act_sections",
        new=AsyncMock(return_value=[]),
    ), patch(
        "civicsetu.api.routes.graph.GraphStore.get_deriving_rule_sections",
        new=AsyncMock(return_value=[]),
    ):
        response = test_client.get(
            "/api/v1/graph/section/4?jurisdiction=CENTRAL&chunk_id=72ecb226-2aae-4c6c-bd23-5dc3a9439642"
        )

    assert response.status_code == 200
    assert response.json()["chunks"][0]["chunk_id"] == "postgres-chunk-4"
    first_lookup_sql = str(mock_db.execute.await_args_list[0].args[0])
    section_lookup_sql = str(mock_db.execute.await_args_list[1].args[0])
    assert "lower(status) = 'active'" in first_lookup_sql
    assert "lower(status) = 'active'" in section_lookup_sql


def test_section_context_query_sets_skip_classifier_and_source_section(client):
    test_client, mock_graph = client
    mock_graph.invoke.return_value = {
        "raw_response": "Section 18 requires refund with interest.",
        "citations": [_make_citation("18")],
        "confidence_score": 0.88,
        "query_type": QueryType.CROSS_REFERENCE,
        "conflict_warnings": [],
        "amendment_notice": None,
    }

    response = test_client.post(
        "/api/v1/query/section-context",
        json={
            "query": "Explain this section",
            "section_id": "18",
            "jurisdiction": "CENTRAL",
            "session_id": "section-thread-1",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == "section-thread-1"
    initial_state, config = mock_graph.invoke.call_args.args
    assert initial_state["skip_classifier"] is True
    assert initial_state["source_section_id"] == "18"
    assert initial_state["query_type"] == QueryType.CROSS_REFERENCE
    assert config["configurable"]["thread_id"] == "section-thread-1"
