from __future__ import annotations

import json
import uuid
from datetime import date
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from civicsetu.models.enums import Jurisdiction, QueryType
from civicsetu.models.schemas import Citation
from tests.conftest import _base_state


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

        from civicsetu.api.main import create_app
        app = create_app()
        app.state.graph = mock_compiled

        with TestClient(app) as c:
            yield c, mock_compiled


# ── POST /api/v1/query ────────────────────────────────────────────────────────

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
