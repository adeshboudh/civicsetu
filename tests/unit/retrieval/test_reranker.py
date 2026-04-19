from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from tests.conftest import _make_rc


# ── _get_ranker ───────────────────────────────────────────────────────────────

def test_get_ranker_uses_settings_model():
    import civicsetu.retrieval.reranker as reranker_mod
    reranker_mod._ranker = None
    with patch("civicsetu.retrieval.reranker.settings") as ms:
        ms.reranker_model = "rank-T5-flan"
        with patch("flashrank.Ranker") as MockRanker:
            MockRanker.return_value = MagicMock()
            reranker_mod._get_ranker()
            MockRanker.assert_called_once_with(model_name="rank-T5-flan", cache_dir=".cache/flashrank")
    reranker_mod._ranker = None


def test_get_ranker_caches_real_ranker():
    import civicsetu.retrieval.reranker as reranker_mod
    reranker_mod._ranker = None
    real_mock = MagicMock()
    real_mock.__class__.__module__ = "flashrank"
    with patch("civicsetu.retrieval.reranker.settings"):
        with patch("flashrank.Ranker", return_value=real_mock):
            r1 = reranker_mod._get_ranker()
            r2 = reranker_mod._get_ranker()
    assert r1 is r2
    reranker_mod._ranker = None


def test_get_ranker_does_not_cache_mock():
    import civicsetu.retrieval.reranker as reranker_mod
    reranker_mod._ranker = None
    with patch("civicsetu.retrieval.reranker.settings"):
        with patch("flashrank.Ranker") as MockRanker:
            MockRanker.return_value = MagicMock()
            reranker_mod._get_ranker()
            assert reranker_mod._ranker is None
    reranker_mod._ranker = None


# ── _apply_score_gap ──────────────────────────────────────────────────────────

def test_apply_score_gap_empty():
    from civicsetu.retrieval.reranker import _apply_score_gap
    assert _apply_score_gap([], gap=0.35) == []


def test_apply_score_gap_single():
    from civicsetu.retrieval.reranker import _apply_score_gap
    c = _make_rc(rerank_score=0.8)
    assert _apply_score_gap([c], gap=0.35) == [c]


def test_apply_score_gap_no_cliff():
    from civicsetu.retrieval.reranker import _apply_score_gap
    chunks = [_make_rc(rerank_score=0.85), _make_rc(rerank_score=0.75), _make_rc(rerank_score=0.65)]
    assert len(_apply_score_gap(chunks, gap=0.35)) == 3


def test_apply_score_gap_stops_at_cliff():
    from civicsetu.retrieval.reranker import _apply_score_gap
    chunks = [
        _make_rc(rerank_score=0.88),
        _make_rc(rerank_score=0.82),
        _make_rc(rerank_score=0.40),
        _make_rc(rerank_score=0.35),
    ]
    result = _apply_score_gap(chunks, gap=0.35)
    assert len(result) == 2
    assert result[0].rerank_score == 0.88
    assert result[1].rerank_score == 0.82


def test_apply_score_gap_first_pair_is_cliff():
    from civicsetu.retrieval.reranker import _apply_score_gap
    chunks = [_make_rc(rerank_score=0.90), _make_rc(rerank_score=0.40)]
    assert len(_apply_score_gap(chunks, gap=0.35)) == 1


# ── Reranker.rerank ───────────────────────────────────────────────────────────

def test_rerank_empty_chunks():
    from civicsetu.retrieval.reranker import Reranker
    assert Reranker.rerank([], query="test") == []


def test_rerank_pinned_chunks_always_first():
    from civicsetu.retrieval.reranker import Reranker
    pinned = _make_rc(section_id="18", is_pinned=True)
    u1 = _make_rc(section_id="3")
    u2 = _make_rc(section_id="7")
    with patch("flashrank.Ranker") as MockRanker:
        MockRanker.return_value.rerank.return_value = [{"id": 0, "score": 0.9}, {"id": 1, "score": 0.8}]
        result = Reranker.rerank([pinned, u1, u2], query="test query")
    assert result[0].is_pinned is True


def test_rerank_max_two_pinned():
    from civicsetu.retrieval.reranker import Reranker
    pinned = [_make_rc(section_id=str(i), is_pinned=True) for i in range(4)]
    with patch("flashrank.Ranker") as MockRanker:
        MockRanker.return_value.rerank.return_value = []
        result = Reranker.rerank(pinned, query="test")
    assert len([c for c in result if c.is_pinned]) <= 2


def test_rerank_deduplicates_by_section_and_doc():
    from civicsetu.retrieval.reranker import Reranker
    chunk_a = _make_rc(section_id="18", doc_name="RERA Act 2016")
    chunk_b = _make_rc(section_id="18", doc_name="RERA Act 2016")
    with patch("flashrank.Ranker") as MockRanker:
        MockRanker.return_value.rerank.return_value = [{"id": 0, "score": 0.9}]
        result = Reranker.rerank([chunk_a, chunk_b], query="test")
    assert len(result) == 1


def test_rerank_drops_below_threshold():
    from civicsetu.retrieval.reranker import Reranker
    c_high = _make_rc(section_id="1")
    c_low = _make_rc(section_id="2")
    with patch("flashrank.Ranker") as MockRanker:
        MockRanker.return_value.rerank.return_value = [{"id": 0, "score": 0.80}, {"id": 1, "score": 0.10}]
        with patch("civicsetu.retrieval.reranker.settings") as ms:
            ms.reranker_model = "rank-T5-flan"
            ms.reranker_score_threshold = 0.3
            ms.reranker_score_gap = 0.35
            result = Reranker.rerank([c_high, c_low], query="test")
    ids = [c.chunk.section_id for c in result]
    assert "1" in ids
    assert "2" not in ids


def test_rerank_applies_score_gap():
    from civicsetu.retrieval.reranker import Reranker
    c1, c2, c3 = _make_rc(section_id="1"), _make_rc(section_id="2"), _make_rc(section_id="3")
    with patch("flashrank.Ranker") as MockRanker:
        MockRanker.return_value.rerank.return_value = [
            {"id": 0, "score": 0.88}, {"id": 1, "score": 0.82}, {"id": 2, "score": 0.40}
        ]
        with patch("civicsetu.retrieval.reranker.settings") as ms:
            ms.reranker_model = "rank-T5-flan"
            ms.reranker_score_threshold = 0.3
            ms.reranker_score_gap = 0.35
            result = Reranker.rerank([c1, c2, c3], query="test")
    ids = [c.chunk.section_id for c in result]
    assert "1" in ids and "2" in ids and "3" not in ids


def test_rerank_fallback_on_ranker_exception():
    from civicsetu.retrieval.reranker import Reranker
    chunks = [_make_rc(section_id=str(i)) for i in range(3)]
    with patch("civicsetu.retrieval.reranker._get_ranker", side_effect=RuntimeError("GPU OOM")):
        result = Reranker.rerank(chunks, query="test")
    assert len(result) > 0


def test_rerank_max_five_results():
    from civicsetu.retrieval.reranker import Reranker
    chunks = [_make_rc(section_id=str(i)) for i in range(10)]
    mock_results = [{"id": i, "score": 0.9 - i * 0.01} for i in range(10)]
    with patch("flashrank.Ranker") as MockRanker:
        MockRanker.return_value.rerank.return_value = mock_results
        with patch("civicsetu.retrieval.reranker.settings") as ms:
            ms.reranker_model = "rank-T5-flan"
            ms.reranker_score_threshold = 0.0
            ms.reranker_score_gap = 999.0
            result = Reranker.rerank(chunks, query="test")
    assert len(result) <= 5
