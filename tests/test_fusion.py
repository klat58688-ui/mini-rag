"""RRF 融合测试：只看排名、不看原始分数；多路交叉命中的 chunk 应排在前面。"""

from src.models import ScoredChunk
from src.retriever.fusion import rrf_fuse


def _sc(chunk_id: str, v=None, b=None, flagged=False) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id, vector_score=v, bm25_score=b, flagged_injection=flagged
    )


def test_rrf_prefers_multi_ranked_chunk():
    v = [_sc("A", v=0.9), _sc("B", v=0.8), _sc("C", v=0.7)]
    b = [_sc("B", b=10.0), _sc("D", b=9.0)]
    fused = rrf_fuse([v, b], k=60)
    assert fused[0].chunk_id == "B"


def test_rrf_score_only_depends_on_rank():
    v = [_sc("A", v=0.99), _sc("B", v=0.01)]
    fused = rrf_fuse([v], k=60)
    assert abs(fused[0].rrf_score - 1 / 61) < 1e-6


def test_rrf_preserves_scores_and_flag():
    v = [_sc("A", v=0.5, flagged=True)]
    fused = rrf_fuse([v], k=60)
    assert fused[0].vector_score == 0.5
    assert fused[0].flagged_injection is True


def test_rrf_k_must_be_positive():
    import pytest

    with pytest.raises(ValueError):
        rrf_fuse([[]], k=0)
