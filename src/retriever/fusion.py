"""RRF 倒数排名融合。score(d) = Σ 1/(k + rank_i(d))，rank 从 1 起。

为什么用 RRF 而非加权分数：向量分 (0~1) 与 BM25 分（无上界）量纲不同，
归一化对分布敏感、调权玄学。RRF 只看排名，免归一化，实现 10 行，这是卖点。

输入必须是"已完成注入降权"的有序列表（修订 2）——本模块不感知 flagged，
保证融合逻辑不含杂质、可解释。
"""

from __future__ import annotations

from ..models import ScoredChunk


def rrf_fuse(rank_lists: list[list[ScoredChunk]], k: int = 60) -> list[ScoredChunk]:
    if k <= 0:
        raise ValueError(f"RRF k 必须为正: {k}")
    merged: dict[str, ScoredChunk] = {}
    for rank_list in rank_lists:
        for rank, item in enumerate(rank_list, start=1):
            entry = merged.get(item.chunk_id)
            if entry is None:
                # 复制一份，避免污染调用方持有的对象，且只透传原始分
                entry = ScoredChunk(
                    chunk_id=item.chunk_id,
                    text=item.text,
                    vector_score=item.vector_score,
                    bm25_score=item.bm25_score,
                    flagged_injection=item.flagged_injection,
                )
                merged[item.chunk_id] = entry
            entry.rrf_score = (entry.rrf_score or 0.0) + 1.0 / (k + rank)
    return sorted(merged.values(), key=lambda x: x.rrf_score or 0.0, reverse=True)
