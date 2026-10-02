"""BM25 关键词索引：rank_bm25 + jieba。

是 Chroma 的派生投影（修订 3）：build 的输入就是 all_chunks()，重启重建，千级语料 < 1s。
退一步说，若哪天语料大到不能全量重建，要改的也是这里，真相源不动。
"""

from __future__ import annotations

from rank_bm25 import BM25Okapi

from ..models import Chunk, ScoredChunk
from .tokenizer import tokenize


class BM25Store:
    def __init__(self):
        self._chunks: list[Chunk] = []
        self._model: BM25Okapi | None = None

    def build(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        if not chunks:
            self._model = None
            return
        corpus = [tokenize(c.text) for c in chunks]
        self._model = BM25Okapi(corpus)

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        if top_k <= 0:
            raise ValueError(f"top_k 必须为正: {top_k}")
        if self._model is None:
            return []
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores = self._model.get_scores(q_tokens)
        ranked = sorted(
            zip(self._chunks, scores, strict=True), key=lambda x: x[1], reverse=True
        )[:top_k]
        out: list[ScoredChunk] = []
        for chunk, score in ranked:
            out.append(
                ScoredChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    bm25_score=float(score),
                    flagged_injection=chunk.flagged_injection,
                )
            )
        return out

    def strong_exact_match(
        self, query: str,
        top_chunk_id: str | None = None,
        top_chunk_ids: list[str] | None = None,
    ) -> bool:
        """修订 1 的兜底放行：query 稀有 token 是否在 BM25 top-N 中存在"干净命中"。

        判据：**top-N 里至少存在一个 chunk**——
          1) 未被 detect_injection 标记（干净的）；
          2) 其 token 集合覆盖 query 的全部稀有 token（rare_tokens）。
        满足 → True；否则 → False。

        演变：
        - 旁路 A 修复（评审轮 2 / P0-2）：旧版只看 top-1，攻击者把 flagged 推到 rank 1
          头顶再让 rank-2 干净 chunk 撞罕词即可悾 strong=True。一度改为"top-N 全部干净
          且全部命中"。
        - T0 修复（复核轮 3）：全部命中在小语料上是过度防御——BM25 的天性就是 top-1
          命中后尾随几个 0 分占位 chunk，导致术语解释类（glossary）query 永远接不到
          strong 信号，实际上把"BM25 强匹配兑底"这条防线变成了哑火。改为现在这种
          "存在性"语义后：
            * 下限仍在：攻击者把 flagged 推到 Rank 1 不影响我们继续看 Rank 2/3
              里是否有干净命中（flagged 走 continue，不会返回 True）。
            * 生效面恢复：术语表 / 短 chunk / 正交多主题语料原本被误伤的区间重新起效，
              实证见 README §15.10/§15.11。

        兼容旧签名：传单个 top_chunk_id 时等价于 top_chunk_ids=[top_chunk_id]。
        """
        if not query.strip():
            return False
        if top_chunk_ids is None:
            top_chunk_ids = [top_chunk_id] if top_chunk_id else []
        if not top_chunk_ids:
            return False

        from .tokenizer import rare_tokens
        q_rare = rare_tokens(query)
        if not q_rare:
            return False

        by_id = {c.chunk_id: c for c in self._chunks}
        for cid in top_chunk_ids:
            chunk = by_id.get(cid)
            if chunk is None:
                continue
            if chunk.flagged_injection:
                continue  # flagged 不兑底但不一票否决；看其它干净 top-N 有没有命中
            c_tokens = set(tokenize(chunk.text))
            if q_rare.issubset(c_tokens):
                return True
        return False
