"""查询主流程编排 —— 唯一知道全部环节的类。

固定顺序（对应修订后的数据流）：
  双路召回 -> 列内注入降权 -> RRF 融合 -> 拒答门控（cosine 原始分）
  -> 生成 -> 引用校验 -> 按 chunk_id 回源组装响应
"""

from __future__ import annotations

from .config import AppConfig
from .guardrails import should_refuse, validate_and_map_citations
from .models import Answer, Citation, ScoredChunk
from .multihop import merge_hop_results, plan_hop2
from .retriever.bm25_store import BM25Store
from .retriever.fusion import rrf_fuse
from .retriever.vector_store import ChromaVectorStore


class RagPipeline:
    def __init__(
        self,
        cfg: AppConfig,
        vector_store: ChromaVectorStore,
        bm25_store: BM25Store,
        generator,
    ):
        self.cfg = cfg
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.generator = generator

    # ── 内部步骤 ─────────────────────────────────────────
    def _dual_recall(self, query: str):
        vector_hits = self.vector_store.search(query, self.cfg.vector_top_k)
        bm25_hits = self.bm25_store.search(query, self.cfg.bm25_top_k)
        return vector_hits, bm25_hits

    def _per_list_penalize(self, hits: list[ScoredChunk]) -> list[ScoredChunk]:
        """修订 2：融合前列内降权。flagged 项整体沉到未 flagged 之后，
        每路最多保留 injection_max_penalized 个名额（超出丢弃，保留极端可答性）。
        消融开关 injection_penalty_enabled=False 时跳过本步，把未降权的列表交给 RRF。"""
        if not self.cfg.injection_penalty_enabled:
            return hits
        clean = [h for h in hits if not h.flagged_injection]
        flagged = [h for h in hits if h.flagged_injection]
        kept_flagged = flagged[: self.cfg.injection_max_penalized]
        return clean + kept_flagged

    def _fuse(self, vector_hits, bm25_hits) -> list[ScoredChunk]:
        return rrf_fuse([vector_hits, bm25_hits], k=self.cfg.rrf_k)

    def _gate(self, query: str, vector_hits, bm25_hits) -> tuple[bool, str]:
        bm25_strong = False
        if bm25_hits:
            # P0-2 旁路 A：判据升级为"BM25 兜底所依据的 top-N 证据集全干净"，
            # 而不是只看 top-1。top-1 干净 + rank 2 注入也必须拒兜底。
            top_chunk_ids = [h.chunk_id for h in bm25_hits[:3]]
            bm25_strong = self.bm25_store.strong_exact_match(
                query, top_chunk_ids=top_chunk_ids
            )
        return should_refuse(
            vector_hits,
            bm25_hits,
            query,
            cosine_threshold=self.cfg.cosine_threshold,
            bm25_strong_match=bm25_strong,
        )

    def _build_refusal_answer(
        self, query: str, reason: str, fused: list[ScoredChunk]
    ) -> Answer:
        """拒答也要给"最接近的 3 条候选"——比干巴巴一句"不知道"有用。"""
        nearest = fused[:3]
        resolved = self.vector_store.resolve([s.chunk_id for s in nearest])
        citations: list[Citation] = []
        for i, sc in enumerate(nearest, start=1):
            chunk = resolved.get(sc.chunk_id)
            if chunk is None:
                continue
            citations.append(
                Citation(
                    ref_id=i,
                    chunk_id=chunk.chunk_id,
                    doc_name=chunk.doc_name,
                    heading_path=chunk.heading_path,
                    snippet=chunk.text[:240],
                    score=round(float(sc.vector_score or sc.bm25_score or 0.0), 4),
                    flagged_injection=chunk.flagged_injection,
                )
            )
        return Answer(
            text="知识库中没有找到足够相关的内容。",
            citations=citations,
            refused=True,
            refusal_reason=reason,
        )

    # ── 第二跳（§15.31，链式多跳的查询扩展层）─────────────
    def _maybe_hop2_fused(
        self, query: str, hop1_fused: list[ScoredChunk]
    ) -> list[ScoredChunk]:
        """若 plan_hop2 给出扩展 query，则补一次双路召回 + penalize + RRF，
        再和第一跳 fused 过一次 RRF。否则原样返回 hop1_fused。

        为什么放在门控之后、送 LLM 之前：
        - 门控拒答时不需要补跳（减少一次 LLM 上下文噪音）。
        - penalize 必须分别作用于两跳各自的列内，不能先合并再 penalize——
          hop2 的 flagged 项也要按列内限额走（防 hop1 clean + hop2 注入挤进）。
        """
        plan = plan_hop2(query)
        if not (plan.triggered and plan.expanded_query):
            return hop1_fused
        v2, b2 = self._dual_recall(plan.expanded_query)
        v2 = self._per_list_penalize(v2)
        b2 = self._per_list_penalize(b2)
        hop2 = self._fuse(v2, b2)
        return merge_hop_results(hop1_fused, hop2, rrf_fuse, self.cfg.rrf_k)

    # ── 对外入口 ─────────────────────────────────────────
    def ask(self, query: str) -> Answer:
        if not query or not query.strip():
            raise ValueError("查询不能为空")

        # P0-2：拒答门控必须看**未被 penalize 干预**的原始 BM25 顶部，
        # 否则 penalize 把 flagged 排到干净后面之后，"top-1 是否 flagged"就失去了信号。
        vector_hits_raw, bm25_hits_raw = self._dual_recall(query)
        refuse, reason = self._gate(query, vector_hits_raw, bm25_hits_raw)

        vector_hits = self._per_list_penalize(vector_hits_raw)
        bm25_hits = self._per_list_penalize(bm25_hits_raw)
        fused = self._fuse(vector_hits, bm25_hits)
        if refuse:
            return self._build_refusal_answer(query, reason, fused)

        # §15.31：链式多跳的查询扩展层（门控通过后才做）。
        fused = self._maybe_hop2_fused(query, fused)

        contexts = fused[: self.cfg.final_top_k]
        llm_text = self.generator.generate(query, contexts)
        if llm_text.strip() == "INSUFFICIENT_CONTEXT":
            return self._build_refusal_answer(
                query, "LLM 判定上下文不足（INSUFFICIENT_CONTEXT）", fused
            )

        resolved = self.vector_store.resolve([c.chunk_id for c in contexts])
        answer = validate_and_map_citations(llm_text, contexts, resolved)
        # P0-2 旁路 B：放行之后再看一眼。"放行即所见"——若 contexts 里有
        # flagged 而 LLM 又给出了"看似正常"的回答，调用方必须能区分这是
        # "基于带毒证据的回答"。不拒答（已经走到这一步），但要标 degraded。
        answer.degraded = any(c.flagged_injection for c in contexts)
        return answer
