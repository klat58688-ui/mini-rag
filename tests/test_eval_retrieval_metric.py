"""检索指标的单测：真 recall@k（gold 覆盖率） vs hit@k（任一命中）——README §15.26。

旧实现只算"top-k 里有没有任一 gold"，那是 hit@k，却被命名为 recall@10。
对单 gold 题目两者等价，所以长期没暴露；多 gold（多跳）题目下必须区分。
这里用 stub 覆盖全部边界。
"""

from eval.run_eval import eval_retrieval
from src.models import ScoredChunk


class _Cfg:
    vector_top_k = 10
    bm25_top_k = 10
    rrf_k = 60
    final_top_k = 5


class _VectorStore:
    """按给定顺序返回命中，模拟融合后的排序。"""

    def __init__(self, ids):
        self.ids = list(ids)

    def search(self, query, top_k):
        return [ScoredChunk(chunk_id=i, text="t", vector_score=1.0) for i in self.ids]


class _BM25:
    def search(self, query, top_k):
        return []


class _Pipe:
    def __init__(self, ids):
        self.cfg = _Cfg()
        self.vector_store = _VectorStore(ids)
        self.bm25_store = _BM25()

    def _per_list_penalize(self, hits):
        return hits


def _q(question, gold, expect_refuse=False):
    return {"question": question, "gold_chunk_ids": list(gold),
            "expect_refuse": expect_refuse}


# ── 单 gold：recall 与 hit 必须一致（保证历史可比） ──────────────

def test_single_gold_hit_at_rank1():
    r = eval_retrieval(_Pipe(["a", "b", "c"]), [_q("q", ["a"])])
    assert r["recall@10"] == 1.0
    assert r["hit@10"] == 1.0
    assert r["mrr@10"] == 1.0
    assert r["details"][0]["coverage"] == 1.0


def test_single_gold_hit_at_rank3():
    r = eval_retrieval(_Pipe(["x", "y", "a"]), [_q("q", ["a"])])
    assert r["recall@10"] == 1.0
    assert r["hit@10"] == 1.0
    assert abs(r["mrr@10"] - 1 / 3) < 1e-9


def test_single_gold_missing():
    r = eval_retrieval(_Pipe(["x", "y"]), [_q("q", ["a"])])
    assert r["recall@10"] == 0.0
    assert r["hit@10"] == 0.0
    assert r["mrr@10"] == 0.0
    assert r["details"][0]["first_hit_rank"] is None


# ── 多 gold：这才是新口径的关键 ──────────────────────────────

def test_multi_gold_all_covered():
    r = eval_retrieval(_Pipe(["a", "b", "c"]), [_q("q", ["a", "b"])])
    assert r["recall@10"] == 1.0
    assert r["hit@10"] == 1.0


def test_multi_gold_partial_coverage_lowers_recall_but_not_hit():
    """只召回一半：hit@10 仍算命中，真 recall 必须掉到 0.5。"""
    r = eval_retrieval(_Pipe(["a", "x", "y"]), [_q("q", ["a", "b"])])
    assert r["recall@10"] == 0.5
    assert r["hit@10"] == 1.0
    d = r["details"][0]
    assert (d["n_gold"], d["n_covered"], d["coverage"]) == (2, 1, 0.5)


def test_multi_gold_none_covered():
    r = eval_retrieval(_Pipe(["x", "y"]), [_q("q", ["a", "b"])])
    assert r["recall@10"] == 0.0
    assert r["hit@10"] == 0.0


def test_multi_gold_three_of_three_partial():
    r = eval_retrieval(_Pipe(["a", "b"]), [_q("q", ["a", "b", "c"])])
    assert abs(r["recall@10"] - 2 / 3) < 1e-9
    assert r["hit@10"] == 1.0


# ── 聚合与过滤 ────────────────────────────────────────────

def test_recall_is_mean_over_questions():
    # 三题覆盖度 1.0 / 0.5 / 0.0 → 均值 0.5
    r = eval_retrieval(
        _Pipe(["a", "x"]),
        [_q("q1", ["a"]), _q("q2", ["a", "b"]), _q("q3", ["z"])],
    )
    assert r["n"] == 3
    assert abs(r["recall@10"] - 0.5) < 1e-9
    assert abs(r["hit@10"] - 2 / 3) < 1e-9


def test_refuse_items_are_skipped():
    r = eval_retrieval(
        _Pipe(["a"]),
        [_q("q1", ["a"]), _q("refuse", [], expect_refuse=True)],
    )
    assert r["n"] == 1
    assert r["recall@10"] == 1.0


def test_custom_k_is_respected():
    # gold 在第 3 位，k=2 时应算未覆盖
    r = eval_retrieval(_Pipe(["x", "y", "a"]), [_q("q", ["a"])], k=2)
    assert r["recall@10"] == 0.0
    assert r["hit@10"] == 0.0
