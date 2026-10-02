"""sqlite-vec 向量库集成测试：真实建库、真实 upsert、真实 KNN。

用 mock embedder 产出可预期维度的向量，把检索结果做成确定性，
绕开"需要真实 OpenAI embedding key"的依赖。
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from src.models import Chunk
from src.retriever.vector_store import ChromaVectorStore


class _MockEmbedder:
    dim = 4

    def embed(self, texts):
        mapping = {
            "退款 3 天内到账": [1.0, 0.0, 0.0, 0.0],
            "退款 7 天内到账": [0.9, 0.1, 0.0, 0.0],
            "完全不相关的内容": [0.0, 0.0, 1.0, 0.0],
            "查询-退款": [1.0, 0.0, 0.0, 0.0],
        }
        return [mapping.get(t, [0.0, 1.0, 0.0, 0.0]) for t in texts]


def _mkchunk(cid: str, doc_id: str, idx: int, text: str, flagged=False) -> Chunk:
    return Chunk(
        chunk_id=cid, doc_id=doc_id, doc_name=f"{doc_id}.md",
        doc_path=f"/docs/{doc_id}.md", chunk_index=idx,
        heading_path="H", text=text, flagged_injection=flagged,
        injection_patterns=["p1"] if flagged else [],
    )


@pytest.fixture
def store_dir():
    d = Path(tempfile.mkdtemp(prefix="mini_rag_test_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_upsert_search_resolve_roundtrip(store_dir):
    vs = ChromaVectorStore(store_dir, "t", _MockEmbedder())
    vs.upsert([
        _mkchunk("c1", "docA", 0, "退款 3 天内到账"),
        _mkchunk("c2", "docA", 1, "退款 7 天内到账"),
        _mkchunk("c3", "docB", 0, "完全不相关的内容", flagged=True),
    ])
    assert vs.count() == 3

    hits = vs.search("查询-退款", top_k=2)
    assert [h.chunk_id for h in hits] == ["c1", "c2"]
    assert hits[0].vector_score > hits[1].vector_score
    assert hits[0].vector_score > 0.9  # 与查询向量几乎同向

    resolved = vs.resolve(["c1", "c3", "missing"])
    assert resolved["c1"].text == "退款 3 天内到账"
    assert resolved["c3"].flagged_injection is True
    assert "missing" not in resolved


def test_same_content_idempotent(store_dir):
    vs = ChromaVectorStore(store_dir, "t", _MockEmbedder())
    chunk = _mkchunk("c1", "docA", 0, "退款 3 天内到账")
    vs.upsert([chunk])
    vs.upsert([chunk])  # 重跑
    assert vs.count() == 1  # 幂等


def test_doc_change_replaces_old_chunks(store_dir):
    vs = ChromaVectorStore(store_dir, "t", _MockEmbedder())
    vs.upsert([_mkchunk("c1", "docA", 0, "退款 3 天内到账")])
    # 同一 doc 内容变更 → 新 doc_id 进来时旧 doc_id 被清
    # 这一层由 ingest 主流程保证，这里只验证 delete_by_doc_id 本身
    vs.delete_by_doc_id("docA")
    assert vs.count() == 0


def test_cosine_score_semantics(store_dir):
    # 验证：搜索"查询-退款"时，与其方向最接近的 c1 应得最高分；
    # 而无方向的 c3 分数应远低于阈值
    vs = ChromaVectorStore(store_dir, "t", _MockEmbedder())
    vs.upsert([
        _mkchunk("c1", "docA", 0, "退款 3 天内到账"),
        _mkchunk("c3", "docB", 0, "完全不相关的内容"),
    ])
    hits = vs.search("查询-退款", top_k=2)
    by_id = {h.chunk_id: h for h in hits}
    assert by_id["c1"].vector_score > 0.5
    assert by_id["c3"].vector_score < 0.5


class _ScaledEmbedder:
    """返回非单位向量：用来抓"把 L2 误当 cosine"的实现 bug。

    L2 距离下同向放大 2 倍的向量会离查询更远；cosine 距离下应距离为 0。
    """

    dim = 4

    def embed(self, texts):
        mapping = {
            "alpha": [2.0, 0.0, 0.0, 0.0],   # 与查询同向但模长 2
            "beta": [0.0, 1.0, 0.0, 0.0],    # 与查询正交
            "q": [1.0, 0.0, 0.0, 0.0],
        }
        return [mapping[t] for t in texts]


def test_nonunit_vectors_use_cosine_not_l2(store_dir):
    """若实现错把 L2 当 cosine，这里会立刻暴露：

    - cosine 语义：alpha(同向放大) sim=1.0，beta(正交) sim=0.0
    - L2 语义：alpha distance=1.0 → sim=0.0（与"几乎同向"的直觉相反）
    """
    vs = ChromaVectorStore(store_dir, "t", _ScaledEmbedder())
    vs.upsert([
        _mkchunk("a", "docA", 0, "alpha"),
        _mkchunk("b", "docA", 1, "beta"),
    ])
    hits = vs.search("q", top_k=2)
    by_id = {h.chunk_id: h for h in hits}
    assert by_id["a"].vector_score > 0.99, (
        f"同向放大向量的 cosine sim 应≈1.0，实际 {by_id['a'].vector_score}"
    )
    assert abs(by_id["b"].vector_score) < 1e-3, (
        f"正交向量的 cosine sim 应≈0.0，实际 {by_id['b'].vector_score}"
    )


class _UniformEmbedder:
    """所有文本同向量，让 count/检索结果可预测，专注验证重灌语义。"""

    dim = 4

    def embed(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


def test_replace_doc_shrinking_chunk_count_leaves_no_residue(store_dir):
    """P0-1 核心场景：同一 doc_id 第二次灌的 chunk 数比第一次少。

    旧实现若只做 upsert 不做全量清理，旧的多余 chunk_id 会残留在 chunks 与
    vec_chunks 两张表里，仍能被检索到——这是审查里指出的真实 bug。
    """
    vs = ChromaVectorStore(store_dir, "t", _UniformEmbedder())

    # 第一次灌：4 个 chunk
    first = [_mkchunk(f"c{i}", "docA", i, f"第 {i} 段") for i in range(4)]
    vs.replace_doc("docA", first)
    assert vs.count() == 4

    # 第二次灌：只剩 2 个 chunk（文档被改短）
    second = [
        _mkchunk("c0", "docA", 0, "第 0 段 v2"),
        _mkchunk("c1", "docA", 1, "第 1 段 v2"),
    ]
    vs.replace_doc("docA", second)

    # 残留的 c2、c3 必须消失：元数据 + 向量表都不应有
    assert vs.count() == 2
    resolved = vs.resolve(["c2", "c3"])
    assert resolved == {}
    hits = vs.search("任意查询", top_k=10)
    assert {h.chunk_id for h in hits} == {"c0", "c1"}

    # 元数据应该是新版本
    assert vs.resolve(["c0"])["c0"].text == "第 0 段 v2"


def test_replace_doc_rejects_mismatched_doc_id(store_dir):
    """replace_doc 必须校验 doc_id 一致性，防御调用方传错"""
    vs = ChromaVectorStore(store_dir, "t", _UniformEmbedder())
    bad = _mkchunk("c0", "docOTHER", 0, "文本")
    try:
        vs.replace_doc("docA", [bad])
    except ValueError as e:
        assert "doc_id" in str(e)
    else:
        raise AssertionError("应对 doc_id 不一致抛出 ValueError")


def test_replace_doc_rejects_empty_chunks(store_dir):
    vs = ChromaVectorStore(store_dir, "t", _UniformEmbedder())
    try:
        vs.replace_doc("docA", [])
    except ValueError:
        pass
    else:
        raise AssertionError("空 chunks 应抛 ValueError")
