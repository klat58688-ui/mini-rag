"""src/api.py 的接口测试（README §15.28）。

`api.py` 此前**零覆盖**。这里用 stub pipeline 顶替真实构建——lifespan 里的
`build_pipeline(".env")` 被 monkeypatch 掉，所以不打 LLM、不加载嵌入模型、不读 .env。
"""

import pytest
from fastapi.testclient import TestClient

import src.api as api
from src.models import Answer, Citation


def _answer(refused: bool = False) -> Answer:
    return Answer(
        text="退款会在 3 个工作日内原路退回 [1]。",
        citations=[
            Citation(
                ref_id=1, chunk_id="c605ca6063206f4f", doc_name="faq.md",
                heading_path="产品使用常见问题 > 退款政策 > 退款多久到账",
                snippet="退款会在 3 个工作日内原路退回。", score=0.9,
                flagged_injection=False,
            )
        ],
        refused=refused,
        refusal_reason="检索相关性过低" if refused else None,
        invalid_refs_removed=False,
        degraded=False,
    )


class _StubPipeline:
    def __init__(self, answer=None, exc=None):
        self._answer = answer
        self._exc = exc
        self.seen: list[str] = []

    def ask(self, question: str) -> Answer:
        self.seen.append(question)
        if self._exc is not None:
            raise self._exc
        return self._answer


def _client(monkeypatch, pipeline) -> TestClient:
    """把 api 里的 build_pipeline 换成返回 stub，避免真实初始化。"""
    monkeypatch.setattr(api, "build_pipeline", lambda env: pipeline)
    return TestClient(api.app)


# ── 启动契约 ──────────────────────────────────────────────

def test_lifespan_builds_pipeline_from_dot_env(monkeypatch):
    calls: list[str] = []
    pipe = _StubPipeline(_answer())

    def _fake_build(env):
        calls.append(env)
        return pipe

    monkeypatch.setattr(api, "build_pipeline", _fake_build)
    with TestClient(api.app) as client:
        client.get("/healthz")
    assert calls == [".env"]


# ── /healthz ─────────────────────────────────────────────

def test_healthz(monkeypatch):
    with _client(monkeypatch, _StubPipeline(_answer())) as client:
        resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ── /ask 正常路径 ─────────────────────────────────────────

def test_ask_returns_serialised_answer(monkeypatch):
    with _client(monkeypatch, _StubPipeline(_answer())) as client:
        resp = client.post("/ask", json={"question": "退款多久到账？"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["text"].startswith("退款会在 3 个工作日")
    assert body["refused"] is False
    assert body["refusal_reason"] is None
    assert body["invalid_refs_removed"] is False
    assert body["degraded"] is False
    assert len(body["citations"]) == 1
    assert body["citations"][0]["chunk_id"] == "c605ca6063206f4f"
    assert body["citations"][0]["ref_id"] == 1
    assert body["citations"][0]["flagged_injection"] is False


def test_ask_preserves_refusal_fields(monkeypatch):
    with _client(monkeypatch, _StubPipeline(_answer(refused=True))) as client:
        resp = client.post("/ask", json={"question": "公司食堂几点开门？"})
    body = resp.json()
    assert body["refused"] is True
    assert body["refusal_reason"] == "检索相关性过低"


def test_question_is_stripped_before_reaching_pipeline(monkeypatch):
    pipe = _StubPipeline(_answer())
    with _client(monkeypatch, pipe) as client:
        client.post("/ask", json={"question": "  退款多久到账？  "})
    assert pipe.seen == ["退款多久到账？"]


# ── /ask 参数校验 ─────────────────────────────────────────

@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},           # 空串
        {"question": "   "},        # 只有空白
        {},                         # 缺字段
        {"q": "退款多久到账？"},      # 字段名写错
    ],
)
def test_ask_rejects_missing_or_blank_question(monkeypatch, payload):
    pipe = _StubPipeline(_answer())
    with _client(monkeypatch, pipe) as client:
        resp = client.post("/ask", json=payload)
    assert resp.status_code == 400
    assert "question" in resp.json()["detail"]
    assert pipe.seen == []  # 校验失败时不应触达 pipeline


# ── /ask 错误映射 ─────────────────────────────────────────

def test_ask_maps_pipeline_exception_to_500(monkeypatch):
    pipe = _StubPipeline(exc=RuntimeError("upstream 挂了"))
    with _client(monkeypatch, pipe) as client:
        resp = client.post("/ask", json={"question": "退款多久到账？"})
    assert resp.status_code == 500
    assert "upstream 挂了" in resp.json()["detail"]
