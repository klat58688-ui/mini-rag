"""答案质量廉价层的单测：引用精度 + 要点覆盖 + 拒答交叉表——README §15.32。

recall@10=1.000 只说明证据进了上下文，不说明答案用对了证据。
这里用预制 Answer 的 stub 覆盖指标计算的全部边界。
"""

from eval.run_eval import answer_point_coverage, citation_precision, eval_answer
from src.models import Answer, Citation


class _Pipe:
    """按 question → Answer 的映射返回预制答案。

    接口缺失是 mock 的责任（§15.31）：eval_answer 只调用 pipeline.ask()，
    所以 stub 只需实现 ask——检索链路与本模式无关。
    """

    def __init__(self, answers):
        self.answers = dict(answers)

    def ask(self, question):
        return self.answers[question]


def _ans(text="正文", cited=(), refused=False):
    return Answer(
        text=text,
        citations=[Citation(ref_id=i + 1, chunk_id=c) for i, c in enumerate(cited)],
        refused=refused,
    )


def _q(question, gold=(), points=(), expect_refuse=False):
    return {"question": question, "gold_chunk_ids": list(gold),
            "gold_answer_points": list(points), "expect_refuse": expect_refuse}


# ── 纯函数：citation_precision ─────────────────────────────

def test_citation_precision_all_hit():
    assert citation_precision(["a", "b"], {"a", "b"}) == 1.0


def test_citation_precision_partial():
    assert citation_precision(["a", "x", "y"], {"a"}) == 1 / 3


def test_citation_precision_none_hit():
    assert citation_precision(["x"], {"a"}) == 0.0


def test_citation_precision_empty_cited_is_undefined():
    assert citation_precision([], {"a"}) is None


def test_citation_precision_counts_each_citation_independently():
    # 同一 gold 块被引两次：两条引用都算对（分母不去重）
    assert citation_precision(["a", "a"], {"a"}) == 1.0


# ── 纯函数：answer_point_coverage ───────────────────────────

def test_answer_point_coverage_full():
    assert answer_point_coverage("请假需提前 3 个工作日申请", ["3 个工作日"]) == 1.0


def test_answer_point_coverage_partial():
    cov = answer_point_coverage("提前 3 天", ["3 个工作日", "直属上级"])
    assert cov == 0.0  # 字面未命中同义改写，是下界估计的体现
    cov2 = answer_point_coverage("提前 3 个工作日，报直属上级", ["3 个工作日", "直属上级"])
    assert cov2 == 1.0


def test_answer_point_coverage_no_points_is_undefined():
    assert answer_point_coverage("任意文本", []) is None


# ── eval_answer：打分与交叉表 ──────────────────────────────

def test_answerable_fully_correct():
    pipe = _Pipe({"q1": _ans("3 个工作日", cited=["a"])})
    r = eval_answer(pipe, [_q("q1", gold=["a"], points=["3 个工作日"])])
    assert r["refusal"] == {"tp": 0, "tn": 1, "fp": 0, "fn": 0}
    cp = r["citation_precision"]
    assert cp["n_scored"] == 1 and cp["macro"] == 1.0 and cp["micro"] == 1.0
    assert r["answer_point_coverage"]["macro"] == 1.0
    d = r["details"][0]
    assert d["cited_ids"] == ["a"] and d["citation_precision"] == 1.0


def test_answerable_partial_citation_lowers_both_means():
    pipe = _Pipe({"q1": _ans(cited=["a", "x"])})
    r = eval_answer(pipe, [_q("q1", gold=["a"])])
    assert r["citation_precision"]["macro"] == 0.5
    assert r["citation_precision"]["micro"] == 0.5
    assert r["details"][0]["citation_precision"] == 0.5


def test_answerable_no_citation_counted_separately():
    pipe = _Pipe({"q1": _ans(cited=[])})
    r = eval_answer(pipe, [_q("q1", gold=["a"])])
    cp = r["citation_precision"]
    assert cp["n_scored"] == 0 and cp["n_no_citation"] == 1
    assert cp["macro"] == 0.0 and cp["micro"] == 0.0
    assert r["details"][0]["citation_precision"] is None


def test_answerable_refused_counts_fp_and_not_scored():
    pipe = _Pipe({"q1": _ans(refused=True)})
    r = eval_answer(pipe, [_q("q1", gold=["a"])])
    assert r["refusal"]["fp"] == 1
    cp = r["citation_precision"]
    assert cp["n_refused_answerable"] == 1 and cp["n_scored"] == 0


def test_refuse_expected_refused_is_tp_not_scored():
    pipe = _Pipe({"q1": _ans(refused=True)})
    r = eval_answer(pipe, [_q("q1", expect_refuse=True)])
    assert r["refusal"]["tp"] == 1
    assert r["citation_precision"]["n_scored"] == 0


def test_refuse_expected_not_refused_is_fn_not_scored():
    # 拒答题即使答了且带引用，也不算"用对证据"——只进交叉表
    pipe = _Pipe({"q1": _ans(cited=["a"])})
    r = eval_answer(pipe, [_q("q1", gold=[], expect_refuse=True)])
    assert r["refusal"]["fn"] == 1
    assert r["citation_precision"]["n_scored"] == 0
    assert r["details"][0]["citation_precision"] is None


def test_macro_and_micro_diverge_when_citation_counts_differ():
    # q1: 1/2 命中（0.5）；q2: 1/1 命中（1.0）→ macro 0.75，micro 2/3
    pipe = _Pipe({"q1": _ans(cited=["a", "x"]), "q2": _ans(cited=["b"])})
    r = eval_answer(pipe, [_q("q1", gold=["a"]), _q("q2", gold=["b"])])
    assert abs(r["citation_precision"]["macro"] - 0.75) < 1e-9
    assert abs(r["citation_precision"]["micro"] - 2 / 3) < 1e-9


def test_coverage_macro_and_micro_aggregation():
    # q1: 1/2 要点命中；q2: 1/1 命中 → macro 0.75，micro 2/3
    pipe = _Pipe({
        "q1": _ans("工作日", cited=["a"]),
        "q2": _ans("顺延", cited=["b"]),
    })
    r = eval_answer(pipe, [
        _q("q1", gold=["a"], points=["工作日", "审批"]),
        _q("q2", gold=["b"], points=["顺延"]),
    ])
    cov = r["answer_point_coverage"]
    assert cov["n_scored"] == 2
    assert abs(cov["macro"] - (0.5 + 1.0) / 2) < 1e-9
    assert abs(cov["micro"] - 2 / 3) < 1e-9


def test_mixed_items_full_pipeline():
    pipe = _Pipe({
        "ok": _ans("要点A", cited=["a"]),
        "no_cite": _ans(cited=[]),
        "wrong_refuse": _ans(refused=True),
        "should_refuse": _ans(refused=True),
        "leak": _ans(cited=["z"]),
    })
    items = [
        _q("ok", gold=["a"], points=["要点A"]),
        _q("no_cite", gold=["b"]),
        _q("wrong_refuse", gold=["c"]),
        _q("should_refuse", expect_refuse=True),
        _q("leak", gold=[], expect_refuse=True),
    ]
    r = eval_answer(pipe, items)
    assert r["n"] == 5
    assert r["refusal"] == {"tp": 1, "tn": 2, "fp": 1, "fn": 1}
    cp = r["citation_precision"]
    assert cp["n_scored"] == 1 and cp["n_no_citation"] == 1
    assert cp["n_refused_answerable"] == 1
