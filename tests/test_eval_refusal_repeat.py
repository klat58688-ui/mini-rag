"""refusal 重复采样聚合的单测（README §15.20）。

背景：refusal 段原本单次采样，边界题的 LLM 判定会抖，导致 fp/fn 带 ±1 噪声。
`aggregate_refusal_trials` 是纯函数，这里覆盖多数票、稳定性判定与平票规则；
再用一个 stub pipeline 验证 `eval_refusal(repeat=N)` 真的采样了 N 次。
"""

from eval.run_eval import aggregate_refusal_trials, eval_refusal


def _trial(refused: bool, bucket: str | None = None) -> dict:
    return {"refused": refused, "bucket": bucket, "reason": "x" if refused else None}


# ── aggregate_refusal_trials：纯函数 ──────────────────────────────

def test_all_refused_is_stable_and_decides_refuse():
    agg = aggregate_refusal_trials([_trial(True, "low_cosine")] * 3, True, "q")
    assert agg["n_trials"] == 3
    assert agg["n_refused"] == 3
    assert agg["refuse_rate"] == 1.0
    assert agg["stable"] is True
    assert agg["decision_refused"] is True
    assert agg["buckets"] == {"low_cosine": 3}


def test_all_answered_is_stable_and_decides_answer():
    agg = aggregate_refusal_trials([_trial(False)] * 4, False, "q")
    assert agg["refuse_rate"] == 0.0
    assert agg["stable"] is True
    assert agg["decision_refused"] is False
    assert agg["buckets"] == {}


def test_mixed_is_unstable_and_majority_wins():
    # 2/6 拒答——正是 README §15.20 里「如何申请退款？」的实测形态
    trials = [_trial(True, "llm_insufficient_context")] * 2 + [_trial(False)] * 4
    agg = aggregate_refusal_trials(trials, True, "如何申请退款？")
    assert agg["n_refused"] == 2
    assert abs(agg["refuse_rate"] - 1 / 3) < 1e-9
    assert agg["stable"] is False
    assert agg["decision_refused"] is False  # 少数派，按多数票判为"未拒答"
    assert agg["buckets"] == {"llm_insufficient_context": 2}


def test_even_split_ties_to_refuse():
    """N 为偶数时的 50/50 平票取保守侧（算拒答）。"""
    agg = aggregate_refusal_trials([_trial(True, "low_cosine"), _trial(False)], True, "q")
    assert agg["refuse_rate"] == 0.5
    assert agg["stable"] is False
    assert agg["decision_refused"] is True


def test_four_of_six_decides_refuse():
    trials = [_trial(True, "low_cosine")] * 4 + [_trial(False)] * 2
    agg = aggregate_refusal_trials(trials, True, "q")
    assert agg["decision_refused"] is True
    assert agg["stable"] is False


def test_empty_trials_does_not_crash():
    agg = aggregate_refusal_trials([], True, "q")
    assert agg["n_trials"] == 0
    assert agg["refuse_rate"] == 0.0
    assert agg["decision_refused"] is False


# ── eval_refusal：repeat 的集成行为 ──────────────────────────────

class _Ans:
    def __init__(self, refused: bool, reason=None):
        self.refused = refused
        self.refusal_reason = reason


class _StubPipeline:
    """按题目给出预设的逐次回答序列，并记录每题被问了几次。"""

    def __init__(self, plan: dict[str, list[bool]]):
        self.plan = {q: list(v) for q, v in plan.items()}
        self.calls: dict[str, int] = {}

    def ask(self, q: str) -> _Ans:
        self.calls[q] = self.calls.get(q, 0) + 1
        seq = self.plan.get(q, [])
        refused = seq.pop(0) if seq else False
        return _Ans(refused, "检索相关性过低" if refused else None)


def test_repeat_calls_ask_n_times_per_question():
    items = [{"question": "a", "expect_refuse": True}, {"question": "b", "expect_refuse": False}]
    pipe = _StubPipeline({"a": [True, True, False], "b": [False, False, False]})
    rep = eval_refusal(pipe, items, repeat=3)
    assert pipe.calls == {"a": 3, "b": 3}
    assert rep["n_trials_per_question"] == 3


def test_repeat_majority_drives_confusion_matrix():
    items = [{"question": "a", "expect_refuse": True}, {"question": "b", "expect_refuse": False}]
    # a：2/3 拒答 → 多数票判拒答 → tp；b：3/3 未拒答 → tn
    pipe = _StubPipeline({"a": [True, True, False], "b": [False, False, False]})
    rep = eval_refusal(pipe, items, repeat=3)
    assert (rep["tp"], rep["tn"], rep["fp"], rep["fn"]) == (1, 1, 0, 0)
    assert rep["unstable_count"] == 1
    assert rep["unstable"][0]["question"] == "a"
    # by_rule 在 repeat>1 时统计的是采样次数（这里 a 有 2 次拒答）
    assert rep["by_rule"] == {"low_cosine": 2}


def test_repeat_one_matches_single_shot_semantics():
    items = [{"question": "a", "expect_refuse": True}, {"question": "b", "expect_refuse": False}]
    pipe = _StubPipeline({"a": [False], "b": [True]})
    rep = eval_refusal(pipe, items, repeat=1)
    # a 该拒未拒 → fn；b 不该拒却拒了 → fp
    assert (rep["tp"], rep["tn"], rep["fp"], rep["fn"]) == (0, 0, 1, 1)
    assert rep["unstable_count"] == 0  # 单次采样恒稳定
