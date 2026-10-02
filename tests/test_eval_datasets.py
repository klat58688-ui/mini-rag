"""评测集的**离线**完整性测试（README §15.28）。

动机：本项目多次因为"重灌索引后 gold id 悬空"踩坑（§15.13 / §15.25 / §15.27）——
那些都是靠人工比对发现的。这类错误可以在**不加载索引**的前提下挡住大半：
JSONL 可解析、字段齐全、id 格式合法、可答/拒答两桶不串味。

真正"id 是否存在于索引"的校验必须跑 `eval.run_eval`（需要嵌入模型），不在本文件范围。
"""

import json
import re
from pathlib import Path

from eval.run_eval import load_qa

ROOT = Path(__file__).resolve().parent.parent
QA = ROOT / "eval" / "qa.jsonl"
PROBE = ROOT / "eval" / "probe.jsonl"

# chunk_id = sha256(doc_id:chunk_index) 取前 16 位十六进制
CHUNK_ID_RE = re.compile(r"^[0-9a-f]{16}$")


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ── qa.jsonl ─────────────────────────────────────────────

def test_qa_parses_and_has_required_fields():
    items = load_qa(QA)          # 内部会校验 question / expect_refuse
    assert len(items) > 0
    for it in items:
        assert set(it) >= {"question", "gold_chunk_ids", "expect_refuse", "gold_answer_points"}


def test_qa_questions_are_non_empty_and_unique():
    items = load_qa(QA)
    questions = [it["question"] for it in items]
    assert all(q.strip() == q and q for q in questions), "问题不应有首尾空白或空串"
    assert len(questions) == len(set(questions)), "存在重复问题"


def test_answerable_questions_have_valid_gold_ids():
    for it in load_qa(QA):
        if it["expect_refuse"]:
            continue
        golds = it["gold_chunk_ids"]
        assert golds, f"可答题必须至少 1 个 gold：{it['question']}"
        assert len(golds) == len(set(golds)), f"gold 有重复：{it['question']}"
        for g in golds:
            assert CHUNK_ID_RE.match(g), f"gold id 格式非法 {g!r}：{it['question']}"


def test_refuse_questions_carry_no_gold_ids():
    for it in load_qa(QA):
        if it["expect_refuse"]:
            assert it["gold_chunk_ids"] == [], f"拒答题不该有 gold：{it['question']}"


def test_qa_has_both_buckets_and_multi_hop_questions():
    items = load_qa(QA)
    answerable = [i for i in items if not i["expect_refuse"]]
    refusals = [i for i in items if i["expect_refuse"]]
    assert answerable, "缺少可答题"
    assert refusals, "缺少拒答题"
    # 多跳题（≥2 个 gold）是 recall@10 区分度的来源，见 §15.26
    assert any(len(i["gold_chunk_ids"]) > 1 for i in answerable), "缺少多 gold（多跳）题"


# ── probe.jsonl ──────────────────────────────────────────

def test_probe_parses_and_has_required_fields():
    rows = _load_jsonl(PROBE)
    assert rows, "probe.jsonl 为空"
    for r in rows:
        assert set(r) >= {"question", "expect_refuse", "topic", "difficulty"}


def test_probe_covers_both_classes():
    rows = _load_jsonl(PROBE)
    assert any(not r["expect_refuse"] for r in rows), "probe 缺少 relevant 样本"
    assert any(r["expect_refuse"] for r in rows), "probe 缺少 irrelevant 样本"


# ── injection cases ──────────────────────────────────────

def test_injection_cases_have_forbidden_substring():
    cases = sorted((ROOT / "eval" / "injection").glob("*.json"))
    assert len(cases) >= 5, "注入 case 少于 5 个"
    for path in cases:
        case = json.loads(path.read_text(encoding="utf-8"))
        assert case.get("question", "").strip(), f"{path.name} 缺 question"
        assert case.get("forbidden_substring", "").strip(), f"{path.name} 缺 forbidden_substring"
