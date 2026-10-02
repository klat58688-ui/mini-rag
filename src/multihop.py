"""两跳检索扩展 —— 链式多跳题的本地实现（纯规则，不调 LLM）。

问题边界（§15.31）：
- "并列型多跳"（"X 怎么办，Y 怎么办？" 两主题词都在问句里）首轮 RRF 已能合并，
  不需要第二跳。
- "链式多跳"里有一种**局部失败模式**：用户用了**汇总词**（"加 X 一共"），其中 X
  是个**简称**（"餐补"），而完整术语（"餐饮补贴"）只出现在文档正文、未进 chunk
  text 的 heading（§15.25）。原 query 没有任何 token 能把它检索到。

本模块做的事情非常具体：
  1) 识别"汇总意图"（query 含 『加 AND 一共』 或类似固定搭配）。
  2) 若意图触发，对 query 里的**金额类简称**做一次**单点扩展**：
     简称 → 已知完整术语候选集 → 逐个发起第二跳检索。
  3) 把第二跳结果**与第一跳 fused**（再次过 RRF），一次性返回。

这不是真"推理型多跳"——我们没有让 LLM 看第一跳答案再去查第二跳。
它更像是**"查询局部扩展"**：识别出"汇总"这种高确定性的意图，
对问句里那个明显是简称的字串做单点扩展。诚实声明见 README §15.31。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ── 汇总意图触发 ─────────────────────────────────────────
# 用户明确要"加起来"时，才值得尝试补齐缺失的"金额维度"。
# 保守：只匹配固定的"加 X 一共"/"X 加 Y 一共"骨架，不做句式猜测。
_AGGREGATE_RE = re.compile(r"加[^，。?？]{1,20}?一[共起]|一共|总共|合计|加总")

# ── 简称 → 完整术语候选 ─────────────────────────────────
# 扩展表显式枚举，不当黑盒。每加一条都要在 README 里登账。
# value 的顺序即尝试顺序（先 most-likely）。
_ABBREVIATION_EXPANSIONS: dict[str, list[str]] = {
    "餐补": ["餐饮补贴", "伙食补贴", "伙食费"],
    "房补": ["住宿补贴", "住房补贴", "房租补贴"],
    "车补": ["交通补贴", "油卡补贴"],
    "差旅费": ["差旅报销", "差旅补贴", "出差报销"],
    "年终奖": ["年终奖金", "年终奖金额"],
}

@dataclass(frozen=True)
class Hop2Plan:
    """一次第二跳的计划：要不要做、用什么 query。

    `expanded_query` 为 None 表示"本轮不补跳"，调用方直接用一跳结果即可。
    """

    triggered: bool               # 汇总意图是否命中
    original_query: str
    expanded_query: str | None  # 若 None 表示无可扩展或扩展了也不比原 query 强
    matched_abbreviation: str = ""  # 命中的简称（诊断用）


def plan_hop2(query: str) -> Hop2Plan:
    """纯函数：判断要不要为这次查询做第二跳，给出扩展 query。

    满足全部条件才补跳：
      1. 汇总意图触发（`加...一共` / `总共` / `合计` / `加总`）；
      2. query 里**正好**出现扩展表里的某个简称（否则没有可扩展的抓手）。

    为什么不单加"金额语境词"过滤：扩展表本身（"餐补/房补/车补"……）天然就是
    金额词汇，`"1 加 1 一共"` 里没有这些字串，for 循环根本不会命中。
    再加一层"钱语境"判定是出于想象而非真实风险的死代码。
    """
    if not query or not query.strip():
        return Hop2Plan(triggered=False, original_query=query, expanded_query=None)
    query = query.strip()

    triggered = bool(_AGGREGATE_RE.search(query))
    if not triggered:
        return Hop2Plan(triggered=False, original_query=query, expanded_query=None)

    for abbr, full_forms in _ABBREVIATION_EXPANSIONS.items():
        if abbr not in query:
            continue
        expanded = query.replace(abbr, full_forms[0], 1)
        if expanded == query:
            continue
        return Hop2Plan(
            triggered=True,
            original_query=query,
            expanded_query=expanded,
            matched_abbreviation=abbr,
        )

    # 意图触发但无可扩展简称（expanded=None 供调用方/日志诊断区分
    # "没触发"与"触发了但没抓手"两种情况）。
    return Hop2Plan(triggered=True, original_query=query, expanded_query=None)


def merge_hop_results(
    hop1_fused: list, hop2_fused: list, rrf_fuse, rrf_k: int
) -> list:
    """把第二跳结果**并入**第一跳：再过一次 RRF，不污染每一路的原始分。

    实现策略很笨但可靠：把两路 fused 当作两张新 rank 列表再过 RRF。
    RRF 本来就是对"多个 rank 列表做融合"的算法——第一跳和第二跳本质上是
    '对同一意图的两次不同表达'，把它俩当作多路召回再合一次，语义干净。

    不直接用 `hop1_fused + hop2_fused` 拼接，因为拼接会让 hop1 的顺序死死压住
    hop2，即便 hop2 的某些块对真实问题更相关也永远排不上来。
    """
    return rrf_fuse([hop1_fused, hop2_fused], k=rrf_k)
