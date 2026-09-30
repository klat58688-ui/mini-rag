"""分词：BM25 中文必需，英文按空格不够。

- 中文走 jieba；
- 保留英文/数字 token（专有名词、错误码 ERR_1042 靠它精确召回）；
- 附带轻量停用词表：只过滤纯标点与高频功能词，名词/术语绝不过滤
  （过滤错了会毁掉精确匹配兜底逻辑）。
"""

from __future__ import annotations

import re

import jieba

_STOPWORDS = {
    "的", "了", "是", "在", "我", "我们", "你", "你们", "他", "她", "它",
    "和", "与", "及", "或", "而", "被", "把", "让", "向", "从", "到",
    "就", "都", "也", "还", "又", "不", "没", "没有", "很", "更",
    "吗", "呢", "吧", "啊", "呀", "嘛", "么", "如何", "怎么", "什么", "哪", "哪些", "为什么",
    "这", "那", "这个", "那个", "这些", "那些", "一个", "一些", "以及",
    # T0 补充：中文术语/口语问询高频修辞词，从来不该作为 rare_tokens 命中要求。
    # 这些词出现在 query 末尾 ("X 是什么意思/干什么/怎么样/介绍/讲讲") 时，
    # 语料 chunk 不会有它们 → strong_exact_match 永远 False → glossary 场景被误拒，
    # 详见 README §15.10/§15.11。
    "意思", "干什么", "做什么", "怎么回事", "什么样", "怎样", "怎么样",
    "介绍", "介绍一下", "讲讲", "讲下", "解释", "解释一下", "说下", "说说",
    "错误", "一下", "能", "能不能", "可以", "可不可以", "请问", "想",
}

_ALNUM_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._\-]*")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in jieba.lcut(text):
        token = raw.strip().lower()
        if not token:
            continue
        if _ALNUM_RE.fullmatch(token):
            tokens.append(token)
        elif re.search(r"[一-鿿]", token):
            if token not in _STOPWORDS:
                tokens.append(token)
    return tokens


def rare_tokens(text: str) -> set[str]:
    """用于 BM25 强精确命中判断：查询里去掉停用词后的剩余 token 集合。"""
    return set(tokenize(text))
