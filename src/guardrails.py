"""护栏：注入检测 + 拒答门控 + 引用校验。

设计原则（写进 README）：
1. 检测 + 降权 + 标注，不删除 —— 规则有误判，且彻底防御注入本就不现实；
2. 真正的纵深是"LLM 无任何工具调用权限"——即使注入了也做不了实事；
3. 引用校验只保证"编号的来源存在"（修订 3 给出可解析性），不管论断-片段对齐，
   后者需要 NLI，本项目不做，诚实写明。

修订 1 落地：门控读原始 cosine；BM25 强精确命中放行（错误码/缩写类 cosine 天然偏低）。
修订 2 落地：列内降权在融合前完成，RRF 输入即干净排名。
"""

from __future__ import annotations

import re

from .models import Answer, Citation, ScoredChunk

# ── 注入检测 ─────────────────────────────────────────────
_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ignore_instructions", re.compile(r"ignore (all )?(previous|prior|above) instructions?", re.IGNORECASE)),
    ("ignore_instructions_zh", re.compile(r"忽略(之前|以上|上述)(的)?(所有)?(指令|指示|命令)")),
    ("role_override_zh", re.compile(r"你现在是(一个|一名|新的)")),
    ("system_tag_spoof", re.compile(r"<\s*/?\s*(system|s|assistant)\s*>", re.IGNORECASE)),
    ("jailbreak_dan", re.compile(r"\bDAN\b.*\bjailbreak\b", re.IGNORECASE)),
    ("output_secret_zh", re.compile(r"(输出|泄露|给出)(你的)?(系统提 示词|prompt|密钥|api.?key)", re.IGNORECASE)),
]

_INVISIBLE_CHARS = re.compile(r"[\u200b\u200c\u200d\ufeff\u200e\u200f]")  # 零宽空格/ZWNJ/ZWJ/BOM/LRM/RLM，用转义写法避免源码中不可见字符被编辑器或工具链意外清除

# P0-2 旁路 C：自然语言里几乎不会出现的可疑特征（基于长度的启发式，
# 不是密码学论证——宁误报不漏报，因被标不会删除只会降权）
_B64_LIKE = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")  # 长 Base64 样串
_HEX_LIKE = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{32,}(?![0-9a-fA-F])")  # 长 16 进制串
_GIBBERISH_STAT = re.compile(r"\b[a-z]{5,}\b")  # 仅用于统计


def detect_injection(text: str) -> list[str]:
    """返回命中的注入模式名列表，空列表 = 干净。"""
    hits = [name for name, pat in _INJECTION_PATTERNS if pat.search(text)]
    if _INVISIBLE_CHARS.search(text):
        hits.append("invisible_chars")
    # P0-2 旁路 C 补强：混淆 payload 特征（Base64 / Hex / 高密度乱码词）
    if _B64_LIKE.search(text):
        hits.append("base64_payload_like")
    if _HEX_LIKE.search(text):
        hits.append("hex_payload_like")
    # 纯自然语言里连续 5+ 个小写乱码词（如 zxqwv qjvkz wurst）极少见。
    # 阈值 3 是 heuristic，不是 crypto 论证——错杀成本低于漏杀。
    gibberish_hits = len(_GIBBERISH_STAT.findall(text.lower()))
    if gibberish_hits >= 3 and detect_repeated_gibberish(text):
        hits.append("gibberish_cluster")
    return hits


def detect_repeated_gibberish(text: str) -> bool:
    """检测"短元音缺失的小写串密集出现"——case3 类攻击的特征。

    不强、不语义，只是启发式：自然中文/英文文本里同时出现 3+ 个
    "5 字母以上、无元音"的极不常见。误报多发生在代码片段/数学公式，
    只会降权不入狱，可接受。
    """
    vowel = re.compile(r"[aeiouAEIOU]")
    words = re.findall(r"[a-z]{5,}", text.lower())
    weird = [w for w in words if not vowel.search(w)]
    return len(weird) >= 3


# ── 拒答门控 ─────────────────────────────────────────────
def should_refuse(
    vector_hits: list[ScoredChunk],
    bm25_hits: list[ScoredChunk],
    query: str,
    *,
    cosine_threshold: float,
    bm25_strong_match: bool,
) -> tuple[bool, str]:
    """返回 (是否拒答, 原因)。

    双条件 AND：cosine 低 **且** 无可用的 BM25 强精确命中。
    P0-2：BM25 兜底被显式约束为"只放宽知识性拒答，永不放宽安全拒答"——
    若 BM25 top-1 已被注入检测标记，该兜底无条件失效，
    避免攻击者塞入高 IDF 词绕过门控。
    """
    if not vector_hits and not bm25_hits:
        return True, "知识库为空"
    top_cosine = max((h.vector_score or 0.0) for h in vector_hits) if vector_hits else 0.0
    if top_cosine >= cosine_threshold:
        return False, ""
    if bm25_strong_match and bm25_hits and not bm25_hits[0].flagged_injection:
        return False, ""  # cosine 低但 BM25 精确命中（错误码/缩写场景）→ 放行
    if bm25_strong_match and bm25_hits and bm25_hits[0].flagged_injection:
        return True, (
            f"cosine_top1={top_cosine:.3f} < {cosine_threshold}；"
            f"BM25 top-1 已被注入检测标记，强匹配兜底已禁用（P0-2）"
        )
    return True, f"检索相关性过低（cosine_top1={top_cosine:.3f} < {cosine_threshold}）"


# ── 引用后处理 ────────────────────────────────────────────
_REF_RE = re.compile(r"\[(\d+)\]")


def validate_and_map_citations(
    llm_text: str,
    contexts: list[ScoredChunk],
    resolved: dict,  # dict[str, Chunk]；延迟注解避免循环依赖
) -> Answer:
    """把 LLM 输出里的 [n] 映射回检索结果，并剔除幻觉引用。

    - 编号必须存在于本次 contexts（1..len(contexts)）；
      不存在的编号从文本中剔除并打 invalid_refs_removed=True。
    - 答案文本原样保留其余内容——内容是否忠实不在本层校验范围内（README 声明）。
    """
    if not contexts:
        raise ValueError("contexts 不能为空（调用前应已过拒答门控）")

    invalid_found = False

    def _replace(m: re.Match[str]) -> str:
        nonlocal invalid_found
        n = int(m.group(1))
        if 1 <= n <= len(contexts):
            return m.group(0)
        invalid_found = True
        return ""

    cleaned_text = _REF_RE.sub(_replace, llm_text).strip()

    cited_idxs = sorted({int(m.group(1)) for m in _REF_RE.finditer(cleaned_text)})
    citations: list[Citation] = []
    for n in cited_idxs:
        sc = contexts[n - 1]
        chunk = resolved.get(sc.chunk_id)
        if chunk is None:
            # 按架构这不应发生（chunk_id 全链路可回源），若发生说明存储层被破坏，抛错不吞。
            raise RuntimeError(f"引用无法回源解析，chunk_id 不在 Chroma: {sc.chunk_id}")
        score = sc.vector_score if sc.vector_score is not None else (sc.bm25_score or 0.0)
        snippet = chunk.text if len(chunk.text) <= 240 else chunk.text[:240] + "…"
        citations.append(
            Citation(
                ref_id=n,
                chunk_id=chunk.chunk_id,
                doc_name=chunk.doc_name,
                heading_path=chunk.heading_path,
                snippet=snippet,
                score=round(float(score), 4),
                flagged_injection=chunk.flagged_injection,
            )
        )

    # 引用校验在输出前兜底：如果一个有效引用都没有，按拒答处理（宁缺毋滥）
    if not citations:
        return Answer(
            text="知识库中的相关内容不足以支撑可靠回答。",
            refused=True,
            refusal_reason="LLM 输出未携带任何有效引用（已被引用校验拦截）",
            invalid_refs_removed=invalid_found,
        )

    return Answer(
        text=cleaned_text,
        citations=citations,
        refused=False,
        invalid_refs_removed=invalid_found,
    )
