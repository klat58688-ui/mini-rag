"""核心数据结构：全链路的单一数据契约。

约定：
- doc_id / chunk_id 为确定性哈希（见 loaders.base），重跑幂等。
- vector_score / bm25_score 由各自检索器写入，融合阶段只算 rrf_score。
- 全链路传 chunk_id，仅最终响应组装时回 Chroma 解析文档名/片段。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Document:
    id: str
    path: Path
    doc_type: str  # markdown / pdf / text
    raw_text: str


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_name: str
    doc_path: str
    chunk_index: int
    heading_path: str
    text: str
    flagged_injection: bool = False
    injection_patterns: list[str] = field(default_factory=list)

    def metadata(self) -> dict:
        """写入 Chroma 的元数据（Chroma 只接受标量）。"""
        return {
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "doc_path": self.doc_path,
            "chunk_index": self.chunk_index,
            "heading_path": self.heading_path,
            "flagged_injection": self.flagged_injection,
            "injection_patterns": ",".join(self.injection_patterns),
        }

    @staticmethod
    def from_store(chunk_id: str, text: str, meta: dict) -> "Chunk":
        patterns = meta.get("injection_patterns") or ""
        return Chunk(
            chunk_id=chunk_id,
            doc_id=meta.get("doc_id", ""),
            doc_name=meta.get("doc_name", ""),
            doc_path=meta.get("doc_path", ""),
            chunk_index=int(meta.get("chunk_index", 0)),
            heading_path=meta.get("heading_path", ""),
            text=text,
            flagged_injection=bool(meta.get("flagged_injection", False)),
            injection_patterns=[p for p in patterns.split(",") if p],
        )


@dataclass
class ScoredChunk:
    """召回/融合阶段的数据单元，携带三路分数，互不覆盖。"""

    chunk_id: str
    text: str = ""
    vector_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rrf_score: Optional[float] = None
    flagged_injection: bool = False


@dataclass
class Citation:
    ref_id: int  # 答案中出现的 [n]
    chunk_id: str = ""
    doc_name: str = ""
    heading_path: str = ""
    snippet: str = ""
    score: float = 0.0  # 检索阶段该 chunk 的相关分（优先 cosine，其次 BM25）
    flagged_injection: bool = False


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    refused: bool = False
    refusal_reason: Optional[str] = None
    invalid_refs_removed: bool = False  # LLM 引用了不存在的编号，已被后处理剔除
    # P0-2 旁路 B：放行路径上由 pipeline 兜底。
    # 若最终拼进 LLM 的 contexts 里仍含被标记的 chunk（比如 penalize 配额内
    # 挤掉了 or 攻击者写得跟查询语义足够近），这里强制标 True，
    # 让调用方能区分"干净回答"与"基于带毒证据的回答"。
    degraded: bool = False
