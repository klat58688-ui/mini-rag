"""固定长度 + 重叠（PDF/纯文本，及无标题 Markdown 的兜底）。

按字符不按 token：简化、可解释、与中文场景对齐（中文一个字约等于一个语义单位），
README 承认这是粗糙但有意的取舍。

边界条件（测试锚定）：
- overlap 必须小于 size，由 config 层保证；
- step = size - overlap，保证前进；
- 最后不足 size 的尾块保留。
"""

from __future__ import annotations

from ..models import Document
from .base import Chunker


class FixedSizeChunker(Chunker):
    name = "fixed_size"

    def __init__(self, chunk_size: int = 400, overlap: int = 80):
        if chunk_size <= 0:
            raise ValueError(f"chunk_size 必须为正: {chunk_size}")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError(f"overlap 必须在 [0, chunk_size) 区间: overlap={overlap}, chunk_size={chunk_size}")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, doc: Document):
        text = doc.raw_text
        step = self.chunk_size - self.overlap
        pieces: list[tuple[str, str]] = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            pieces.append(("", text[start:end]))
            if end >= len(text):
                break
            start += step
        return self._wrap(doc, pieces)
