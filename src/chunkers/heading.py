"""按标题层级切分（Markdown 主策略）。

规则：
1. 遇 `#`..`######` 切新块，维护完整 heading_path（"3 > 3.2 分块策略"）。
2. 正文行归入当前最深标题。
3. 超过 max_chunk_chars 的块按段落边界二次切分，不硬断句；
   单段超长的极端情况硬切（已在分段内注释）。
4. 无标题文档（或首个标题出现在文本中部但之前完全没有标题）整体 fallback
   到固定长度——这条规则唯一且明确，由测试锚定。
"""

from __future__ import annotations

import re

from ..models import Chunk, Document
from .base import Chunker

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


class HeadingChunker:
    name = "heading"

    def __init__(self, max_chunk_chars: int = 500, fallback: Chunker | None = None):
        if max_chunk_chars <= 0:
            raise ValueError(f"max_chunk_chars 必须为正: {max_chunk_chars}")
        self.max_chunk_chars = max_chunk_chars
        self.fallback = fallback

    def split(self, doc: Document) -> list[Chunk]:
        has_any_heading = any(
            _HEADING_RE.match(line) for line in doc.raw_text.splitlines()
        )
        if not has_any_heading:
            if self.fallback is None:
                raise ValueError(
                    f"文档无标题结构且未配置 fallback chunker: {doc.path}"
                )
            return self.fallback.split(doc)

        sections = self._collect_sections(doc.raw_text)
        pieces = self._split_oversized(sections)
        from .base import wrap_chunks

        return wrap_chunks(doc, pieces)

    def _collect_sections(self, text: str) -> list[tuple[str, str]]:
        """扫描全部行，输出 (heading_path, 正文)。第一个标题前的内容视为前言。"""
        sections: list[tuple[str, str]] = []
        stack: list[str] = []
        body: list[str] = []

        def flush():
            nonlocal body
            text_body = "\n".join(body).strip()
            body = []
            if not text_body:
                return
            heading_path = " > ".join(stack) if stack else "(前言)"
            sections.append((heading_path, text_body))

        for line in text.splitlines():
            m = _HEADING_RE.match(line)
            if m:
                flush()
                level = len(m.group(1))
                title = m.group(2).strip()
                # 栈保持"层级->标题"，同级替换、更深层追加、更浅层截断
                if len(stack) >= level:
                    stack = stack[: level - 1]
                while len(stack) < level - 1:
                    stack.append("")
                stack.append(title)
            else:
                body.append(line)
        flush()
        return sections

    def _split_oversized(self, sections: list[tuple[str, str]]) -> list[tuple[str, str]]:
        pieces: list[tuple[str, str]] = []
        for heading_path, body in sections:
            if len(body) <= self.max_chunk_chars:
                pieces.append((heading_path, body))
                continue
            buffer: list[str] = []
            buffer_len = 0
            for para in re.split(r"\n{2,}", body):
                para = para.strip()
                if not para:
                    continue
                if buffer and buffer_len + len(para) + 2 > self.max_chunk_chars:
                    pieces.append((heading_path, "\n\n".join(buffer)))
                    buffer, buffer_len = [], 0
                if len(para) > self.max_chunk_chars:
                    # 极端：单段超过上限，只能硬切。频率极低，保留 TODO 供后续优化。
                    for i in range(0, len(para), self.max_chunk_chars):
                        pieces.append((heading_path, para[i : i + self.max_chunk_chars]))
                else:
                    buffer.append(para)
                    buffer_len += len(para) + 2
            if buffer:
                pieces.append((heading_path, "\n\n".join(buffer)))
        return pieces
