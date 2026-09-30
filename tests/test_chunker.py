"""分块器测试：策略可配置性的边界条件全部锚定，不能只写进文档。"""

from pathlib import Path

import pytest

from src.chunkers.fixed_size import FixedSizeChunker
from src.chunkers.heading import HeadingChunker
from src.models import Document


def _doc(text: str, name: str = "x.md", doc_type: str = "markdown") -> Document:
    return Document(id="d" * 16, path=Path(name), doc_type=doc_type, raw_text=text)


class TestFixedSizeChunker:
    def test_basic_split_size_and_overlap(self):
        text = "a" * 1000
        c = FixedSizeChunker(chunk_size=400, overlap=80)
        chunks = c.split(_doc(text, "x.txt", "text"))
        assert len(chunks[0].text) == 400
        # step=320 → 各块起点 0/320/640/960，第 2 块起点应在第 1 块内部（重叠）
        assert chunks[1].text[:80] == chunks[0].text[320:400]

    def test_tail_chunk_covers_full_text(self):
        text = "a" * 410
        c = FixedSizeChunker(chunk_size=400, overlap=80)
        chunks = c.split(_doc(text, "x.txt", "text"))
        # 起点 0/320，stride 320、size 400 → 第 2 块覆盖 [320, 410)，长度 90
        assert chunks[-1].text == text[320:410]
        # 且拼接后能覆盖完整原文（用起点与长度验证连续覆盖性）
        start = 0
        for ch in chunks:
            assert text[start:start + len(ch.text)] == ch.text
            start = start + (400 - 80) if len(ch.text) == 400 else 410

    def test_overlap_must_less_than_size(self):
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=100, overlap=100)
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=100, overlap=-1)


class TestHeadingChunker:
    def test_no_heading_falls_back(self):
        text = "一段没有标题的文本，只是流水账，长度也短。"
        fallback = FixedSizeChunker(chunk_size=10, overlap=2)
        c = HeadingChunker(max_chunk_chars=50, fallback=fallback)
        chunks = c.split(_doc(text))
        # 应走 fallback（固定长度），而不是直接返回一大块
        assert all(len(ch.text) <= 10 for ch in chunks)

    def test_no_heading_no_fallback_raises(self):
        c = HeadingChunker(max_chunk_chars=50, fallback=None)
        with pytest.raises(ValueError, match="无标题"):
            c.split(_doc("全是没有标题的内容"))

    def test_heading_path_preserved(self):
        text = "# A\n\nintro\n\n## B\n\nbody\n\n## C\n\ntail\n"
        c = HeadingChunker(max_chunk_chars=200, fallback=None)
        chunks = c.split(_doc(text))
        paths = {ch.heading_path for ch in chunks}
        assert any("A" in p for p in paths)
        assert any(p.endswith("B") or "A > B" in p for p in paths)

    def test_oversized_section_split_by_paragraph(self):
        long_a = "字" * 200
        long_b = "符" * 200
        text = f"# H\n\n{long_a}\n\n{long_b}\n"
        c = HeadingChunker(max_chunk_chars=250, fallback=None)
        chunks = c.split(_doc(text))
        # 必须拆成至少 2 块，且各块都不超 250
        assert len(chunks) >= 2
        assert all(len(ch.text) <= 250 for ch in chunks)

    def test_chunk_id_deterministic_by_position(self):
        text = "# A\n\nx\n\n## B\n\ny\n"
        c = HeadingChunker(max_chunk_chars=200, fallback=None)
        chunks1 = c.split(_doc(text))
        chunks2 = c.split(_doc(text))
        assert [c.chunk_id for c in chunks1] == [c.chunk_id for c in chunks2]
