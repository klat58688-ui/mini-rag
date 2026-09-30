"""PDF 解析：pypdf 逐页提取，已知局限（README 有声明）：
- 无 OCR，扫描件提取为空 -> 直接抛错让用户感知，而不是静默入库空文档；
- 复杂双栏/表格会串行丢结构。
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from ..models import Document
from .base import DocumentLoader


class PdfLoader(DocumentLoader):
    extensions = (".pdf",)

    def load(self, path: Path) -> Document:
        reader = PdfReader(str(path))
        pages = []
        for i, page in enumerate(reader.pages):
            extracted = page.extract_text() or ""
            if extracted.strip():
                pages.append(f"[page {i + 1}]\n{extracted}")
        text = "\n\n".join(pages)
        if not text.strip():
            raise ValueError(
                f"PDF 未提取到文本（可能是扫描件，本系统不做 OCR）: {path}"
            )
        return self._make_document(path, "pdf", text)
