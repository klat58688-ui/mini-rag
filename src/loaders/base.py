"""Loader 基类 + 按扩展名分发。

doc_id 用文件原始字节做 sha256（而非路径或 mtime）：
- 内容不变则 id 不变 -> 重跑 ingest 幂等（upsert 不重复）；
- 内容一变 id 即变 -> 旧块整体删除重写；
- 文件移动/重命名不影响（mtime 在跨机器拷贝时会变，故弃用）。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from ..models import Document


class DocumentLoader:
    extensions: tuple[str, ...] = ()

    def load(self, path: Path) -> Document:
        raise NotImplementedError

    def _make_document(self, path: Path, doc_type: str, text: str) -> Document:
        raw_bytes = path.read_bytes()
        doc_id = hashlib.sha256(raw_bytes).hexdigest()[:16]
        if not text.strip():
            raise ValueError(f"文档解析结果为空: {path}")
        return Document(id=doc_id, path=path, doc_type=doc_type, raw_text=text)


def dispatch_loader(path: Path) -> DocumentLoader:
    from .markdown_loader import MarkdownLoader
    from .pdf_loader import PdfLoader
    from .text_loader import TextLoader

    suffix = path.suffix.lower()
    for loader_cls in (MarkdownLoader, PdfLoader, TextLoader):
        if suffix in loader_cls.extensions:
            return loader_cls()
    raise ValueError(f"不支持的文档类型: {path}（支持 md/markdown/pdf/txt/text）")
