from __future__ import annotations

from pathlib import Path

from ..models import Document
from .base import DocumentLoader


class MarkdownLoader(DocumentLoader):
    extensions = (".md", ".markdown")

    def load(self, path: Path) -> Document:
        return self._make_document(path, "markdown", path.read_text(encoding="utf-8"))
