from __future__ import annotations

from pathlib import Path

from ..models import Document
from .base import DocumentLoader


class TextLoader(DocumentLoader):
    extensions = (".txt", ".text")

    def load(self, path: Path) -> Document:
        return self._make_document(path, "text", path.read_text(encoding="utf-8"))
