"""Chunker 抽象 + chunk_id 生成规则（修订 3，全系统唯一实现处）。

chunk_id = sha256(f"{doc_id}:{chunk_index}")[:16]
- 确定性：同一文档同一序号永远得到同一 id；
- 幂等：重跑 ingest 结果完全一致，Chroma upsert 不重复；
- 16 hex = 64bit，千级 chunk 规模碰撞概率可忽略（README 注明该假设）。
"""

from __future__ import annotations

import hashlib

from ..models import Chunk, Document


def make_chunk_id(doc_id: str, chunk_index: int) -> str:
    return hashlib.sha256(f"{doc_id}:{chunk_index}".encode("utf-8")).hexdigest()[:16]


class Chunker:
    name: str = "base"

    def split(self, doc: Document) -> list[Chunk]:
        raise NotImplementedError

    def _wrap(self, doc: Document, pieces: list[tuple[str, str]]) -> list[Chunk]:
        """pieces: [(heading_path, text)] -> Chunk，强制连续赋序并生成 id。"""
        return wrap_chunks(doc, pieces)


def wrap_chunks(doc: Document, pieces: list[tuple[str, str]]) -> list[Chunk]:
    """全系统唯一的 Chunk 装配入口（修订 3：chunk_id 只此一处生成）。"""
    chunks: list[Chunk] = []
    for heading_path, text in pieces:
        cleaned = text.strip()
        if not cleaned:
            continue
        chunks.append(
            Chunk(
                chunk_id="",  # 占位，下方统一赋序后再生成 id
                doc_id=doc.id,
                doc_name=doc.path.name,
                doc_path=str(doc.path),
                chunk_index=len(chunks),
                heading_path=heading_path.strip(),
                text=cleaned,
            )
        )
    for index, c in enumerate(chunks):
        c.chunk_id = make_chunk_id(c.doc_id, index)
    return chunks
