"""向量库：SQLite + sqlite-vec（cosine 距离手算，不依赖重型向量库）。

为什么换它而不是 Chroma：
- 纯 Python、零编译依赖，Windows/py3.13 也能 pip 一把过；
- 单一真相源天然满足：向量列 + 文本列 + 元数据列同一张表，见修订 3。

注意：sqlite-vec 的虚表本身只存向量，这里用"普通表 + 向量虚表"的小绑定模式——
nv/run 阶段文本与元数据存普通表，虚表只做近邻查询，两者按 chunk_id 对齐。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import sqlite_vec

from ..models import Chunk, ScoredChunk

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
  chunk_id    TEXT PRIMARY KEY,
  doc_id      TEXT NOT NULL,
  doc_name    TEXT NOT NULL,
  doc_path    TEXT NOT NULL,
  chunk_index INTEGER NOT NULL,
  heading_path TEXT NOT NULL,
  text        TEXT NOT NULL,
  flagged_injection INTEGER NOT NULL DEFAULT 0,
  injection_patterns TEXT NOT NULL DEFAULT ''
);
CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
  chunk_id TEXT PRIMARY KEY,
  embedding FLOAT[{dim}] distance_metric=cosine
);
"""





class ChromaVectorStore:
    """类名保留不变：调用方（pipeline/CLI/ingest）零改动。"""

    def __init__(self, persist_dir: Path, collection_name: str, embedder):
        if embedder is None:
            raise ValueError("VectorStore 需要 embedder（查询要向量化）")
        self.embedder = embedder
        self.dim = embedder.dim
        persist_dir.mkdir(parents=True, exist_ok=True)
        db_path = persist_dir / f"{collection_name}.db"
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)
        self._conn.executescript(_SCHEMA.format(dim=self.dim))

    def upsert(self, chunks: list[Chunk]) -> None:
        """按 chunk_id 幂等 upsert：用于首次入库或单点修复，
        不用于"同 doc 重灌"——后者请走 replace_doc（P0-1）。"""
        if not chunks:
            return
        embeddings = self.embedder.embed([c.text for c in chunks])
        with self._conn:
            for c, vec in zip(chunks, embeddings):
                self._conn.execute(
                    """INSERT INTO chunks
                       (chunk_id, doc_id, doc_name, doc_path, chunk_index,
                        heading_path, text, flagged_injection, injection_patterns)
                       VALUES (?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(chunk_id) DO UPDATE SET
                         doc_id=excluded.doc_id, doc_name=excluded.doc_name,
                         doc_path=excluded.doc_path, chunk_index=excluded.chunk_index,
                         heading_path=excluded.heading_path, text=excluded.text,
                         flagged_injection=excluded.flagged_injection,
                         injection_patterns=excluded.injection_patterns
                    """,
                    (
                        c.chunk_id, c.doc_id, c.doc_name, c.doc_path, c.chunk_index,
                        c.heading_path, c.text, int(c.flagged_injection),
                        ",".join(c.injection_patterns),
                    ),
                )
                # sqlite-vec 虚表不支持 UPSERT，先按 chunk_id 清掉再插入
                self._conn.execute(
                    "DELETE FROM vec_chunks WHERE chunk_id=?", (c.chunk_id,)
                )
                self._conn.execute(
                    "INSERT INTO vec_chunks(chunk_id, embedding) VALUES (?, ?)",
                    (c.chunk_id, sqlite_vec.serialize_float32(vec)),
                )

    def replace_doc(self, doc_id: str, chunks: list[Chunk]) -> None:
        """P0-1：同一 doc_id 下"先全量删除、再插入新块"，且必须在同一事务里。

        为什么必须同事务：若先 delete 再 upsert 分两步走，中途崩溃会留下"半灌"状态；
        若只 upsert 不 delete，旧文档块数比新文档多时，多余的旧块会残留并被检索到。
        chunks 的 doc_id 必须与参数 doc_id 一致，否则抛错——防御调用方传错。

        事务边界提示（复核轮 2 实测确认的坑）：Python sqlite3 传统（非 autocommit）
        模式下，`with self._conn:` 的 commit/rollback 范围是**整条连接上尚未提交的
        所有写入**，不止本函数体内的 DELETE+INSERT。这意味着：
          1. 若调用方在此函数之前做过未 commit 的写，本函数异常回滚会把它们一并吞掉；
             好处是整库 all-or-nothing，坏处是事务边界比函数体宽，调用方务必知晓。
          2. 测试 fixture 在调用 replace_doc 之前的种子数据**必须显式 commit()**，
             否则异常路径会把种子一起回滚，误得"数据全丢"的证伪结论。
        BM25 崩溃窗口说明：BM25 是**内存中的派生投影**，不落库；进程若在本函数
        commit 之后、ingest 流程触发 BM25 重建之前死掉，老 BM25 仍指向已删 chunk_id。
        下一次启动时 BM25 会从 chunks 表全量重建，恢复一致——因此此窗口只影响
        当前进程内的检索，重启即自愈，不做 DB 层事务并入。
        """
        if not chunks:
            raise ValueError(f"replace_doc 收到空 chunks（doc_id={doc_id}），与非空约定矛盾")
        for c in chunks:
            if c.doc_id != doc_id:
                raise ValueError(
                    f"replace_doc 收到 doc_id 不一致的 chunk: 期望 {doc_id}, 实得 {c.doc_id}"
                )
        embeddings = self.embedder.embed([c.text for c in chunks])
        with self._conn:  # 单事务：delete + insert 原子生效
            old_ids = [r[0] for r in self._conn.execute(
                "SELECT chunk_id FROM chunks WHERE doc_id=?", (doc_id,)
            )]
            for cid in old_ids:
                self._conn.execute("DELETE FROM vec_chunks WHERE chunk_id=?", (cid,))
            self._conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            for c, vec in zip(chunks, embeddings):
                self._conn.execute(
                    """INSERT INTO chunks
                       (chunk_id, doc_id, doc_name, doc_path, chunk_index,
                        heading_path, text, flagged_injection, injection_patterns)
                       VALUES (?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        c.chunk_id, c.doc_id, c.doc_name, c.doc_path, c.chunk_index,
                        c.heading_path, c.text, int(c.flagged_injection),
                        ",".join(c.injection_patterns),
                    ),
                )
                self._conn.execute(
                    "INSERT INTO vec_chunks(chunk_id, embedding) VALUES (?, ?)",
                    (c.chunk_id, sqlite_vec.serialize_float32(vec)),
                )

    def delete_by_doc_id(self, doc_id: str) -> None:
        with self._conn:
            ids = [r[0] for r in self._conn.execute(
                "SELECT chunk_id FROM chunks WHERE doc_id=?", (doc_id,)
            )]
            for cid in ids:
                self._conn.execute("DELETE FROM vec_chunks WHERE chunk_id=?", (cid,))
            self._conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def list_doc_ids(self) -> set[str]:
        return {r[0] for r in self._conn.execute("SELECT DISTINCT doc_id FROM chunks")}

    def _row_to_chunk(self, row) -> Chunk:
        (
            chunk_id, doc_id, doc_name, doc_path, chunk_index,
            heading_path, text, flagged, patterns,
        ) = row
        return Chunk(
            chunk_id=chunk_id, doc_id=doc_id, doc_name=doc_name, doc_path=doc_path,
            chunk_index=int(chunk_index), heading_path=heading_path, text=text,
            flagged_injection=bool(flagged),
            injection_patterns=[p for p in patterns.split(",") if p],
        )

    def all_chunks(self) -> list[Chunk]:
        rows = self._conn.execute(
            "SELECT chunk_id, doc_id, doc_name, doc_path, chunk_index,"
            " heading_path, text, flagged_injection, injection_patterns FROM chunks"
        ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def resolve(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        rows = self._conn.execute(
            f"SELECT chunk_id, doc_id, doc_name, doc_path, chunk_index,"
            f" heading_path, text, flagged_injection, injection_patterns"
            f" FROM chunks WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
        return {r[0]: self._row_to_chunk(r) for r in rows}

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        if top_k <= 0:
            raise ValueError(f"top_k 必须为正: {top_k}")
        q_vec = self.embedder.embed([query])[0]
        # 用 KNN 查询虚表
        rows = self._conn.execute(
            "SELECT chunk_id, distance FROM vec_chunks"
            " WHERE embedding MATCH ? AND k = ?",
            (sqlite_vec.serialize_float32(q_vec), top_k),
        ).fetchall()
        if not rows:
            return []
        ids = [r[0] for r in rows]
        meta = self.resolve(ids)
        out: list[ScoredChunk] = []
        for cid, dist in rows:
            chunk = meta.get(cid)
            cosine_sim = float(1.0 - dist)  # sqlite-vec cosine 距离
            out.append(
                ScoredChunk(
                    chunk_id=cid,
                    text=chunk.text if chunk else "",
                    vector_score=cosine_sim,
                    flagged_injection=chunk.flagged_injection if chunk else False,
                )
            )
        return out
