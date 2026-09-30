"""索引构建入口：扫描 docs/ -> 解析 -> 分块 -> 注入检测 -> 入库。

幂等性（修订 3 + P0-1）：
- doc_id = sha256(文件字节)。内容不变 → 同一 doc_id，replace_doc 内部删干净重灌。
- 内容变化 → 新 doc_id，旧 doc_id 由孤儿清理分支全量删除。
- 文件被删除 → 扫描时其 doc_id 消失，孤儿清理分支处理。
- P0-1：同一 doc 重灌时若新 chunk 数变少，必须有"先清后灌"兜底，
  否则旧 chunk 会带旧向量残留——见 vector_store.replace_doc 的事务语义。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .chunkers.fixed_size import FixedSizeChunker
from .chunkers.heading import HeadingChunker
from .config import AppConfig, load_config
from .guardrails import detect_injection
from .loaders.base import dispatch_loader
from .retriever.bm25_store import BM25Store
from .retriever.embedder import Embedder, LocalEmbedder, WordHashEmbedder
from .retriever.vector_store import ChromaVectorStore

log = logging.getLogger("mini-rag.ingest")


def choose_chunker(doc_type: str, cfg: AppConfig):
    fixed = FixedSizeChunker(
        chunk_size=cfg.fixed_chunk_size, overlap=cfg.fixed_chunk_overlap
    )
    if doc_type == "markdown":
        return HeadingChunker(max_chunk_chars=cfg.max_chunk_chars, fallback=fixed)
    return fixed


def iter_doc_paths(docs_dir: Path):
    if not docs_dir.is_dir():
        raise FileNotFoundError(f"docs 目录不存在: {docs_dir.resolve()}")
    for path in sorted(docs_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in (
            ".md", ".markdown", ".pdf", ".txt", ".text",
        ):
            yield path


def build_index(cfg: AppConfig) -> dict:
    # 与 cli._build_embedder 一致：EMBEDDING_PROVIDER 路由三种来源
    if cfg.embedding_provider == "fake":
        embedder = WordHashEmbedder(dim=512)
        log.warning(
            "EMBEDDING_PROVIDER=fake：使用 WordHashEmbedder，仅贯通管线，"
            "不验证语义检索有效性。"
        )
    elif cfg.embedding_provider == "local":
        embedder = LocalEmbedder(cfg.embedding_model, cfg.embedding_dim)
        log.info(
            "EMBEDDING_PROVIDER=local：sentence-transformers 本地推理 "
            f"({cfg.embedding_model}, dim={cfg.embedding_dim})，"
            "首次运行会下载模型权重到 HF 缓存。"
        )
    else:
        embedder = Embedder(
            cfg.embedding_api_key, cfg.embedding_base_url,
            cfg.embedding_model, cfg.embedding_dim,
        )
    vector_store = ChromaVectorStore(cfg.chroma_dir, cfg.chroma_collection, embedder)
    bm25_store = BM25Store()

    stats = {"loaded": 0, "skipped": 0, "errors": [], "flagged_chunks": 0}

    seen_doc_ids: set[str] = set()
    for path in iter_doc_paths(cfg.docs_dir):
        try:
            loader = dispatch_loader(path)
            doc = loader.load(path)
            seen_doc_ids.add(doc.id)

            chunker = choose_chunker(doc.doc_type, cfg)
            chunks = chunker.split(doc)

            for c in chunks:
                hits = detect_injection(c.text)
                if hits:
                    c.flagged_injection = True
                    c.injection_patterns = hits
                    stats["flagged_chunks"] += 1

            # P0-1：先清后灌包在 replace_doc 的事务里；同一 doc 重灌即使新块更少也不留残渣
            if not chunks:
                raise ValueError(f"{path.name} 分块结果为空，不应入库")
            vector_store.replace_doc(doc.id, chunks)
            stats["loaded"] += 1
            log.info("loaded %s -> %d chunks", path.name, len(chunks))
        except Exception as e:  # 单文档失败不中断整场，但要汇总到调用方
            msg = f"{path.name}: {type(e).__name__}: {e}"
            log.error("skip doc, %s", msg)
            stats["errors"].append(msg)
            stats["skipped"] += 1

    # 清理孤儿 doc_id（文档已被删除的情况）。已知集合 = 本轮扫到的 + 库里有的。
    existing = vector_store.list_doc_ids()
    orphans = existing - seen_doc_ids
    for doc_id in orphans:
        vector_store.delete_by_doc_id(doc_id)
        log.info("removed orphan doc_id=%s", doc_id)

    # BM25 是派生投影：从 Chroma 重建（修订 3 单一真相源）
    bm25_store.build(vector_store.all_chunks())
    stats["total_chunks"] = vector_store.count()
    return stats


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="构建 RAG 索引")
    parser.add_argument("--env", default=".env", help="配置文件路径，默认 .env")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.env)
    except Exception as e:
        print(f"配置加载失败: {e}", file=sys.stderr)
        return 2

    stats = build_index(cfg)
    print(f"完成：{stats['loaded']} 篇入库，{stats['skipped']} 篇跳过，"
          f"{stats['flagged_chunks']} 个注入标记，共 {stats['total_chunks']} chunk")
    if stats["errors"]:
        print("失败明细：", *stats["errors"], sep="\n  - ", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
