"""CLI 入口：python -m src.cli "你的问题"

启动顺序：配置 -> 组件初始化 -> BM25 从 Chroma 重建 -> 问答循环。
任何一步失败都显式抛错退出，不静默兜底（不要吞异常）。
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import ConfigError, load_config
from .generator import Generator
from .models import Answer
from .pipeline import RagPipeline
from .retriever.bm25_store import BM25Store
from .retriever.embedder import Embedder, LocalEmbedder, WordHashEmbedder
from .retriever.vector_store import ChromaVectorStore


def _build_embedder(cfg):
    """EMBEDDING_PROVIDER 路由：
    - fake:  WordHashEmbedder（完全本地、确定性哈希投影，仅贯通管线，
             **不验证语义检索有效性**）；
    - local: LocalEmbedder（sentence-transformers 本地推理，bge 等开源模型，
             不打外部 API、不交第三方费用，首次运行会下载模型权重）；
    - 其他:  OpenAI 兼容 Embedder（默认）。"""
    if cfg.embedding_provider == "fake":
        # 假 embedder 用 512 维（2 的幂）；与 cfg.embedding_dim 解耦，
        # 因为 cfg.embedding_dim 是给真供应商声明维度用的。
        return WordHashEmbedder(dim=512)
    if cfg.embedding_provider == "local":
        # 本地模型维度必须与 cfg.embedding_dim 一致（向量表 schema 按它建）；
        # LocalEmbedder 内部会做维度校验，不符直接抛错。
        return LocalEmbedder(cfg.embedding_model, cfg.embedding_dim)
    return Embedder(
        cfg.embedding_api_key, cfg.embedding_base_url,
        cfg.embedding_model, cfg.embedding_dim,
    )


def build_pipeline(env_path: str) -> RagPipeline:
    cfg = load_config(env_path)
    embedder = _build_embedder(cfg)
    vector_store = ChromaVectorStore(cfg.chroma_dir, cfg.chroma_collection, embedder)
    if vector_store.count() == 0:
        raise RuntimeError(
            f"Chroma 集合为空：请先运行 python -m src.ingest --env {env_path}"
        )
    bm25_store = BM25Store()
    bm25_store.build(vector_store.all_chunks())
    generator = Generator(cfg.llm_api_key, cfg.llm_base_url, cfg.llm_model)
    return RagPipeline(cfg, vector_store, bm25_store, generator)


def print_answer(ans: Answer) -> None:
    print("\n=== 回答 ===")
    print(ans.text)
    if ans.refused:
        print(f"\n[拒答] 原因: {ans.refusal_reason}")
        if ans.citations:
            print("\n=== 最接近的候选（供你自行判断） ===")
            for c in ans.citations:
                print(
                    f"[{c.ref_id}] {c.doc_name}"
                    + (f" / {c.heading_path}" if c.heading_path else "")
                    + f"  score={c.score}"
                )
                print(f"    {c.snippet}")
        return
    if ans.invalid_refs_removed:
        print("\n[提示] LLM 输出中存在无效引用编号，已被引用校验剔除。")
    print("\n=== 引用 ===")
    for c in ans.citations:
        marker = "  [注入标记]" if c.flagged_injection else ""
        print(
            f"[{c.ref_id}] {c.doc_name}"
            + (f" / {c.heading_path}" if c.heading_path else "")
            + f"  score={c.score}{marker}"
        )
        print(f"    {c.snippet}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="mini-rag CLI 问答")
    parser.add_argument("question", nargs="?", help="单次提问；不填则进入交互模式")
    parser.add_argument("--env", default=".env")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出（管道用）")
    args = parser.parse_args(argv)

    try:
        pipeline = build_pipeline(args.env)
    except ConfigError as e:
        print(f"配置错误: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001 — 初始化失败的兜底出口码
        print(f"初始化失败: {e}", file=sys.stderr)
        return 2

    if args.question:
        ans = pipeline.ask(args.question)
        if args.json:
            print(json.dumps(ans, ensure_ascii=False, default=lambda o: o.__dict__))
        else:
            print_answer(ans)
        return 1 if ans.refused else 0

    print("mini-rag 交互模式，输入 exit/quit 退出。")
    while True:
        try:
            q = input("\n问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if q.lower() in {"exit", "quit"}:
            return 0
        if not q:
            continue
        try:
            ans = pipeline.ask(q)
            print_answer(ans)
        except Exception as e:  # noqa: BLE001 — 单问失败不退出 REPL
            print(f"查询失败: {e}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
