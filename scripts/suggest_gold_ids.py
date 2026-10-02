"""为 qa.jsonl 中 expect_refuse=False 的题目打印 top-N 候选 chunk，辅助人工标注 gold_chunk_ids。

为什么需要这个脚本：
- eval/qa.jsonl 的 gold_chunk_ids 全部为空（历史遗留），导致 eval.run_eval --mode retrieval
  报表中的 recall@10 / mrr@10 永远 0.000，没有诊断意义。
- 人工标注需要看到「当前检索链路的 top-N 候选 + 文档名 + heading_path + snippet」，
  否则要去 sqlite 里手工 JOIN，太麻烦。

用法（在仓库根目录，确保 data/chroma 已灌库）：
    python scripts/suggest_gold_ids.py
    python scripts/suggest_gold_ids.py --top-k 8

输出后由人工逐题判断「哪一行 chunk 真正包含答案 / 是评估人认可的正解」，
把对应的 chunk_id 填进 eval/qa.jsonl 的 gold_chunk_ids 列表里。
本脚本不写 qa.jsonl，只打印。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cli import build_pipeline
from src.retriever.fusion import rrf_fuse


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=".env")
    ap.add_argument("--qa", default="eval/qa.jsonl")
    ap.add_argument("--top-k", type=int, default=8, help="打印候选数量")
    args = ap.parse_args()

    pipeline = build_pipeline(args.env)
    by_id = {c.chunk_id: c for c in pipeline.vector_store.all_chunks()}

    qa_path = Path(args.qa)
    items = [
        json.loads(line)
        for line in qa_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    for it in items:
        if it.get("expect_refuse"):
            continue
        q = it["question"]
        v = pipeline.vector_store.search(q, pipeline.cfg.vector_top_k)
        b = pipeline.bm25_store.search(q, pipeline.cfg.bm25_top_k)
        v = pipeline._per_list_penalize(v)
        b = pipeline._per_list_penalize(b)
        fused = rrf_fuse([v, b], k=pipeline.cfg.rrf_k)[: args.top_k]

        print("=" * 78)
        print(f"Q: {q}")
        print(f"   gold_answer_points: {it.get('gold_answer_points', [])}")
        for rank, sc in enumerate(fused, 1):
            c = by_id.get(sc.chunk_id)
            if c is None:
                continue
            snippet = c.text.replace("\n", " ")[:120]
            print(
                f"  [{rank}] {sc.chunk_id}  rrf={sc.rrf_score or 0:.4f}  "
                f"cos={sc.vector_score or 0:.3f}  bm25={sc.bm25_score or 0:.3f}"
            )
            print(f"      {c.doc_name}  >  {c.heading_path}")
            print(f"      {snippet}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
