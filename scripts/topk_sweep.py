"""top-k 扫描：把 README §15.16 的诊断变成可证伪的测量。

§15.16 的结论是「语料规模 < 检索 top_k」导致三条观测通道退化：
  - `recall@10` 在 `语料 <= top_k` 时接近退化（gold 只有排最后才漏）；
  - `answer.degraded`（旁路 B）在 `干净 chunk 数 >= final_top_k` 时结构上不可达；
  - `flagger_seen_in_citations` 同理恒为 0。

本脚本用**同一份** eval/qa.jsonl + eval/injection/*.json，扫不同
(vector_top_k, bm25_top_k, final_top_k) 取值，直接观测：

  1. 候选池大小   每路实际返回多少 chunk（= min(top_k, 语料大小)）
  2. 检索质量     recall@10（真覆盖率）/ mrr@10
  3. 丢弃 gold    已进融合池、但排在 final_top_k 之后没进上下文的 gold 个数
  4. degraded 可达 有多少个 injection case 的 contexts 里真的出现了 flagged chunk

`final_top_k` 维度是 §15.27 新增的：它决定多少条进 LLM 上下文。
调大能把"已召回却被丢掉"的 gold 捡回来，但也会让 flagged 项更容易进上下文——
所以这两个数必须一起看。

**不调 LLM**：`degraded` 的判定条件是 `any(c.flagged_injection for c in contexts)`，
纯检索侧就能算，无需生成。所以本脚本很快（只付一次 embedder 加载 + N 次嵌入）。

用法（在仓库根目录）：
    python scripts/topk_sweep.py
    python scripts/topk_sweep.py --ks 5 --final-ks 5 8 10
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cli import build_pipeline
from src.retriever.fusion import rrf_fuse


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _retrieval(pipeline, items: list[dict]) -> tuple[float, float, int]:
    """真 recall@10（gold 覆盖率均值）/ mrr@10，只统计 expect_refuse=False 的题。"""
    cov_sum = 0.0
    rr = 0.0
    n = 0
    for it in items:
        if it.get("expect_refuse"):
            continue
        gold = set(it.get("gold_chunk_ids", []))
        v = pipeline._per_list_penalize(
            pipeline.vector_store.search(it["question"], pipeline.cfg.vector_top_k)
        )
        b = pipeline._per_list_penalize(
            pipeline.bm25_store.search(it["question"], pipeline.cfg.bm25_top_k)
        )
        top_ids = [c.chunk_id for c in rrf_fuse([v, b], k=pipeline.cfg.rrf_k)[:10]]
        n += 1
        cov_sum += (len(gold & set(top_ids)) / len(gold)) if gold else 0.0
        rank = next((r for r, cid in enumerate(top_ids, 1) if cid in gold), None)
        if rank is not None:
            rr += 1.0 / rank
    return (cov_sum / n if n else 0.0), (rr / n if n else 0.0), n


def _discarded_gold(pipeline, items: list[dict]) -> int:
    """**已召回但被 `final_top_k` 丢掉**的 gold 个数（README §15.27）。

    这是纯配置损失：信息在融合池里，只是没进 LLM 上下文。
    """
    lost = 0
    for it in items:
        if it.get("expect_refuse"):
            continue
        v = pipeline._per_list_penalize(
            pipeline.vector_store.search(it["question"], pipeline.cfg.vector_top_k)
        )
        b = pipeline._per_list_penalize(
            pipeline.bm25_store.search(it["question"], pipeline.cfg.bm25_top_k)
        )
        ids = [c.chunk_id for c in rrf_fuse([v, b], k=pipeline.cfg.rrf_k)]
        for g in it.get("gold_chunk_ids", []):
            if g in ids and ids.index(g) + 1 > pipeline.cfg.final_top_k:
                lost += 1
    return lost


def _pool_and_degraded(pipeline, inj_dir: Path) -> tuple[int, int, int]:
    """返回 (每路候选池大小, 触及 flagged 的 case 数, case 总数)。

    候选池大小取所有 injection 问题里「vector 召回条数」的最大值——
    在 top_k >= 语料大小时它恒等于语料大小，这正是"全量返回"的直接证据。
    """
    flagged_ids = {
        c.chunk_id for c in pipeline.vector_store.all_chunks() if c.flagged_injection
    }
    pool_max = 0
    reachable = 0
    paths = sorted(inj_dir.glob("*.json"))
    for path in paths:
        q = json.loads(path.read_text(encoding="utf-8"))["question"]
        raw_v = pipeline.vector_store.search(q, pipeline.cfg.vector_top_k)
        raw_b = pipeline.bm25_store.search(q, pipeline.cfg.bm25_top_k)
        pool_max = max(pool_max, len(raw_v), len(raw_b))
        v = pipeline._per_list_penalize(raw_v)
        b = pipeline._per_list_penalize(raw_b)
        fused = rrf_fuse([v, b], k=pipeline.cfg.rrf_k)
        contexts = fused[: pipeline.cfg.final_top_k]
        if any(c.chunk_id in flagged_ids for c in contexts):
            reachable += 1
    return pool_max, reachable, len(paths)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=".env")
    ap.add_argument("--qa", default="eval/qa.jsonl")
    ap.add_argument("--injection-dir", default="eval/injection")
    ap.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[5],
        help="要扫的 top_k 取值（vector 与 bm25 同步）",
    )
    ap.add_argument(
        "--final-ks",
        type=int,
        nargs="+",
        default=[5],
        help="要扫的 final_top_k 取值（决定多少条进 LLM 上下文）",
    )
    args = ap.parse_args()

    pipeline = build_pipeline(args.env)
    items = _load_jsonl(Path(args.qa))
    inj_dir = Path(args.injection_dir)

    all_chunks = pipeline.vector_store.all_chunks()
    n_clean = sum(1 for c in all_chunks if not c.flagged_injection)
    n_flag = sum(1 for c in all_chunks if c.flagged_injection)
    print(
        f"语料: {len(all_chunks)} chunk（干净 {n_clean} / flagged {n_flag}）  "
        f"rrf_k={pipeline.cfg.rrf_k}  评测集: {len(items)} 条"
    )
    print(
        "degraded 可达条件（§15.16）: contexts 里出现 flagged chunk "
        "⇒ 需要候选池中的干净 chunk 数 < final_top_k"
    )
    print(
        "「丢弃 gold」= 已进融合池、但排在 final_top_k 之后没进上下文的 gold 个数（§15.27）"
    )

    header = (
        f"\n{'top_k':>6} {'final_k':>8} {'候选池':>7} {'recall@10':>10} {'mrr@10':>8} "
        f"{'丢弃gold':>9} {'degraded可达':>13}"
    )
    print(header)
    print("-" * len(header))
    for k in args.ks:
        for fk in args.final_ks:
            pipeline.cfg = replace(
                pipeline.cfg, vector_top_k=k, bm25_top_k=k, final_top_k=fk
            )
            recall, mrr, _ = _retrieval(pipeline, items)
            lost = _discarded_gold(pipeline, items)
            pool, reach, total = _pool_and_degraded(pipeline, inj_dir)
            flag = "  ← 全量返回" if pool >= len(all_chunks) else ""
            print(
                f"{k:>6} {fk:>8} {pool:>7} {recall:>10.3f} {mrr:>8.3f} "
                f"{lost:>9} {f'{reach}/{total}':>13}{flag}"
            )
    print(
        f"\n注: 候选池 >= 语料({len(all_chunks)}) 时检索没有做选择；\n"
        f"    final_top_k 越大，越多「已召回」的 gold 能进上下文，但 flagged 项也更容易进上下文。"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
