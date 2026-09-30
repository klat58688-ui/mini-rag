"""拒答闸门 cosine 阈值统计校准（针对 EMBEDDING_PROVIDER=local 的真实嵌入）。

策略：
- 不调用 LLM；只走 双路召回 → 列内降权 → RRF 融合 → 取 vector_hits 的 top-1 cosine
  + bm25_strong_match 判定应该拒答（这等价于 pipeline._gate 的内部行为，但允许我们
  在事后遍历不同阈值做扫描而不是只能用一个阈值）。
- probe 集：eval/probe.jsonl，relevant 26 条 + irrelevant 10 条。
- 输出：每个候选阈值下的 precision / recall / F1 / TP / FP / TN / FN 表格，
  以及被错分的具体 case（用于人工诊断）。

跑法：
    python -m eval.run_threshold_probe --env .env
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 让脚本既可作为 python -m eval.run_threshold_probe 运行，也可直接 python eval/run_threshold_probe.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cli import _build_embedder  # noqa: E402
from src.config import load_config  # noqa: E402
from src.guardrails import should_refuse  # noqa: E402
from src.retriever.bm25_store import BM25Store  # noqa: E402
from src.retriever.vector_store import ChromaVectorStore  # noqa: E402


def _load_probe(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _sweep(cases: list[dict], thresholds: list[float]) -> None:
    print(f"\n{'阈值':>6}  {'TP':>3} {'FP':>3} {'TN':>3} {'FN':>3}  "
          f"{'prec':>6} {'rec':>6} {'F1':>6}")
    print("-" * 60)
    for th in thresholds:
        tp = fp = tn = fn = 0
        for c in cases:
            if c["expect_refuse"]:
                # 期望拒答：cosine < th → 拒答（正确 TN）；否则误接收 FP
                if c["cosine_top1"] < th:
                    tn += 1
                else:
                    fp += 1
            else:
                # 期望放行：cosine >= th → 放行（正确 TP）；否则误拒 FN
                if c["cosine_top1"] >= th:
                    tp += 1
                else:
                    fn += 1
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        print(f"{th:>6.2f}  {tp:>3} {fp:>3} {tn:>3} {fn:>3}  "
              f"{prec:>6.3f} {rec:>6.3f} {f1:>6.3f}")


def _dump_misclassified(cases: list[dict], threshold: float) -> None:
    fp = [c for c in cases if c["expect_refuse"] and c["cosine_top1"] >= threshold]
    fn = [c for c in cases if not c["expect_refuse"] and c["cosine_top1"] < threshold]
    if fp:
        print(f"\n误接收（期望拒答但 cosine>={threshold}，{len(fp)} 条）:")
        for c in fp:
            print(f"  {c['cosine_top1']:.4f}  [{c['topic']:>12}] {c['question']}")
    if fn:
        print(f"\n误拒（期望放行但 cosine<{threshold}，{len(fn)} 条）:")
        for c in fn:
            print(f"  {c['cosine_top1']:.4f}  [{c['topic']:>12}] {c['question']}")
    if not fp and not fn:
        print(f"\n阈值 {threshold} 下 0 误分类 ✓")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env", default=".env")
    p.add_argument("--probe", default="eval/probe.jsonl")
    args = p.parse_args()

    repo = Path.cwd()  # 假定从 repo 根运行；与其它 CLI 一致
    cfg = load_config(repo / args.env)
    probe_path = repo / args.probe

    embedder = _build_embedder(cfg)
    vector_store = ChromaVectorStore(cfg.chroma_dir, cfg.chroma_collection, embedder)
    bm25_store = BM25Store()
    bm25_store.build(vector_store.all_chunks())
    print(f"probe 集: {probe_path}  阈值现状: {cfg.cosine_threshold}")

    probes = _load_probe(probe_path)
    print(f"载入 {len(probes)} 条: "
          f"relevant={sum(1 for r in probes if not r['expect_refuse'])} "
          f"irrelevant={sum(1 for r in probes if r['expect_refuse'])}")

    cases = []
    for i, r in enumerate(probes, 1):
        q = r["question"]
        vector_hits = vector_store.search(q, cfg.vector_top_k)
        bm25_hits = bm25_store.search(q, cfg.bm25_top_k)
        # 现状门槛判定
        bm25_strong = False
        if bm25_hits:
            top_ids = [h.chunk_id for h in bm25_hits[:3]]
            bm25_strong = bm25_store.strong_exact_match(q, top_chunk_ids=top_ids)
        refuse_now, reason = should_refuse(
            vector_hits, bm25_hits, q,
            cosine_threshold=cfg.cosine_threshold,
            bm25_strong_match=bm25_strong,
        )
        cosine_top1 = (vector_hits[0].vector_score or 0.0) if vector_hits else 0.0
        cases.append({
            "question": q,
            "topic": r["topic"],
            "difficulty": r["difficulty"],
            "expect_refuse": r["expect_refuse"],
            "cosine_top1": cosine_top1,
            "bm25_strong": bm25_strong,
            "current_decision_refuse": refuse_now,
            "current_reason": reason,
        })
        print(f"  [{i:>2}/{len(probes)}] cos={cosine_top1:.4f} "
              f"bm25_strong={bm25_strong} "
              f"current={'REFUSE' if refuse_now else 'PASS  '} "
              f"expect={'REFUSE' if r['expect_refuse'] else 'PASS  '}  {q}")

    # 阈值扫描
    thresholds = [round(0.20 + 0.02 * i, 2) for i in range(26)]  # 0.20 ~ 0.70
    _sweep(cases, thresholds)

    print("\n== 在当前阈值下的错分明细 ==")
    _dump_misclassified(cases, cfg.cosine_threshold)


if __name__ == "__main__":
    main()
