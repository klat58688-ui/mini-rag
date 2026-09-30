"""评测入口：检索质量 + 拒答正确性 + 注入防护 + 对比消融。

指标定义见 README 第 9 节，此处只做"趋势报告"，不做统计显著性声明。

用法：
  python -m eval.run_eval --env .env --mode retrieval
  python -m eval.run_eval --env .env --mode refusal
  python -m eval.run_eval --env .env --mode injection
  python -m eval.run_eval --env .env --mode all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cli import build_pipeline
from src.retriever.fusion import rrf_fuse


def load_qa(path: Path) -> list[dict]:
    items = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"qa 文件第 {lineno} 行不是合法 JSON: {e}") from e
    required = {"question", "expect_refuse"}
    for i, it in enumerate(items):
        if not required.issubset(it):
            raise ValueError(f"qa 第 {i} 条缺少字段 {required - set(it)}")
    return items


def eval_retrieval(pipeline, items: list[dict]) -> dict:
    """Recall@10 / MRR@10，只统计 expect_refuse=False 的题。"""
    hits_at_k = 0
    rr_sum = 0.0
    n = 0
    details = []
    for it in items:
        if it.get("expect_refuse"):
            continue
        gold = set(it.get("gold_chunk_ids", []))
        v = pipeline.vector_store.search(it["question"], pipeline.cfg.vector_top_k)
        b = pipeline.bm25_store.search(it["question"], pipeline.cfg.bm25_top_k)
        v = pipeline._per_list_penalize(v)
        b = pipeline._per_list_penalize(b)
        fused = rrf_fuse([v, b], k=pipeline.cfg.rrf_k)
        top_ids = [c.chunk_id for c in fused[:10]]
        n += 1
        first_rank = next(
            (rank for rank, cid in enumerate(top_ids, 1) if cid in gold), None
        )
        if first_rank is not None:
            hits_at_k += 1
            rr_sum += 1.0 / first_rank
        details.append({
            "question": it["question"],
            "gold": list(gold),
            "top10": top_ids,
            "first_hit_rank": first_rank,
        })
    return {
        "n": n,
        "recall@10": hits_at_k / n if n else 0.0,
        "mrr@10": rr_sum / n if n else 0.0,
        "details": details,
    }


def _refusal_bucket(reason: str | None) -> str:
    """按拒答原因粗分类——P2 拆分，便于诊断是哪条规则触发的。"""
    if not reason:
        return "unknown"
    if "知识库为空" in reason:
        return "empty_index"
    if "注入检测标记" in reason:
        return "injection_bypass_blocked"
    if "检索相关性过低" in reason:
        return "low_cosine"
    if "INSUFFICIENT_CONTEXT" in reason:
        return "llm_insufficient_context"
    if "引用校验" in reason:
        return "citation_validation"
    return "other"


def eval_refusal(pipeline, items: list[dict]) -> dict:
    """拒答混淆矩阵：拒答题该拒（TP），可答题不该拒（TN）。
    P2：把拒答原因按规则拆分统计，便于看出"主要是哪条路径在工作"。"""
    tp = tn = fp = fn = 0
    by_rule: dict[str, int] = {}
    details = []
    for it in items:
        ans = pipeline.ask(it["question"])
        expect = bool(it.get("expect_refuse"))
        if expect and ans.refused:
            tp += 1
        elif expect and not ans.refused:
            fn += 1
        elif not expect and not ans.refused:
            tn += 1
        else:
            fp += 1
        if ans.refused:
            bucket = _refusal_bucket(ans.refusal_reason)
            by_rule[bucket] = by_rule.get(bucket, 0) + 1
        details.append({
            "question": it["question"],
            "expect_refuse": expect,
            "refused": ans.refused,
            "bucket": _refusal_bucket(ans.refusal_reason) if ans.refused else None,
            "reason": ans.refusal_reason,
        })
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "by_rule": by_rule, "details": details}


def eval_injection(pipeline, injection_dir: Path) -> dict:
    """喂入带注入载荷的文档，验证回答未被挟持。

    复核轮 2 新增：把 flagger 召回与端到端拦截率分列报告——
    - flagger_seen: 该 case 的引用 chunks 中是否有被 flagger 标中的（反映 flagger
      在 ingest 阶段的召回，作为纵深第一道）
    - intercepted: 该 case 是否被端到端拦截（refused 或未出现 forbidden_substring）
    两者独立计算：flagger 漏标 ≠ 一定被劫持（可能 penalize 后没进上下文），
    flagger 标中 ≠ 一定拦截成功（可能走 degraded 放行路径）。
    """
    results = []
    for path in sorted(injection_dir.glob("*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        ans = pipeline.ask(case["question"])
        target = case.get("forbidden_substring", "")
        hijacked = bool(target) and target in ans.text
        flagged_used = any(c.flagged_injection for c in ans.citations)
        intercepted = ans.refused or not hijacked
        results.append({
            "case": path.name,
            "question": case["question"],
            "refused": ans.refused,
            "degraded": getattr(ans, "degraded", False),
            "flagger_seen_in_citations": flagged_used,
            "hijacked": hijacked,
            "intercepted": intercepted,
            "text": ans.text,
        })
    n = len(results)
    attack_success = sum(1 for r in results if r["hijacked"])
    flagger_seen = sum(1 for r in results if r["flagger_seen_in_citations"])
    intercepted = sum(1 for r in results if r["intercepted"])
    degraded = sum(1 for r in results if r["degraded"])

    # 复核轮 3：再加一条与 case 完全独立的口径——ingest 阶段 flagger 在
    # 语料里实际标中了多少 chunk。这条不依赖任何 case 是否触达，直接反映
    # flagger 在入库时的召回面。
    try:
        all_chunks = pipeline.vector_store.all_chunks()
        flagged_total = sum(1 for c in all_chunks if c.flagged_injection)
        by_pattern: dict[str, int] = {}
        for c in all_chunks:
            for p in c.injection_patterns or []:
                by_pattern[p] = by_pattern.get(p, 0) + 1
        corpus_flagger = {
            "total_chunks": len(all_chunks),
            "flagged_chunks": flagged_total,
            "by_pattern": by_pattern,
        }
    except Exception:
        corpus_flagger = {"error": "vector_store 不支持 all_chunks"}

    return {
        "n": n,
        "attack_success_count": attack_success,
        "attack_success_rate": (attack_success / n) if n else 0.0,
        # 复核轮 2 分列：三个指标各自独立，相加不为 1
        "flagger_seen_rate": (flagger_seen / n) if n else 0.0,
        "intercepted_rate": (intercepted / n) if n else 0.0,
        "degraded_rate": (degraded / n) if n else 0.0,
        "corpus_flagger": corpus_flagger,
        "cases": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=".env")
    parser.add_argument("--qa", default="eval/qa.jsonl")
    parser.add_argument("--injection-dir", default="eval/injection")
    parser.add_argument("--mode", choices=["retrieval", "refusal", "injection", "all"],
                        default="all")
    args = parser.parse_args(argv)

    pipeline = build_pipeline(args.env)
    items = load_qa(Path(args.qa))

    report: dict = {}
    if args.mode in ("retrieval", "all"):
        report["retrieval"] = eval_retrieval(pipeline, items)
        r = report["retrieval"]
        print(f"[retrieval] n={r['n']}  recall@10={r['recall@10']:.3f}  mrr@10={r['mrr@10']:.3f}")
    if args.mode in ("refusal", "all"):
        report["refusal"] = eval_refusal(pipeline, items)
        r = report["refusal"]
        rules = " ".join(f"{k}={v}" for k, v in sorted(r["by_rule"].items()))
        print(f"[refusal] tp={r['tp']} tn={r['tn']} fp={r['fp']} fn={r['fn']}  by_rule: {rules}")
    if args.mode in ("injection", "all"):
        report["injection"] = eval_injection(pipeline, Path(args.injection_dir))
        r = report["injection"]
        print(
            f"[injection] n={r['n']} 攻击成功率={r['attack_success_rate']:.1%}  "
            f"flagger_seen={r['flagger_seen_rate']:.1%}  "
            f"intercepted={r['intercepted_rate']:.1%}  "
            f"degraded={r['degraded_rate']:.1%}"
        )
        cf = r.get("corpus_flagger", {})
        if "error" not in cf:
            print(
                f"  corpus_flagger: {cf['flagged_chunks']}/{cf['total_chunks']} chunks "
                f"被标记  by_pattern={cf['by_pattern']}"
            )

    out = Path("eval/results") / "last_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完整报告: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
