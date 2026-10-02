"""一次性对照实验：bge-large-zh-v1.5 在「加 / 不加官方查询指令前缀」下的 top-1 cosine。

背景：BAAI bge 系列论文/model card 建议——非对称检索（短 query 检索长 doc）时,
在 query 前加「为这个句子生成表示以用于检索相关文章：」。但这是在大语料/长文档上的结论。
本项目语料为 9 个 200~500 字的中文 FAQ 短 chunk，需要重新实测。

跑法：
    python scripts/bge_prefix_ab.py

实测结论（2026-09-30，bge-large-zh-v1.5 / dim=1024 / normalize_embeddings=True）：
| query                          | 无前缀  | 带前缀  | delta   |
|--------------------------------|---------|---------|---------|
| 今天上海的天气怎么样？         | 0.2241  | 0.2190  | -0.0051 |
| 退款申请提交后，多久收到钱？   | 0.6849  | 0.6727  | -0.0122 |
| 如何申请退款？                 | 0.6989  | 0.6240  | -0.0749 |

- `LocalEmbedder.embed` 默认**不**加前缀。
- 若未来语料变化/阈值失配，重跑此脚本再看是否需要打开前缀。
"""

from __future__ import annotations

import os

# 提前种子：huggingface.co 直连不通时改走镜像；不覆盖调用方显式设置的 HF_ENDPOINT。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from sentence_transformers import SentenceTransformer

QUERIES = [
    "今天上海的天气怎么样？",              # 离题：期望 cosine 远低于阈值
    "退款申请提交后，多长时间能收到钱？",  # 自然口语：期望 cosine 远高于阈值
    "如何申请退款？",                      # 擦边：语义接近但答案 chunk 内没"申请流程"
]

DOCS = [
    "退款会在 3 个工作日内原路退回。遇到法定节假日顺延到下一个工作日。",
    "七天内且未拆封支持无理由退款。已拆封商品需联系客服评估。",
    "开票后 30 天内可申请一次换开，税率信息有误可撤单重开。",
]

PREFIX = "为这个句子生成表示以用于检索相关文章："


def cosine(a: list[float], b: list[float]) -> float:
    # normalize_embeddings=True 下向量已单位化，点积即 cosine。
    return sum(x * y for x, y in zip(a, b, strict=True))


def main() -> None:
    st = SentenceTransformer("BAAI/bge-large-zh-v1.5")
    doc_embs = st.encode(
        DOCS, normalize_embeddings=True, convert_to_numpy=True
    ).tolist()

    print(f"{'query':<32}  {'无前缀':>10}  {'带前缀':>10}  delta")
    for q in QUERIES:
        e_plain = st.encode(
            [q], normalize_embeddings=True, convert_to_numpy=True
        ).tolist()[0]
        e_pfx = st.encode(
            [PREFIX + q], normalize_embeddings=True, convert_to_numpy=True
        ).tolist()[0]
        cos_plain = max(cosine(e_plain, d) for d in doc_embs)
        cos_pfx = max(cosine(e_pfx, d) for d in doc_embs)
        print(f"{q:<32}  {cos_plain:>10.4f}  {cos_pfx:>10.4f}  {cos_pfx - cos_plain:+.4f}")


if __name__ == "__main__":
    main()
