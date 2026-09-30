"""OpenAI 兼容 Embedding 客户端 + 仅用于演示/无网环境的 WordHashEmbedder。

- Embedder: 仅依赖 openai SDK 的 embeddings endpoint，任何兼容供应商可用；
- WordHashEmbedder: 完全本地、确定性的字符 n-gram 哈希投影。**不验证语义**——
  只保证"含相同字面 token 的 query 与 doc 有高 cosine"。用于：
    * 在没有外部 embedding 凭证时把 ingest → ask 全管线贯通（演示/CI 冒烟）；
    * 验证注入护栏、拒答门控、degraded 标记等"与语义无关"的下游逻辑。
  不应用于衡量检索质量、不应用于校准 cosine 阈值。生产环境必须用真实 embedding。
"""

from __future__ import annotations

import hashlib
import math
import os
import re

from openai import OpenAI

# 在导入 sentence_transformers 之前预设 HF 镜像：直连 huggingface.co 在国内不稳，
# hf-mirror.com 是社区维护的对称镜像。setdefault 保证不覆盖调用方显式设置的 HF_ENDPOINT。
# 只影响 LocalEmbedder 路径（Embedder / WordHashEmbedder 不接触 HF hub）。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


class Embedder:
    def __init__(self, api_key: str, base_url: str, model: str, dim: int,
                 batch_size: int = 32, timeout: float = 60.0):
        if dim <= 0:
            raise ValueError(f"embedding_dim 必须为正: {dim}")
        self.model = model
        self.dim = dim
        self.batch_size = batch_size
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            resp = self._client.embeddings.create(model=self.model, input=batch)
            ordered = sorted(resp.data, key=lambda d: d.index)
            for d in ordered:
                if len(d.embedding) != self.dim:
                    raise ValueError(
                        f"embedding 维度不符: 配置 EMBEDDING_DIM={self.dim}, "
                        f"实际 {len(d.embedding)}。请修正 .env"
                    )
                vectors.append(d.embedding)
        return vectors


_TOKEN = re.compile(r"[A-Za-z0-9]+|[一-鿿]")


class WordHashEmbedder:
    """确定性哈希嵌入：token → md5 低位 → 稀疏 one-hot 累加，最后归一化。

    特性：
    - "abc" 与 "abc" 的点积为 1 量级，与 "abd" 仅为共享 token 上的部分重叠；
    - 中英文混排安全（按 [A-Za-z0-9]+ 与单 CJK 字符切 token）；
    - 不做词序、不捕捉同义词——所以它对**关键词完全命中**的查询表现好，
      对"语义相近但字面不同"几乎返回 0。这是它的诚实边界。

    接口与 Embedder 一致：embed(texts: list[str]) -> list[list[float]]。
    """

    def __init__(self, dim: int = 512):
        if dim <= 0 or (dim & (dim - 1)) != 0:
            raise ValueError("WordHashEmbedder 的 dim 必须是 2 的幂 (推荐 512)")
        self.dim = dim
        self.model = "word-hash"

    def _tokens(self, text: str) -> list[str]:
        return _TOKEN.findall(text.lower())

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for tok in self._tokens(text):
                h = int.from_bytes(
                    hashlib.md5(tok.encode("utf-8")).digest()[:8], "little", signed=False
                )
                vec[h % self.dim] += 1.0
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            out.append([x / norm for x in vec])
        return out


class LocalEmbedder:
    """sentence-transformers 本地嵌入：真正把模型权重下到本机，CPU/GPU 都能跑。

    与 Embedder（OpenAI 兼容 HTTP）接口一致：embed(texts) -> list[list[float]]；
    - model 形如 "BAAI/bge-large-zh-v1.5"，首次调用会从 HuggingFace 下载 ~1GB 到
      `~/.cache/huggingface/hub/`，之后离线可用。
    - 不在此处做 normalize_embeddings 开关的硬编码：bge 系列**需要** normalize，
      用 sentence-transformers 默认 encode 逻辑往往会带 normalize=True 选项；
      这里显式传 True，保证 cosine 阈值语义稳定。
    - batch_size 默认 32 与 OpenAI 客户端保持一致；CPU 上 9 chunk 量级本身无所谓，
      仅在批量灌大语料时才会感知到差异。
    """

    def __init__(self, model: str, dim: int, batch_size: int = 32,
                 device: str | None = None):
        # 延迟 import：optional 依赖，只在 EMBEDDING_PROVIDER=local 时才需要装
        from sentence_transformers import SentenceTransformer
        self._st = SentenceTransformer(model, device=device)
        self.model = model
        self.batch_size = batch_size
        actual_dim = (
            self._st.get_embedding_dimension()
            if hasattr(self._st, "get_embedding_dimension")
            else self._st.get_sentence_embedding_dimension()
        )
        if actual_dim is not None and actual_dim != dim:
            raise ValueError(
                f"embedding 维度不符: 配置 EMBEDDING_DIM={dim}, "
                f"实际 {actual_dim}。请修正 .env"
            )
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embs = self._st.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [list(map(float, row)) for row in embs]
