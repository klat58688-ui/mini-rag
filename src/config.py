"""配置加载：全部走环境变量（可用 .env），启动时校验、失败即报错退出。

不吞异常是本模块的原则：缺关键配置就死在启动，而不是跑到一半再炸。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


class ConfigError(RuntimeError):
    pass


def _load_dotenv(path: Path) -> dict[str, str]:
    """极简 .env 解析，不引入额外依赖。真实环境变量优先。"""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ConfigError(f"{path}:{lineno} 不是合法的 KEY=VALUE 行: {raw!r}")
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _get(env: dict[str, str], key: str, default: Optional[str] = None) -> str:
    if key in os.environ:
        return os.environ[key]
    if key in env:
        return env[key]
    if default is not None:
        return default
    raise ConfigError(f"缺少必需的配置项 {key}，请在环境变量或 .env 中设置（参考 .env.example）")


@dataclass(frozen=True)
class AppConfig:
    # LLM（OpenAI 兼容接口，换供应商只需改 base_url/model）
    llm_api_key: str
    llm_base_url: str
    llm_model: str
    # Embedding（同样 OpenAI 兼容；允许 chat 与 embedding 走不同供应商）
    embedding_api_key: str
    embedding_base_url: str
    embedding_model: str
    embedding_dim: int
    # "fake" 则 cli 层实例化 WordHashEmbedder 顶替 OpenAI 客户端。
    # 仅在没有外部 embedding 凭证时用于演示/贯通管线，**不验证语义检索**。
    embedding_provider: str = "openai"

    # 分块
    max_chunk_chars: int = 500      # 标题切分的块上限，超出按段落二次切分
    fixed_chunk_size: int = 400     # 固定长度切分（pdf/txt 兜底及无标题 md 的 fallback）
    fixed_chunk_overlap: int = 80

    # 检索
    vector_top_k: int = 20
    bm25_top_k: int = 20
    rrf_k: int = 60
    final_top_k: int = 5

    # 拒答门控（修订 1：用原始 cosine 分，而非 RRF 融合分）
    cosine_threshold: float = 0.35
    bm25_strong_match_enabled: bool = True

    # 注入防护（修订 2：融合前列内降权；flagged 项在每张召回列表内最多保留的名额）
    injection_max_penalized: int = 3
    # P1 消融开关：关掉后融合前不再对 flagged 项降权，
    # 用于 E4 对比实验"列内降权是否伤害合法查询的 Recall"。默认开。
    injection_penalty_enabled: bool = True

    # 目录
    docs_dir: Path = Path("docs")
    chroma_dir: Path = Path("data/chroma")
    chroma_collection: str = "mini_rag"


def load_config(env_path: str | Path = ".env") -> AppConfig:
    env = _load_dotenv(Path(env_path))
    cfg = AppConfig(
        llm_api_key=_get(env, "LLM_API_KEY"),
        llm_base_url=_get(env, "LLM_BASE_URL"),
        llm_model=_get(env, "LLM_MODEL"),
        embedding_api_key=_get(env, "EMBEDDING_API_KEY"),
        embedding_base_url=_get(env, "EMBEDDING_BASE_URL"),
        embedding_model=_get(env, "EMBEDDING_MODEL"),
        embedding_dim=int(_get(env, "EMBEDDING_DIM")),
        embedding_provider=_get(env, "EMBEDDING_PROVIDER", "openai").lower(),
        max_chunk_chars=int(_get(env, "MAX_CHUNK_CHARS", "500")),
        fixed_chunk_size=int(_get(env, "FIXED_CHUNK_SIZE", "400")),
        fixed_chunk_overlap=int(_get(env, "FIXED_CHUNK_OVERLAP", "80")),
        vector_top_k=int(_get(env, "VECTOR_TOP_K", "20")),
        bm25_top_k=int(_get(env, "BM25_TOP_K", "20")),
        rrf_k=int(_get(env, "RRF_K", "60")),
        final_top_k=int(_get(env, "FINAL_TOP_K", "5")),
        cosine_threshold=float(_get(env, "COSINE_THRESHOLD", "0.35")),
        bm25_strong_match_enabled=_get(env, "BM25_STRONG_MATCH", "true").lower() == "true",
        injection_max_penalized=int(_get(env, "INJECTION_MAX_PENALIZED", "3")),
        injection_penalty_enabled=_get(env, "INJECTION_PENALTY", "true").lower() == "true",
        docs_dir=Path(_get(env, "DOCS_DIR", "docs")),
        chroma_dir=Path(_get(env, "CHROMA_DIR", "data/chroma")),
        chroma_collection=_get(env, "CHROMA_COLLECTION", "mini_rag"),
    )
    if cfg.fixed_chunk_overlap >= cfg.fixed_chunk_size:
        raise ConfigError(
            f"FIXED_CHUNK_OVERLAP({cfg.fixed_chunk_overlap}) 必须小于 FIXED_CHUNK_SIZE({cfg.fixed_chunk_size})"
        )
    if not 0.0 <= cfg.cosine_threshold <= 1.0:
        raise ConfigError(f"COSINE_THRESHOLD 必须在 [0, 1] 区间: {cfg.cosine_threshold}")
    return cfg
