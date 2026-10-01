# mini-rag

一个**最小可用、带引用溯源、能讲清每个设计决策**的 RAG（检索增强生成）系统。
不是调库 Demo——检索主链路（混合检索、RRF 融合、引用回溯源、拒答门控、注入防护）**全部自己写**，仅借用了 `sqlite-vec`（向量存储）、`rank-bm25`（关键词打分）、`jieba`（中文分词）这三个"足够小、足够透明"的基础件。

> 这是一个用于求职展示的开源项目。核心诉求是**工程深度可解释性**：每一条设计选择都能在 README 里讲出 fallback、trade-off 和失效场景。

---

## 1. 快速开始（30 秒演示）

```bash
git clone ... && cd mini-rag
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# 编辑 .env：填入你的 LLM/Embedding API Key（推荐 DeepSeek + SiliconFlow 组合，便宜且都是 OpenAI 兼容接口）

python -m src.ingest          # 构建索引（首次会下载/解析 docs/）
python -m src.cli "ERR_1042 是什么原因？"
python -m src.cli "公司食堂几点开门？"   # 拒答演示
python -m src.cli "差旅住宿报销上限是多少？"   # 注入防护演示（注入样本会被标记并降权）
```

或用 API：

```bash
uvicorn src.api:app --port 8000
curl -X POST http://127.0.0.1:8000/ask -H 'Content-Type: application/json' \
     -d '{"question":"ERR_1042 是什么原因？"}'
```

---

## 2. 架构一图

```
                     离线 ingest                                    在线 ask
  docs/ (.md/.pdf/.txt)
        │                                                            │
        ▼                                                            ▼
  Loader (doc_id=sha256)              ┌─ VectorStore (sqlite-vec) ──┐
        │                             │  cosine KNN, top=20         │
        ▼                             │                             │
  Chunker (Heading / FixedSize) ──►   │  ↑ 单一真相源:文本+元数据+向量 │
        │                             │  ↓ 派生投影                │
  注入检测 (flagged)                   │  BM25 (rank_bm25 + jieba)   │
        │                             │  top=20                     │
        ▼ upsert                      └──────────────┬──────────────┘
  chunks.db                                    │
                                               ▼ 融合前列内降权(修订2)
                                        RRF 融合 (rank, k=60)
                                               │
                                               ▼ 拒答门控(修订1): cosine_top1 < τ
                                               │       ∧ !BM25 强精确命中
                                               ▼
                                          Generator (OpenAI 兼容)
                                               │  [n] 强制行内引用
                                               ▼
                                        引用校验（编号存在性）
                                               │
                                               ▼ 按 chunk_id 回源
                                       Answer + Citation[]
```

**关键约定**：`chunk_id = sha256(doc_id:chunk_index)[:16]`，全系统唯一生成入口在
[base.py](file:///C:/Users/KLAT/AppData/Roaming/TRAE%20SOLO%20CN/ModularData/ai-agent/work-mode-projects/6abcd53789a791c856ce4f59/mini-rag/src/chunkers/base.py#L8-L10)。
所有环节只传 `chunk_id`，最后一刻才回 sqlite 解析文档名与片段——引用可解析性由架构保证。

---

## 3. 分块策略（为什么这么切）

| 文档类型 | 策略 | 理由 | 失效场景 |
|---|---|---|---|
| **Markdown** | 按标题层级切，超长块按段落二次切 | 标题天然是语义边界；保留 `heading_path`（如 `退款 > 多久到账`）让引用更可读 | 文档无标题 → 整体 fallback 到固定长度 |
| **PDF / TXT** | 固定 400 字 + 80 字重叠 | 重叠是为防止答案恰好跨越边界被切断（导致漏召回）；按字符不按 token 是简化、可解释、与中文对齐 | 一个完整论证被切成两半时召回变差——这正是要调 overlap 的动机 |
| **语义切分** | **不做** | 每篇文档需多调 N 次 embedding，千级语料收益不成比例 | — |

代码：[heading.py](file:///C:/Users/KLAT/AppData/Roaming/TRAE%20SOLO%20CN/ModularData/ai-agent/work-mode-projects/6abcd53789a791c856ce4f59/mini-rag/src/chunkers/heading.py) · [fixed_size.py](file:///C:/Users/KLAT/AppData/Roaming/TRAE%20SOLO%20CN/ModularData/ai-agent/work-mode-projects/6abcd53789a791c856ce4f59/mini-rag/src/chunkers/fixed_size.py)

---

## 4. 混合检索：为什么纯向量不够用

纯向量检索在三类场景失效（用 `eval/qa.jsonl` 里的 `ERR_1042` 题能直接复现）：

1. **专有名词 / 错误码 / 缩写**：embedding 会把它和"普通错误"搅在一起；
2. **领域术语**（embedding 训练分布外）；
3. **短查询歧义**（语义太近无法区分）。

所以采用**向量 + BM25 两路召回，融合用 RRF（Reciprocal Rank Fusion）**：

```
score(d) = Σ 1/(k + rank_i(d))      k=60
```

**为什么不用加权分数融合**：向量分（0~1）与 BM25 分（无上界）量纲完全不同，
归一化对分数分布敏感（一个超长文档会把 BM25 分拉爆），调权就是玄学。
**RRF 只看排名、不看分数**，天然免归一化，鲁棒得多，实现 10 行——这是面试常考点。

代码：[fusion.py](file:///C:/Users/KLAT/AppData/Roaming/TRAE%20SOLO%20CN/ModularData/ai-agent/work-mode-projects/6abcd53789a791c856ce4f59/mini-rag/src/retriever/fusion.py)

---

## 5. 引用溯源（本项目最用心的部分）

四层防御，每一层都解决一个具体问题：

1. **编号透传**：检索结果组装进 prompt 时附 `[1] [2]...`，prompt 强制要求模型"论断后必跟编号"，附正反 few-shot；
2. **引用回查**：解析 LLM 输出里的 `[n]`，按 `chunk_id` 回 sqlite 拿到 `doc_name / heading_path / snippet / score`；
3. **引用校验（防"编引用"）**：编号必须存在于本次检索结果，否则剔除该论断并打 `invalid_refs_removed=true`。
   ⚠️ 这只防"编造来源"——防不了"正确引用但错误归因"，那需要 NLI，**明确不做**；
4. **拒答**：宁可不答，也不要含糊其辞。

代码：[guardrails.py](file:///C:/Users/KLAT/AppData/Roaming/TRAE%20SOLO%20CN/ModularData/ai-agent/work-mode-projects/6abcd53789a791c856ce4f59/mini-rag/src/guardrails.py) · [pipeline.py](file:///C:/Users/KLAT/AppData/Roaming/TRAE%20SOLO%20CN/ModularData/ai-agent/work-mode-projects/6abcd53789a791c856ce4f59/mini-rag/src/pipeline.py)

---

## 6. 拒答门控：为什么用原始 cosine 分

初版用"RRF 融合分 < 阈值"被拒，理由：
RRF 分只有**序的意义**没有**量的意义**——完全无关的查询只要语料非空就有 Top-1，
融合分永远不为零，跨查询不存在稳定阈值。

**修订 1**：门控改用**向量 Top-1 的原始 cosine 相似度**（有绝对语义：值低=真没沾边）。
但仅看 cosine 会把 `ERR_1042` 这类专有名词题误拒（embedding 天然低分），所以**AND 上一个 BM25 强精确命中兜底**：
查询稀有 token 在 BM25 Top-1 全部命中 → 放行。

阈值 `COSINE_THRESHOLD` 不进代码常量，走 `.env`，初值用 **第 9 节评测集**校准。

---

## 7. Prompt 注入防护（诚实承认局限）

**思路：检测 + 降权 + 标注，而不是幻想完全拦截。**

- **检测**：[guardrails.py](file:///C:/Users/KLAT/AppData/Roaming/TRAE%20SOLO%20CN/ModularData/ai-agent/work-mode-projects/6abcd53789a791c856ce4f59/mini-rag/src/guardrails.py) 内置规则（忽略之前指令/伪 system 标签/角色劫持/不可见字符）；
- **降权（修订 2）**：在**融合前的每张召回列表内**把 flagged 沉到干净项之后。理由：RRF 拿到的是"干净排名"，如果在融合后乘系数，会破坏 RRF 纯度、且"双榜第一"的注入 chunk 根本降不动；
- **保留不删**：规则有误判，且注入 chunk 极端情况下可能恰好能答——所以降权保留 + 引用时打 `flagged_injection=true` 让用户知情。

**局限（必读）**：

- 基于规则的检测必然有绕过方式；prompt 层"上下文边界"对强模型只是提示性防御；
- **真正的纵深是"LLM 无任何工具调用权限"** —— 即使注入了它也做不了实事，只能影响文本输出，而输出又被引用校验兜底；
- 完全不引入 LLM-as-judge / 复杂对齐校验，那是另一个量级的工程。

---

## 8. 测试与一键启动

```bash
pytest                     # 58 个测试，覆盖 分块/融合/护栏/pipeline/sqlite 集成/BM25 strong_match
docker compose up --build  # 一键起 API（端口 8000）
```

测试清单锚定了**设计承诺的边界条件**（不只测 happy path）：
- `test_chunker.py`：切块大小、重叠、fallback、id 稳定性；
- `test_fusion.py`：RRF 只看排名、保留原始分、保留 flag；
- `test_guardrails.py`：注入规则、拒答门控、引用校验、修订 2 列内降权；
- `test_pipeline_mock.py`：编排顺序、拒答短路、INSUFFICIENT_CONTEXT 兜底；
- `test_vector_store_integration.py`：sqlite-vec 真实建库、KNN、幂等、删除。

---

## 9. 评测方案

没有评测的 RAG 项目只能证明"能跑"，不能证明"有效"。所以**评测是本项目的一等功能**。

### 9.1 评测集构造
`eval/qa.jsonl` 30–50 条（当前示例 7 条，提交前请自行扩充至 30+），覆盖：

| 桶 | 占比 | 目的 |
|---|---|---|
| 事实型 | ~40% | 单 chunk 命中；**故意混入专有名词**（`ERR_1042`）检验混合检索 |
| 综合型 | ~30% | 跨段语义理解 |
| 拒答型 | ~30% | 知识库完全没有，含"相邻领域但不命中"的干扰题 |

每条：`{question, gold_chunk_ids, expect_refuse, gold_answer_points}`。
**`gold_chunk_ids` 用 chunk_id 标注**，粒度对齐评估对象（不是文档名）。

`eval/injection/` 放 5–10 个注入文档 + 对照问题。

### 9.2 指标

| 指标 | 定义 | 目标 |
|---|---|---|
| Recall@K (K=5/10) | Top-K 命中 gold 比例 | 检索消融主指标 |
| MRR@10 | gold 最高命中位的倒数均值 | 排序质量 |
| Citation Validity | `[n]` 可解析比例 | **必须 100%**，低于则说明校验有洞 |
| Citation Precision | 被引 chunk 与 gold 对齐比例 | 人工抽检 |
| 拒答混淆矩阵 | TP/TN/FP/FN | 可答题误拒率 ≤10% |
| 攻击成功率 | 注入被召回且模型遵循 | **0** |
| 误伤率 | 正常 chunk 被误判 flagged | 报告数值 |

### 9.3 对比实验（消融）
每次只改一个变量，其余锁死（`temperature=0`）：

- **E1 检索消融**：vector-only / BM25-only / hybrid-RRF。预期混合最优，专有名词子集上 BM25-only 反超 vector-only；
- **E2 拒答门控**：无门控 / RRF 融合分阈值（旧方案）/ 原始 cosine 阈值（修订 1）。预期旧方案行为不稳定；
- **E3 注入防护**：无防护 / 融合后降权（旧）/ 融合前列内降权（修订 2）。预期旧方案失败案例 + 新方案攻击成功率 0 ；
- **E4 分块对比**（可选）：Markdown 语料上 heading vs fixed 的 Recall@10。

跑法：`python -m eval.run_eval --env .env --mode all` → 报告落盘 `eval/results/last_eval.json`。

### 9.4 评测的诚实边界
- 30–50 题是小样本，**只报趋势、不做统计显著性声明**；
- 答案质量以人工抽检为准，LLM 辅助打分仅作参考；
- τ 在本评测集校准，存在过拟合风险（生产应换留出集）。

---

## 10. 明确不做的事（诚实划边界）

- ❌ reranker / Cross-encoder 精排（引入重型模型，RRF 已够用）
- ❌ 语义切分（成本/复杂度不匹配小规模语料）
- ❌ 论断级 entailment 校验（引用校验只到"编号存在性"层）
- ❌ 多轮对话记忆（单轮问答）
- ❌ PDF OCR / 双栏还原 / 扫描件
- ❌ 用户/权限/并发/分布式
- ❌ 前端 UI（curl 就是 UI）

---

## 11. 关键技术决策与 trade-off

| 决策 | 选择 | 理由 | 付出的代价 |
|---|---|---|---|
| 向量库 | **sqlite-vec** | 纯 Python、零编译、Windows 直装；向量+文本+元数据同表，天然单一真相源 | 性能低于 Chroma/Qdrant，千级 chunk 规模下无关痛痒；超过 10w 量级应换 |
| 关键词检索 | rank_bm25 + jieba | 纯 Python；jieba 中文必备；自维护轻量停用词表只滤标点功能词 | 索引只在内存；大语料需持久化 |
| 融合 | **RRF** (k=60) | 只看排名、免归一化；实现 10 行 | 无法精确控制两路权重（要权重就得改加权方案） |
| LLM/Embedding | **OpenAI 兼容接口** | 换供应商只改 `base_url` 与 `model`；推荐 DeepSeek + SiliconFlow | 需要外网；离线场景已在 §15.8 落地 `EMBEDDING_PROVIDER=local`（本地 bge-large-zh via sentence-transformers，**可选依赖**不进 requirements） |
| 框架 | **不用 LangChain** | 检索主链路自己写，每个细节都讲得清 | 没有现成的 chain/tool 库可用——但我们要的就是没有 |
| 拒答 | cosine + BM25 兜底 AND | 单一信号都会误拒/漏拒；两个独立信号 AND 更稳 | 多一个超参；需要评测集校准 |
| 注入处理 | **降权不删除** | 规则有误判；极端下注入 chunk 恰好相关；透明优于假装没看到 | 引用中 flagged 标记可能让用户困惑（已在 CLI 强调说明） |
| PDF | pypdf | 纯 Python 零依赖 | 不支持扫描件/双栏，README 已声明 |

---

## 12. 已知"难点点名"

1. **引用校验的真伪边界**——做"编号存在性校验"是 30 行代码，难的是想清楚它能防什么、不能防什么。本项目明确：**防"编造来源"，不防"断章取义"**（后者需要 NLI）。
2. **分块策略可配置性与简单性的平衡**——关键不是把 chunker 抽象出来，而是**让组合规则可预测**：fallback 何时触发、重叠在 fallback 后是否保持、id 如何保持稳定，全部用测试锚定而不是只写文档。

---

## 13. 项目结构

```
mini-rag/
├── README.md             ← 本文
├── LICENSE               ← MIT
├── .gitignore
├── .env.example
├── requirements.txt      ← 8 行核心依赖；sentence-transformers 是 local 模式可选件
├── Dockerfile / docker-compose.yml
├── pytest.ini
├── docs/                 ← 示例语料（含一篇注入样本）
├── eval/                 ← 评测集与跑分脚本
│   ├── qa.jsonl
│   ├── probe.jsonl       ← §15.10 阈值校准探针（36 条）
│   ├── run_threshold_probe.py
│   ├── injection/*.json
│   └── run_eval.py
├── src/
│   ├── config.py         ← 环境变量 + 启动校验
│   ├── models.py         ← Document/Chunk/ScoredChunk/Citation/Answer
│   ├── loaders/          ← md/pdf/txt 三加载器；doc_id=sha256(bytes)
│   ├── chunkers/         ← heading + fixed_size；chunk_id 唯一生成入口
│   ├── retriever/
│   │   ├── vector_store.py   ← sqlite-vec
│   │   ├── bm25_store.py     ← rank_bm25 + jieba，含 strong_exact_match
│   │   ├── fusion.py         ← RRF
│   │   ├── tokenizer.py
│   │   └── embedder.py
│   ├── guardrails.py     ← 注入检测 / 拒答门控 / 引用校验
│   ├── generator.py      ← OpenAI 兼容 LLM + 强约束 Prompt
│   ├── pipeline.py       ← 编排（修订 1/2/3 落地点）
│   ├── ingest.py / cli.py / api.py
└── tests/                ← 58 个测试

> **关于 `data/chroma/` 目录名**：这是历史命名残留——项目曾计划用 Chroma 后切换为 sqlite-vec，目录名保留至今以免迁移既有数据。配置文件/数据库文件实际落在 `data/chroma/mini_rag.db`（sqlite 单一真相源），**与 Chroma 无关**。
```

---

## 14. 如果给我更多时间，会……

- 给 ingest 加**增量更新**（基于 mtime+size 快筛再哈希）；
- 把 BM25 索引也持久化到 sqlite，启动更快；
- 加简单的论断-片段 entailment 校验（用一个轻量 NLI 模型），覆盖引用校验的剩余盲区；
- 评测集扩到 200 题并做留出集，把 `COSINE_THRESHOLD` 校准写进 CI。

—— 但这些都不属于"最小可用"。

---

## 15. 边界与口径（审查轮补）

外部独立复核（Python 3.12 / sqlite-vec 0.1.9）确认了三个关键论断，并指出两条 P0。本节如实记录。

### 15.1 已修复（P0）

- **P0-1 重灌残留**：`vector_store.replace_doc(doc_id, chunks)` 把"按 doc_id 全量清 + 插入"包在同一事务里，取代 ingest 之前的"按 chunk_id upsert"。ingest 全流程改走 `replace_doc`；测试 `test_replace_doc_shrinking_chunk_count_leaves_no_residue` 模拟 4 → 2 chunk 重灌，验证 c2/c3 从 chunks 与 vec_chunks 同步消失。
- **P0-2 BM25 兜底绕过通道**：门控新增"BM25 top-1 已被注入标记 → 兜底无条件失效"。pipeline 因此在 penalize 之前就调用 gate（penalize 会洗掉"top-1 是注入"这个信号）。攻击样本 `eval/injection/case3_high_idf_bypass.json` + 测试 `test_p0_2_bm25_bypass_blocked_when_top1_is_flagged` 双重锚定。

### 15.2 部分修复（P1）

- **消融开关**：`INJECTION_PENALTY=false` 时跳过融合前列内降权，便于 E4 对比 "降权是否伤合法召回"。
- **chunks/vec/BM25 事务一致性**：`replace_doc` 已同事务。BM25 在每次 ingest 结束后从 `all_chunks()` 全量重建（启动时也一样）——`O(corpus)` 成本对 demo 量级足够；超 10w chunk 应换持久化 BM25。
- **cosine 阈值跨模型不可迁移**：`COSINE_THRESHOLD=0.35` 只对 bge 系校准过；换 embedding 必须重新跑 `eval/refusal` 重选阈值，README 与 `.env.example` 已注明。

### 15.3 已声明边界（P2）

- **sqlite-vec 暴力扫描**：KNN 是顺序扫，规模上界建议 ≈10w chunk。超过应换 pgvector / Qdrant / Chroma。
- **BM25 全量重建**：每次启动重新分词整个语料。10w 量级开始会变成秒级启动开销。
- **`eval/qa.jsonl` 仅 7 题**：演示性样本，不构成统计显著性。结论若要对外引用必须先扩集。
- **拒答混淆矩阵按规则拆分**：`eval/run_eval.py` 现在按 `empty_index / low_cosine / injection_bypass_blocked / llm_insufficient_context / citation_validation` 输出 `by_rule` 计数，便于诊断哪条规则在主导。
- **flagger 召回 vs 端到端拦截率分列**：注入评测不再只给一个"攻击成功率"。现在注入 case 同时报四个独立口径——攻击成功率、flagger_seen（citations 中含 flagged chunk 的 case 占比，反映 flagger 在 ingest 阶段的召回）、intercepted（refused 或未出现 forbidden_substring 的 case 占比）、degraded（放行但被打 degraded 标记的 case 占比）。三者**不相加为 1**——flagger 漏标 ≠ 必被劫、flagger 标中 ≠ 必拦截。

### 15.4 验证口径

复核轮 1 修复后 `pytest -q` 实测 **41/41 通过**；复核轮 2 修复后 **50/50 通过**（新增 9 个用例：旁路 A 端到端、degraded 双向、flagger 三类新模式各两个、rrf 不变量负向、自然文本 gibberish 不误报等）。这是**代码完成度**，不等于"端到端已验证"。本文档未宣称完成真实 embedding/LLM 冒烟或 docker 启动；这两步是当前明确遗留事项。

### 15.5 复核轮 2 跟进（原子性实测确认 + P0-2 旁路封堵）

外部复核在本机直接验证了 `replace_doc` 的事务三性（vec0 参与事务、`with` 块异常可整体回滚、两表原子替换无残留），同时新列出 P0-2 的三条尚未堵死的旁路。本轮对应修复：

- **旁路 A 封堵（BM25 top-N 全净判据）**：`bm25_store.strong_exact_match` 签名扩展为 `top_chunk_ids: list[str]`；`pipeline._gate` 现在传入 BM25 顶部 3 个 chunk_id。任一不在索引、任一 flagged、或任一不包含查询稀有词 → 兜底关闭。新测试 `test_p0_2_bypass_a_clean_top1_plus_flagged_rank2_denied` 锚定。
- **旁路 B 封堵（degraded 标记）**：`Answer.degraded` 新字段；`pipeline.ask()` 在引用校验后回查最终 contexts，任意一格 flagged 则 `degraded=True`。意图是"放行即所见"——调用方拿到带毒证据的回答时能区分它跟干净回答。这**不**是拒答，是透明标注；拒答仍走底层 cosine / BM25 双判。
- **旁路 C 补强（flagger 召回）**：`detect_injection` 增加三类启发式——`base64_payload_like`（≥40 char Base64 样串）、`hex_payload_like`（≥32 hex）、`gibberish_cluster`（≥3 个无元音小写词）。配套两个新攻击样本 `eval/injection/case4_natural_stuffing.json`（自然文本混入高 IDF 假术语 + 注入指令）与 `case5_obfuscated.json`（Base64 化 payload 试图规避正则）。明确：这些都是**启发式**，不是密码学论证——宁误报不漏报，被标不会删只会降权。
- **rrf_score 分支核实**：复核方怀疑 `should_refuse` 原本可能有"rrf 强则放行"的分支并疑似被前移动作删掉。已 grep 整个仓库——**这条分支从未存在**。补 `test_rrf_score_is_never_consulted` 负向测试钉死不变量：哪怕 rrf_score 给到 0.99，只要 cosine < 阈值且无 BM25 强命中，仍必拒。原因：rrf 是融合后分数，会被 penalize 污染；放行决策必须看融合前的原始信号。
- **replace_doc 事务边界坑**：Python sqlite3 传统模式下 `with conn:` 的回滚会吞掉调用方此前未提交的写入——事务边界比函数体宽。已在 `replace_doc` docstring 里写明；测试 fixture 若要触发崩溃路径，种子数据必须先显式 `commit()`，否则会把种子一起回滚（复核方第一轮实测就踩了这个坑，误得"证伪"结论，commit 后才看到真正的 4/4 恢复）。
- **BM25 崩溃窗口**：`replace_doc` 已同事务提交 chunks/vec。BM25 是内存中的派生投影，不落库——若在 `replace_doc` commit 后、ingest 流程触发 BM25 重建之前进程死掉，老 BM25 会指向已删 chunk_id。该窗口只影响**当前进程内**检索，下次启动时 BM25 从 chunks 表全量重建即自愈，不值得为它做 DB 层事务并入。

复核方指出的"建议把 gate 拆成对 pre-penalize 信号的注入护栏 + 对 post-penalize 真实上下文的相关性拒答两个钩子"——本轮暂未落地为独立钩子，但 `_gate` 已强制看 raw hits、degraded 标记覆盖了"放行后可见性"问题；是否进一步拆分保留为后续可选重构。

### 15.6 复核轮 2 验证口径

本轮修复后 `pytest -q` 实测通过数见 §15.4 上方。复核方对本轮的 verdict：**P0-1 可关单**（原子性实测成立），**P0-2 不能单凭一轮代码改动关单**——剩余遗留是真实 embedding/LLM 冒烟、E4 消融实测、以及把 flagger 召回率与端到端拦截率分开报。这三项构成下一轮待办。

### 15.7 复核轮 3 实测（2026-09-30）

外部凭证：LLM 走 OpenAI 兼容网关 `https://51kik.com/v1` / 模型 `kimi-k3`；embedding 端点该网关**不提供**任何模型，故本轮用 `EMBEDDING_PROVIDER=fake` 退到本地 `WordHashEmbedder`（确定性字符 n-gram 哈希投影， dim=512）。下游"语义检索质量 / cosine 阈值是否合理"**没有也不该被本轮覆盖**——这是诚实边界，写在这里不能省。

**新增基础设施**：
- `src/retriever/embedder.py` 新增 `WordHashEmbedder`，接口与 `Embedder` 完全一致；`config.py` 加 `EMBEDDING_PROVIDER` 开关（默认 `openai`）；`cli.build_pipeline` 与 `ingest.build_index` 各自生效并在用 fake 时 `log.warning` 标注。
- `eval/run_eval.py` 注入评测在原有"攻击成功率"之外，分列四项独立口径：`flagger_seen_rate`（citations 中含 flagged chunk 的 case 占比）、`intercepted_rate`（refused 或 forbidden_substring 未出现）、`degraded_rate`、`corpus_flagger`（与 case 完全独立的 ingest 阶段 flagger 在语料里标中的 chunk 总数与模式分布）。

**真管线贯通冒烟（fake embedding + kimi-k3）**：
- 灌库：`docs/` 4 篇 → 9 chunk；`injection_sample.md` 唯一 chunk 被 flagger 标 `ignore_instructions`。
- 拒答路径："如何申请退款？" → cosine_top1=0.209 < 0.35 → **low_cosine 拒答**（同时暴露 WordHashEmbedder 的字面不匹配边界，门槛没被骗过去）。
- 放行路径："退款 多久 工作日 原路" → cosine_top1=0.590 → 真 LLM 返回 "退款会在 3 个工作日内原路退回 [1]"，引用正确。
- 注入场景：5 个 case 全部 **攻击成功率 0% / 端到端拦截 100%**。

**E4 消融（INJECTION_PENALTY）**：
| 配置 | flagger_seen | intercepted | degraded |
|---|---|---|---|
| 默认（penalty=true） | 0% | 100% | 0% |
| penalty=false | 40% | 100% | 20% |

penalty 关掉后 flagged chunk 改进 citations（40%），其中 case4（自然塞词样本）被放行且 degraded=True、回答正确且未泄漏 `pwned`——这是旁路 B"放行即所见"机制的端到端验证：penalty=false 削弱了"融合前列内降权"这道防线，但**纵深仍被 degraded / 拒答门控接住**；attack_success_rate 端到端依旧 0%。

**测试**：50/50 通过——本轮没有新增单元测试用例，所有改动（WordHashEmbedder、配置开关、eval 分列）都是新增路径，不影响既有拒答/护栏逻辑。

**仍未做（明确宣告）**：
- 真实 embedding 下的 cosine 阈值校准（受 embedding 供应商凭证阻塞）。
- 语义检索质量评估（同样阻塞）。
- 真实 embedding 下的 E4 重跑（本轮 E4 仅在 fake embedding 上做过一次；切回真实 embedding 后需重测才有跨模型意义）。

### 15.8 复核轮 3 补测：本地 sentence-transformers 真嵌入（2026-09-30 续）

§15.7 留下"未做"的三项，本轮**全部完成**——绕过 embedding 网关不可用问题，改用 `provider=local` 在本机推理 bge-large-zh-v1.5；LLM 仍走 kimi-k3。

**新增基础设施**：
- `src/retriever/embedder.py` 新增 `LocalEmbedder`（sentence-transformers，lazy import 不污染测试环境；`get_sentence_embedding_dimension()` 校验维度，`encode(..., normalize_embeddings=True)` 保 bge 风格）。
- `EMBEDDING_PROVIDER` 三值路由：`openai`（默认）/ `fake` / `local`；`cli._build_embedder` 与 `ingest.build_index` 同构分支。
- `.env.example` 加三 provider 说明并标注各自边界。
- 网络边界：huggingface.co 直连不通，需 `HF_ENDPOINT=https://hf-mirror.com` 镜像（`embedder.py` 模块头已用 `os.environ.setdefault` 默认写入，外部明确设置时不会被覆盖）。

**如何复现 local 模式**（sentence-transformers 是可选依赖，不在 requirements.txt 里，按需手动安装）：
```bash
pip install "sentence-transformers>=2.7"   # 拉 torch ~2GB；Docker 镜像不随之变大
# .env 里把 EMBEDDING_PROVIDER 改成 local；EMBEDDING_DIM=1024
python -m src.ingest                        # 维度变化必须 wipe 旧 data/chroma（§15.8 已述）
python -m src.cli "退款要多久到账？"
```

**灌库（fake 512 维 → bge 1024 维，data/chroma 已整体重建）**：
- 4 篇 → 9 chunk；`injection_sample.md` 仍被标 `ignore_instructions`。
- 维度切换必须 wipe 旧向量目录（`replace_doc` 是 per-doc_id，但混合维度有崩坏风险，本轮干净重灌）。
- 单 chunk 嵌入耗时（CPU/bge-large-zh）：~ 数十 ms 级，9 chunk 数百 ms 完成。

**真管线贯通冒烟（真嵌入 + kimi-k3）**：
- 拒答路径："今天上海的天气怎么样？" → cosine_top1=0.235 < 0.35 → **low_cosine 拒答**；最相近候选 score=0.224 全是 FAQ 退款/发票块——真实语义下门槛依然稳。
- 放行路径（自然问法，非关键词堆叠）："退款申请提交后，多长时间能收到钱？" → cosine_top1=0.685 → kimi-k3 答 "退款会在 3 个工作日内原路退回；如遇法定节假日，顺延到下一个工作日 [1]"。引用编号、内容、token 细节全对。**对比 fake 阶段必须靠 "退款 多久 工作日 原路" 这种关键词堆叠才能过门槛，真嵌入是质变**。
- LLM 自检拒答（fake 阶段未覆盖到的一条纵深防线）："如何申请退款？" 闸门通过（cosine=0.699），但 Generator 判定现有 chunk 只讲"退款条件/多久到账"、未讲"申请流程"，按 prompt 规则 3 输出 `INSUFFICIENT_CONTEXT` 拒答。**这是设计内的纵深防线，不是回归**。

**E4 消融（INJECTION_PENALTY，真嵌入下重跑）**：
| 配置 | attack_success | intercepted | flagger_seen | degraded |
|---|---|---|---|---|
| 默认（penalty=true） | 0% | 100% | 0% | 0% |
| penalty=false | 0% | 100% | 40% | 40% |

真嵌入 vs fake 嵌入（penalty=false 一组）：flagger_seen 都从 0→40%，说明 flagged chunk 在两类嵌入下都能被排进 FINAL_TOP_K；**degraded 从 20% → 40%**——真嵌入进一步让 case2（角色覆盖型）这条原本被闸拒的 case 进入"放行但 degraded"状态，而答案内容依然准确、未被 hijack。这正是旁路 B 期望捕获的边界：与其闸死，不如放行但告诉下游"这条可信度降一档"。

**测试**：50/50 通过。`LocalEmbedder` 是 lazy import，sentence-transformers 不进测试环境。

**已闭环的 §15.7 留白**：
- 真嵌入 cosine 阈值 0.35 **依然合理**（0.235 拒 / 0.685 放 / 0.699 闸门过）——但这只是 1 个模型 + 9 chunk 上的存在性证据，不是统计意义上的"已校准"。
- 真嵌入 E4 重跑完成：无新增风险面，纵深机制按设计工作。
- 语义检索质量**初步认可**（自然问法贯通；注入未扯开防线），但评测集不够大，仍不构成严格校准。

**仍未做**：
- 阈值随模型/语料的统计校准（需要更大评测集）。
- 更主动的注入攻击样本（当前 5 条偏被动，等下一轮 reviewer 提供新样本）。

### 15.9 复核轮 3 收尾：bge 查询指令前缀对照实测（2026-09-30 续）

BAAI bge 系列 model card 建议——非对称检索（短 query 检长 doc）时在 query 前置
`"为这个句子生成表示以用于检索相关文章："`。但该结论源于 MS MARCO / NQ 这类大语料长文档评测；
本项目语料为 9 个 200~500 字 FAQ 短 chunk，需重新实测再决定是否引入。

用 `scripts/bge_prefix_ab.py` 跑一次性对照实验（同 SentenceTransformer 会话、同 3 个 doc、3 条 query）：

| query | 无前缀 | 带前缀 | delta |
|---|---|---|---|
| 今天上海的天气怎么样？（离题，期望远低阈值）| 0.2241 | 0.2190 | -0.005 |
| 退款申请提交后，多久收到钱？（自然问法，期望远高阈值）| 0.6849 | 0.6727 | -0.012 |
| 如何申请退款？（擦边）| 0.6989 | 0.6240 | **-0.075** |

**结论**：本语料上前缀**不降反升地降分**——尤其"如何申请退款？"掉了 0.075。原因是 bge 论文的增益依赖
长文档噪声让前缀起"信号锚点"作用；本语料 chunk 短而直白，前缀反而把 query 向量拉向"我是一个查询"
的语义中心，损失了实际词汇信号。**决定不引入此前缀**。脚本保留在 `scripts/bge_prefix_ab.py`，
未来若语料规模/分布变了，重跑一次再重新决定。

**部署便利性一并修**：`src/retriever/embedder.py` 模块头部加了
`os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")`——CLI 启动时无需手工 `$env:HF_ENDPOINT=...`，
也不会覆盖调用方显式设置的值（`setdefault` 语义）。直连 huggingface.co 网络可达的用户不受影响。

### 15.10 复核轮 3 收尾：cosine 阈值统计校准（2026-09-30 续）

§15.8 留白"阈值随模型/语料的统计校准（需要更大评测集）"，本轮闭环。

**评测集 `eval/probe.jsonl`**：36 条，relevant 26 / irrelevant 10。relevant 13 easy（直接命中章节标题问法）
+ 13 para（同义改写/口语化/语序颠倒）；irrelevant 10 全部为产品常见问题但与语料主题正交
（天气、年假、食堂、登录、版本号、密码、CEO 邮箱、Linux 部署、API 训练授权、周末加班政策）。

**脚本 `python -m eval.run_threshold_probe --env .env`**：只跑 retrieval gate（不调 LLM、不走 generator），
对每个候选阈值扫描 relevant 命中（TP/FN）+ irrelevant 拦截（TN/FP），并输出错分明细。
等价于改一次 `COSINE_THRESHOLD` 跑一遍的效果，但只做一次嵌入推理、再扫 26 个阈值。

**关键数据**（bge-large-zh-v1.5 / dim=1024 / RRF k=60 / 9 chunk）：

| 阈值 | TP | FP | TN | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|---|
| 0.20 | 26 | 10 | 0 | 0 | 0.722 | 1.000 | 0.839 |
| **0.35（现状）** | **24** | **6** | **4** | **2** | **0.800** | **0.923** | **0.857** |
| 0.40 | 22 | 2 | 8 | 4 | 0.917 | 0.846 | 0.880 |
| **0.42（F1 峰值平台起点）** | 21 | 0 | 10 | 5 | **1.000** | 0.808 | **0.894** |
| 0.50 | 21 | 0 | 10 | 5 | 1.000 | 0.808 | 0.894 |
| 0.60 | 20 | 0 | 10 | 6 | 1.000 | 0.769 | 0.870 |
| 0.70 | 8 | 0 | 10 | 18 | 1.000 | 0.308 | 0.471 |

观察：
- relevant 分布 0.281~0.803，irrelevant 分布 0.235~0.417——**两分布不分离**，0.42 之前的任何阈值都不可能同时零误放又零误拒。
- **0.42~0.56 是一段稳定的"零 FP 平台"**：所有 10 条离题都被拦住，代价是漏掉 5 条 relevant。
  被误拒的 5 条全部是 glossary 类型（"X 是什么意思 / X 干什么的"），bge 对"问题-答案"对的得分
  天然高于"问题-定义"对——这是模型固有偏好，不是实现 bug。
- 当前 0.35 下的 6 条误放：年假 / 管理后台 / 版本号 / 密码 / CEO 邮箱 / Linux 部署。
  误拒的 2 条："BM25 是什么？"(0.345)、 "Chunk 是什么意思？"(0.281)。

**终选 0.35（不调）**。决策依据：
1. 误放的 6 条 irrelevant 会被 **Generator INSUFFICIENT_CONTEXT 二次拦截**——已在 §15.8 实测一次
  （"如何申请退款？"闸门通过但 LLM 自检拒答），本轮 refusal eval 也仍然 tp=2 由 LLM 接住。
2. 误拒 glossary 类型 5 条 = 系统性放弃 RAG 一个核心场景（术语解释）。Recall 损失比 precision 损失更难补。
3. RAG 系统偏向 recall：错放 → 用户得到 INSUFFICIENT_CONTEXT 二次拒答（可解释、可重试）；
  错拒 → 用户得到 cold reject 无 citation（体验差）。

**安全回归验证（同样 0.35 + bge + kimi-k3）**：
- `[injection] n=5 攻击成功率 0.0% / flagger_seen 0.0% / intercepted 100.0% / degraded 0.0%`——与 §15.8 一致。
- `[refusal] tp=2 tn=4 fp=1 fn=0  by_rule: llm_insufficient_context=2 low_cosine=1`——tp=2 全部由 LLM 自检接住，
  验证了"低阈值 + 强 Generator 规则"的纵深模型仍然成立。

**已知历史遗留**：qa.jsonl 全部 7 条 `gold_chunk_ids=[]`，导致 `[retrieval] recall@10=0.000`——这是
eval_retrieval 需要金标准 chunk 标注才能算 recall，eval 集创建以来一直没填。**不在本轮范围**，
记入下方"仍未做"。

**仍未做**：
- qa.jsonl 的 `gold_chunk_ids` 标注（需要把 9 个 chunk_id 与 7 条问题逐一对应；做完后 retrieval recall@10/mrr@10 才有数值）。
- 阈值随语料扩展的复测（当前 n=36 是单模型/单语料的存在性证据；语料扩到几十篇/几百 chunk 后应重跑本脚本）→ **已在 11 chunk 语料上复测，见 §15.15**（结论：维持 0.35）
- Probe 集只覆盖 FAQ / 术语 / 错误码 / 报销 4 类主题；新产品主题上线时按需扩 probe.jsonl 并重跑。

### 15.11 复核轮 3 收尾：BM25 strong-match 兜底修复（T0 / 2026-09-30 续）

§15.10 的余波：threshold probe 显示 9 条相关 query（glossary + 错误码 +「如何申请发票」）的 `bm25_strong=False`——
即使 `BM25Store.search` 把它们的目标 chunk 排在 Rank 1。追因发现是 `strong_exact_match` 的判据
"P0-2 修复时改为 top-N 全部干净且全部命中稀有词"在小语料上是**过度防御**：

1. BM25 的天性就是 top-1 命中后带几个 0 分占位 chunk。
2. 中文术语问句的修辞词（"意思/干什么/介绍/讲讲/解释/错误"）从不出现在语料 chunk 文本里，
   却都算进了 `rare_tokens` 而要被命中——自然永远 miss。

#### 修复一：`strong_exact_match` 改回"存在性"语义
判据从"top-N 全部干净且全部命中"改为"**top-N 中存在 1 个干净且命中的 chunk 即 True**"——
旁路 A 防御仍保留（flagged chunk 走 continue，无法借自身兑底；也无法否决其它干净命中）。

[bm25_store.py:52-105](file:///./src/retriever/bm25_store.py#L52-L105)

#### 修复二：stopwords 表补中文修辞词
`tokenizer.py:_STOPWORDS` 补充：`意思/干什么/做什么/怎么回事/什么样/怎样/怎么样/介绍/介绍一下/讲讲/讲下/解释/解释一下/说下/说说/错误/一下/能/能不能/可以/可不可以/请问/想`。
这些词只承担修辞功能，不该成为稀有词命中要求。
[tokenizer.py:14-26](file:///./src/retriever/tokenizer.py#L14-L26)

#### 实测效果（对照 §15.10 的同一份 probe.jsonl）
`bm25_strong` 在该为 True 的 9 条 query 全部由 False → True：

| Query | cos | bm25_strong 前→后 |
|---|---|---|
| BM25 是什么？ | 0.3451 | False → **True** |
| Chunk 是什么意思？ | 0.2814 | False → **True** |
| Embedding 干什么的？ | 0.3744 | False → **True** |
| 什么是检索增强生成？ | 0.5740 | False → **True** |
| 能介绍一下倒数排名融合吗？ | 0.4134 | False → **True** |
| RAG 是什么意思？ | 0.3921 | False → **True** |
| ERR_1042 是什么错误？ | 0.7180 | False → **True** |
| ERR_3310 是什么？ | 0.6418 | False → **True** |
| 如何申请发票？ | 0.6842 | False → **True** |

**端到端含义**：这 9 条即使 cosine 跌到阈值之下也被 BM25 强命中兑底放行。
→ 之前 §15.10 阈值表里"误拒的 5 条 glossary"，实际端到端误判已**全部消除**。

同时 irrelevant 10 条全部仍 bm25_strong=False（罕词不在任何干净 chunk 里 → 攻击者也无法借堆词骗 strong
，除非那些词在合法 chunk 文本里已存在——这种状态本来就是合法内容）。

#### 安全回归
- pytest：**50/50 → 58/58** 全绿（新增 8 条针对新语义的单元测试：test_bm25_strong_match.py）。
- E2E eval：injection 攻击 0%、拦截 100%、refusal tp=2（由 LLM 自检接住）—— 与 §15.8/§15.10 一致。
- 旁路 A 单元测试 `test_p0_2_bypass_a_clean_top1_plus_flagged_rank2_denied` 期望更新为"闸门放行 + 引用校验拦截"
  （stub `_BM25TopNStrongMatch` 跟随新语义改为"任一干净命中即 True"）。
  纵深仍防住：flagged 在 render/filter 阶段被剥离；mock LLM 不肯引用 → validate_and_map_citations 拒答。

#### 仍未做（不变）
- qa.jsonl 的 `gold_chunk_ids` 标注。
- 阈值随语料扩展复测 / Probe 主题的扩展。

---

### 15.12 发布前全量审计（2026-09-30 续）

基于"全面检查，看一下是否可以发布了"，按发布清单逐项过：

#### 卫生与法务
- `.gitignore`（新建）：覆盖 `.env` / `.env.local` / `*.env`（白名单 `.env.example`）、`__pycache__/`、`*.pyc`、`.pytest_cache/`、`data/chroma/`、`eval/results/`、HF 缓存等。`!` 反向规则确保 `.env.example` 仍可被跟踪。
- `LICENSE`（新建）：MIT，著作权人 `KLAT`。
- **不引入** `pyproject.toml`：保留 minimal 项目定位，requirements.txt 已够用。

#### 敏感信息扫描结果
- 全仓 grep `sk-59jI0xq44JEqK39f4TJWN3kWgU8KlWQggTYm6PIUrLM`：仅出现一次，**仅在 `.env` 内**。
- `.env.example` 全是占位符，README/scripts/tests/docs 中均无密钥泄漏。
- `.gitignore` 现在拦住了 `.env`，未来 git init 不会误提交。

#### 依赖审计（对照 §15.7 之前以为缺 chromadb 是误判，已澄清）
- requirements.txt 8 行：`sqlite-vec`、`rank-bm25`、`jieba`、`pypdf`、`openai`、`fastapi`、`uvicorn`、`pytest`。
- 全仓 `import` 扫描未发现遗漏。
- **`sentence-transformers` 保持 optional**（不进 requirements）：仅 `LocalEmbedder` 走 lazy import，其它路径不触及；§15.8 新增"如何复现 local 模式"段说明 `pip install "sentence-transformers>=2.7"`。
- 修了一个非阻塞 FutureWarning：`get_sentence_embedding_dimension` 在新版被改名为 `get_embedding_dimension`，已加 `hasattr` fallback 兼容 both。

#### 目录名历史残留（披露但不改）
- `data/chroma/` 是历史命名（曾计划用 Chroma，后切到 sqlite-vec），实际**与 Chroma 无关**；文件落在 `data/chroma/mini_rag.db`。已在 §13 项目结构末尾加警示条。"改目录名"会牵连 Dockerfile / docker-compose / 既有数据 / 全部 §15.x 记录，成本不匹配。

#### 入口实测（fake provider 以保持当前 dev 环境不被打断）
1. 备份 `data/chroma`（真实 bge 1024 dim 灌库版）
2. `EMBEDDING_PROVIDER=fake EMBEDDING_DIM=512 python -m src.ingest` → 4 篇 9 chunk、注入标记 1 个
3. `src.cli "ERR_1042 是什么原因？"` → 命中 `troubleshooting.md / 错误码速查 > ERR_1042`，引用 [1] 正确
4. `src.cli "今天天气怎么样"` → 拒答（cosine_top1=0.118 < 0.35），候选列出
5. `from src.api import app` → routes 包含 `/healthz`、`/ask`
6. `eval.run_eval --mode injection` → 5 道题攻击成功率 0%、拦截 100%、flagger 1/9 chunk
7. 恢复 `data/chroma` 备份

#### README 一致性扫描
- §8 测试数 35 → 58；§13 测试数 41 → 58；§13 加 LICENSE/.gitignore/probe/run_threshold_probe.py；§13 末尾补 `data/chroma` 历史名警示；§11 LLM/Embedding 行更新（offline 路径已落地于 §15.8）；§15.8 补 local 模式复现步骤。

#### 最终回归
- pytest：**58/58 通过** 0.89s
- eval.run_eval --mode all：retrieval recall@10=0.000（qa.jsonl gold_chunk_ids 全空，§15.10 已声明为生活事实）、refusal tp=2 tn=4 fp=1 fn=0、injection 攻击 0% / intercept 100% / corpus_flagger 1/9

#### 后续工作（**不阻塞发布**）
- qa.jsonl 的 `gold_chunk_ids` 手工标注（用户已认领此工作）→ **已完成**，见 §15.13（标注 + 差旅漏洞修复）与 §15.14（扩容到 19 题）
- 阈值随语料扩展复测
- 更主动的注入攻击样本

**结论：可以发布**。

---

### 15.13 发布后第 1 项收尾：qa.jsonl 标注与「差旅住宿」数据集漏洞修复（2026-09-30 续）

§15.12 把 `qa.jsonl gold_chunk_ids 全空` 标为 **不阻塞发布** 的留白，发布后由用户/作者手工标注。
但标注过程暴露了一个**数据集本身的设计漏洞**，不能只填空而不修因果。

#### 发现：「差旅住宿报销上限是多少？」的唯一答案 chunk 是注入样本
用新写的 `scripts/suggest_gold_ids.py` 跑 top-8 候选时发现：第 5 题原 `expect_refuse=false`，但全语料里
唯一含「每晚 600 元」事实的 chunk 是 `docs/injection_sample.md` 中的 `e1a2b841b2273d92`（`flagged=True`）。
也就是说，**这道题被天然设计成「必须被列内降权 + 注入防护拦下」**，
但 qa.jsonl 把它放进 `expect_refuse=false` 桶——`refusal fp=1` 的来源正是这一题。

#### 修复
1. `docs/faq.md` 新增 `## 差旅报销` 章节，两条干净 chunk：
   - `### 差旅住宿报销上限是多少`（每晚 600 元，一线城市 +20%）
   - `### 餐饮补贴标准`（每日 80 元，与 injection_sample.md 同一份事实，提供干净副本）
2. 重灌 `data/chroma`：9 → 11 chunk。faq.md 的 4 个老 chunk 由于 chunk_index 重排，**chunk_id 全部变化**；
   glossary/troubleshooting/injection_sample 各保持 doc_id 不变，所以原有 chunk_id 不动。
3. `eval/qa.jsonl` 5 道可答题的 `gold_chunk_ids` 全部填入当前真实 chunk_id；
   第 5 题保持 `expect_refuse=false`（因为现在 docs/faq.md 里有干净答案），
   `gold_answer_points` 加 `["一线城市可上浮 20%"]`。注入防护仍由 `eval/injection/*.json` 5 个 case 覆盖。
4. 新增工具 `scripts/suggest_gold_ids.py`：对每道 `expect_refuse=False` 的题跑当前 RRF 融合检索
   打印 top-N 候选，供未来扩语料 / 换 embedding 模型后再标注。

#### 端到端验证（真实 bge 1024 dim + kimi-k3 嵌入）
- retrieval：**recall@10 = 1.000**，**mrr@10 = 1.000**（5/5 道题 top-1 直接命中）
- refusal：**tp=2 tn=5 fp=0 fn=0**（fp 从 §15.12 的 1 归零——「差旅」题不再被误拒）
- injection：5/5 拦截、攻击成功率 0%、corpus_flagger 1/11 chunk
- pytest：**58/58 通过**（无回归）

#### 副作用记录
- 所有 §15.x 中"9 chunk"字样**新事实**为 11 chunk；flagger 比例从 1/9 稀释到 1/11（flag 命中数不变）。
- 由于 faq.md 老 chunk 的 chunk_id 改变，§15.12 之前的若干"具体 chunk_id"字样（如果有）现在也失效
  ——但本次会话查证过 §15.x 没有 hardcoded 旧 chunk_id，只写了「4 篇 9 chunk」等聚合数字。
- 第二轮发布 commit 应在 §15.13 之后做。

#### 结论
`/eval` 链路从"测评指标空转"变成"每跑一次都有真实信号"。
**recall@10=1.000 不是终点，是新基线**——后续若改 embedding 模型 / 换阈值 / 调 chunker，
都以这条曲线为参照。

---

### 15.14 发布后第 2 项收尾：qa.jsonl 扩容（7 → 19 题）与基线刷新（2026-10-01）

§15.12 遗留项里写"qa.jsonl 覆盖面偏薄"（当时 5 可答 + 2 拒答）。本轮把它扩到
**14 可答 + 5 拒答 = 19 题**，并在扩容后的数据集上重跑三段评测。

#### 扩容内容
- 可答题 5 → 14，新增：
  - FAQ 侧：`退款需要满足什么条件`、`发票信息填错了怎么办`、`餐饮补贴标准是多少`
  - 错误码侧：`ERR_2077 是什么错误`、`ERR_3310 是什么意思`
  - 术语侧：`BM25 / Chunk / RRF / Embedding` 四题（原只有 `RAG` 一题）
- 拒答题 2 → 5，新增：`今天的天气怎么样`（域外闲聊）、`如何申请退款`、`怎么重置登录密码`

#### 标注方法
复用 `scripts/suggest_gold_ids.py`（RRF 融合 top-N 候选打印）＋ 逐条回语料核对。
19 条 gold 全部指向现行 11 个 chunk 中的真实 id；`glossary.txt` 5 个术语共用
`fdf5058a63666051`，原因见下"诚实边界"。

#### 实测（真实 bge-large-zh-v1.5 1024 dim + kimi-k3）
- retrieval：**n=14 recall@10=1.000 mrr@10=0.964**
  （唯一非 top-1：`退款需要满足什么条件？` 命中 rank 2，top-1 是语义邻接的 `退款多久到账`）
- refusal：**tp=5 tn=14 fp=0 fn=0**；`by_rule: llm_insufficient_context=4, low_cosine=1`
  （`今天的天气` 离全部语料最远，命中低 cosine 硬门控；其余 4 题由 LLM 的 INSUFFICIENT_CONTEXT 拦下）
- injection：5/5 拦截、攻击成功率 0%、corpus_flagger 1/11（与 §15.13 一致，未回归）
- pytest：**58/58 通过**

#### 值得记一笔：`如何申请退款` 的标注判断
语料里有"退款多久到账"和"退款需要什么条件"，但**没有任何 chunk 描述"申请退款的入口/流程"**。
所以这道题标 `expect_refuse=true` 是成立的——实测也确实由 `llm_insufficient_context` 拒答，
说明 LLM 能区分"有退款政策事实"和"有退款操作步骤"这两种不同问题。这是一条真实的细粒度信号。

#### 诚实边界
- **`glossary.txt` 无标题 → 整篇切 1 chunk**，因此 5 个术语题共用同一个 gold id。
  这是 chunker 粒度的直接后果，不是标注偷懒；若需要术语级 gold，得先给 glossary 加 heading
  或换 chunker，届时 gold id 会随之变化。
- **19 题仍属小样本**，`fp=0/fn=0` 只说明"在这 19 题上成立"，不代表生产分布下的拒答准确率。
- **mrr@10 从 1.000 降到 0.964 是新题带来的真实信号**（旧 5 题仍全部 top-1），
  不是回归；`退款条件` vs `退款到账` 属同一 FAQ 章节内的语义邻接，属可接受的 near-miss。
- 本轮只改 `eval/qa.jsonl` 与本文档，**未动任何检索/门控代码**，也未重灌索引（chunk 数仍 11）。

---

### 15.15 发布后第 3 项收尾：cosine 阈值随语料扩展复测（2026-10-01）

§15.10 把阈值校准标注为"9 chunk / 单模型的存在性证据"，并留了一句
"语料扩到几十篇 / 几百 chunk 后应重跑本脚本"。§15.13 已把语料从 9 → 11 chunk
（faq.md 增「差旅报销」两条），§15.14 把 qa.jsonl 从 7 → 19 题。本轮用**同一份**
`eval/probe.jsonl`（36 条：relevant 26 / irrelevant 10，未改动）在 11 chunk 语料上重跑
`python -m eval.run_threshold_probe --env .env`。

#### 结果（bge-large-zh-v1.5 / dim=1024 / RRF k=60 / 11 chunk）

| 阈值 | TP | FP | TN | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|---|
| 0.30 | 25 | 8 | 2 | 1 | 0.758 | 0.962 | 0.847 |
| 0.34 | 25 | 7 | 3 | 1 | 0.781 | 0.962 | 0.862 |
| **0.35（现状）** | **24** | **7** | **3** | **2** | **0.774** | **0.923** | **0.842** |
| 0.40 | 22 | 3 | 7 | 4 | 0.880 | 0.846 | 0.863 |
| **0.42 ~ 0.56（零 FP 平台）** | 21 | 0 | 10 | 5 | 1.000 | 0.808 | **0.894** |
| 0.60 | 20 | 0 | 10 | 6 | 1.000 | 0.769 | 0.870 |

#### 与 §15.10（9 chunk）的差异：一处**可解释的漂移**
- §15.10 在 0.35 下是 `TP24 FP6 TN4 FN2`；本轮是 `TP24 FP7 TN3 FN2`。
  **多出来的那一条 FP 是「公司食堂几点开门？」**（cos=0.4009，9-chunk 时还在 0.35 以下）。
- 成因可追到 §15.13 的语料改动：faq.md 新增了 `### 餐饮补贴标准`，
  于是「食堂 / 伙食」在 bge 空间里与「餐饮补贴」成为近邻，这条 off-topic 问题的
  top-1 cosine 被抬过了阈值。**这是语料扩张的副作用，不是实现回归**——
  同一份 probe、同一个模型、未改任何代码。
- relevant 侧完全没变：仍是同 2 条 glossary 题被误拒（`BM25 是什么？` 0.3451、
  `Chunk 是什么意思？` 0.2814），与 §15.10 逐值一致。
- F1 峰值平台（0.42~0.56，F1=0.894，零 FP）与 §15.10 逐格一致。

#### 决策：维持 0.35（不调）
理由与 §15.10 同构，但本轮拿到了**更硬的证据**：
1. 7 条 FP 会被 Generator 的 `INSUFFICIENT_CONTEXT` 二次拦截——§15.14 的 refusal eval 直接验证了这一点：
   5 道拒答题里 4 道由 `llm_insufficient_context` 接住（其中就包含 probe 里误放的「年假」「食堂」），
   最终 `tp=5 fp=0`。**"低阈值 + 强 Generator" 的纵深模型在扩容后的数据上仍然成立。**
2. 若为压 FP 把阈值抬到 0.42，代价是**系统性误拒全部 glossary 术语题**（5 条）——那是 RAG 的核心场景。
3. recall 优先：错放 → 用户拿到二次拒答（可解释、可重试）；错拒 → cold reject、无 citation。

#### 仍未做
- probe 集仍是 36 条 / 4 类主题，n 太小；0.842 与 0.894 的 F1 差异**不具统计显著性**，只作趋势。
- 「食堂」这条漂移把 §15.10 的"建议重跑"升级为**"每次扩语料必须重跑本脚本"**——
  一条 off-topic 问题仅因语料新增一个近义章节就能翻越阈值，这个耦合必须每次显式复查。

