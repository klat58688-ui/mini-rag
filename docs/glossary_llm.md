# 进阶术语

## Token

Token 是 LLM 处理文本的最小单位，中文大约一个汉字对应一到两个 Token。

## Temperature

Temperature 控制生成的随机性，越高越发散，常用的严肃场景一般取 0 到 0.3。

## Prompt

Prompt 是喂给模型的指令文本，系统 Prompt 与用户输入都会计入上下文。

## Hallucination

Hallucination 指模型一本正经地编造事实，RAG 的核心动机之一就是压制它。

## 微调 Fine-tuning

微调指在预训练模型基础上用领域数据继续训练，与 RAG 是互补而非互斥的两条路线。
