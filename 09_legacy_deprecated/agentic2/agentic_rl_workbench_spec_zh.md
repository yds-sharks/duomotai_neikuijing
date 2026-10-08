# Agentic-RL 工作台执行规格

## 1. 数据规模

- 目标规模：4,000 条选择题样本。
- 训练数据来源：仅使用自有向量库、原始 PDF、PDF 图片、图注、邻近正文。
- Benchmark 数据不得进入训练集，只允许用于统计问题类型分布和最终测试。
- 数据划分：
  - train：3,200
  - dev：400
  - internal_test：400
- 题型比例不在本规格中固定。执行层需要先统计 benchmark 和其它常见问题集的题型分布，再制定匹配或有意调整后的数据构造比例。

## 2. 样本字段

每条样本必须是四选一选择题，答案必须是 `A/B/C/D` 之一。

```json
{
  "qid": "unique_id",
  "split": "train/dev/internal_test",
  "query_type": "执行层定义的问题类型",
  "question": "问题文本",
  "options": {
    "A": "...",
    "B": "...",
    "C": "...",
    "D": "..."
  },
  "answer": "A/B/C/D",
  "answer_text": "...",
  "query_image_path": "可选图片路径",
  "source": {
    "source_type": "vector_db/pdf",
    "doc_id": "...",
    "doc_name": "...",
    "page_idx": 0,
    "image_id": "...",
    "image_path": "...",
    "evidence_text": "...",
    "level1": "器官/部位",
    "level2": "病理/类型"
  },
  "provenance": {
    "generated_by": "gpt/qwen/manual_template",
    "benchmark_used_for_training": false
  }
}
```

## 3. 字段责任划分

由程序生成或写入的字段：

- `qid`
- `split`
- `query_type`
- `query_image_path`
- `source`
- `provenance`
- `answer_text`
- `benchmark_used_for_training`

由 API 模型生成或润色的字段：

- `question`
- `options`
- 可选：`rewrite_seed`，如果执行层希望保存初始改写候选

由源数据元信息或确定性构造逻辑决定的字段：

- `answer`
- `answer_text`
- 全部 `source` 字段
- 正例来源证据
- 负选项候选

API 模型不得从零判断正确答案。它只能在正确选项已由源数据或构造逻辑确定后，负责选择题文本和选项表述。

## 4. 最小数据验收标准

- 每条样本必须是合法四选一选择题。
- `answer` 必须是 `A/B/C/D` 之一。
- `answer_text` 必须等于 `options[answer]`。
- `benchmark_used_for_training` 必须为 `false`。
- `question` 不能包含答案字母泄露，例如“选A”“答案是B”“正确答案”。
- 每条样本必须保留来自向量库或 PDF 的简单来源记录。

## 5. 固定前端链路要求

执行层必须封装统一调用接口：

```text
input: query_text + optional query_image
output: top5_evidence（按粗召回分数取出）
```

固定前端链路：

```text
query
 -> 粗召回 top20（文本库 + 图像库）
 -> 按检索分数取 top5（文本 + 图片）
 -> agent（证据筛选 + ACCEPT/REWRITE 决策）
```

要求：

- 粗召回只负责召回。
- 不部署独立 reranker 模型；top5 直接由粗召回分数排序取出（文本+图片混合）。
- agent 接收 top5 evidence，负责从中筛选更有用的证据，并决定 ACCEPT 或 REWRITE。
- agent 不负责对 top20/top20 候选做大规模 rerank，只在 top5 观察窗口内筛选。
- 器官/部位过滤、图文权重路由可以保留为冻结前端组件。

## 6. 证据取用要求

本方案不部署独立 reranker 模型。粗召回后，直接按 first-stage 检索分数对候选排序，取 top5（文本+图片混合）作为 agent 的观察窗口。

要求：

- top5 仅由粗召回分数决定，不引入任何额外排序模型。
- 文本证据与图像证据统一进入同一个 top5 候选池，按分数混排。
- 原来由固定多模态 reranker 承担的“证据压缩/选择”职责，改由 agent 在 top5 上完成筛选。

## 7. GPT Agent（证据筛选 + Rewrite）测试接口

小模型训练前，先使用 GPT 作为 agent，验证证据筛选 + query rewrite 是否能提升检索与最终 QA 效果。

Agent 输入：

```json
{
  "question": "...",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "query_image_available": true,
  "top5_evidence": [
    {
      "eid": "E1",
      "modality": "text/image",
      "text": "...",
      "image_path": "...",
      "score": 0.0
    }
  ]
}
```

Agent 输出必须同时包含证据筛选与动作决策：

```json
{"selected_evidence": ["E1", "E3"], "action": "ACCEPT"}
```

```json
{"selected_evidence": ["E2"], "action": "REWRITE", "rewrite_query": "..."}
```

初始 prompt 要求：

```text
你是医学多模态 RAG 的证据筛选与 query rewrite 控制器。
给定问题、四个选项和当前按检索分数取出的 top5 证据，先从中筛选出更有助于回答的证据（selected_evidence），再判断这些证据是否足以支持回答。
如果足够，输出 {"selected_evidence":[...],"action":"ACCEPT"}。
如果不足，输出 {"selected_evidence":[...],"action":"REWRITE","rewrite_query":"..."}。
rewrite_query 应提升下一轮检索对关键部位、病理/类型、视觉鉴别特征和选项区分信息的覆盖。
禁止输出答案字母，禁止写“正确答案是...”，禁止解释。
只输出 JSON。
```

该 prompt 仅为初始版本，执行层可以继续调整。

## 8. Generator / Reward 接口

需要封装冻结 generator 接口：

```text
input: question + options + optional query_image + agent 选中的证据
output: predicted_answer + answer_logprobs + correctness
```

要求：

- generator 固定，不训练。
- generator 必须支持图文混合输入。
- GPT rewrite 测试时必须比较：
  - 原始 query 的 top5 reward
  - rewrite query 的 top5 reward
- 只有 rewrite 后最终 answer utility 高于原始 query，才算有效 rewrite。

## 9. 后续训练模型候选

当 GPT rewrite 证明有效后，再训练小型 rewrite agent。

推荐 policy 模型：

- Qwen/Qwen2.5-VL-3B-Instruct：https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
- Qwen/Qwen2.5-VL-7B-Instruct：https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct

训练设定：

- 冻结 retriever、generator（系统无独立 reranker）。
- 只训练 agent。
- agent 输出：证据筛选 `selected_evidence` + 动作 `ACCEPT`/`REWRITE`。
- 优化方式：GRPO。
- reward：agent 选中证据下的最终 answer utility 减去基线（原始 query + 全部 top5）的 answer utility。

## 10. 小模型训练前 Go / No-Go 标准

只有当 GPT rewrite 在同一固定前端上优于 no-rewrite 时，才进入小模型 GRPO 训练。

需要达到的结果：

- 最终 QA accuracy 相比 no-rewrite 有提升。
- rewrite query 的最终 answer utility 高于原始 query。
- 无效 JSON、答案泄露、明显偏题 rewrite 的比例可控。

如果 GPT rewrite 没有带来提升，暂缓小模型训练，优先检查数据构造、召回、reranker 行为以及 generator/reward 可靠性。
