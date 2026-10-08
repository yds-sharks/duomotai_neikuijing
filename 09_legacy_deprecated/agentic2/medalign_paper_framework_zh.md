# MedAlign-RAG 论文框架与训练思路

## 1. 论文定位

本文不把核心贡献定义为新的 retriever 或 reranker，而是定义为：

> 在固定多模态 RAG 检索前端上，训练一个 evidence-selection + query-rewrite agent。系统不部署独立的重排序模型：粗召回后直接按检索分数取 top5 候选（文本+图片）交给 agent。agent 一方面从 top5 中筛选出更有助于回答的证据，另一方面判断证据是否足够，不足时主动改写 query 重新召回，从而提升最终多模态问答效果。

系统中的 retriever、generator 均固定不训练，且不存在独立 reranker 模块。唯一被训练的组件是该 agent。

这样可以避免论文被理解为简单堆叠多个模块。各模块职责划分如下：

| 模块 | 作用 | 是否训练 | 论文角色 |
|---|---|---|---|
| 粗召回 | 从文本库和图像库召回候选证据，按分数取 top5（文本+图片） | 否 | 固定前端 |
| evidence-selection + rewrite agent | 从 top5 筛选更有用的证据，并决定 ACCEPT 或改写 query 重新召回 | 是 | 核心贡献 |
| generator | 基于 agent 选中的证据回答选择题并提供 reward | 否 | 冻结评估器 |

## 2. 核心问题

现有多模态 RAG 通常采用一次性流程：

```text
query -> retrieve -> rerank -> generate
```

一旦初始 query 低密度、图像依赖强，或者召回结果被同部位不同病理的相似证据干扰，系统没有自我恢复机制。单纯的重排序只能在已有候选中调整顺序，无法改变召回分布。

本文关注的问题是：

> 当 reranked top5 evidence 不足以支持回答时，模型能否通过 answer-utility reward 学会改写 query，使下一轮召回更接近真正有用的医学证据？

## 3. 主要痛点

### 3.1 一次性检索不可恢复

医学多模态 QA 中，原始 query 常常很短或依赖图像，例如“图中是什么部位”“该病变最符合哪一项”。这类 query 的文本检索信号弱，初始召回容易失败。

传统 RAG 即使加 reranker，也只能在已有候选内重排，不能主动改变检索方向。

### 3.2 相关性不等于答案效用

reranker 的高分证据不一定能帮助 generator 答对。医学图像中，同一部位不同病理可能视觉相似，相关但误导。

因此训练信号不应只来自 relevance，而应来自最终 answer utility。

### 3.3 小模型不应承担大规模 rerank

让小型 agent 直接读取 top20 文本和 top20 图像并完成细粒度排序，成本高且目标混乱。

因此本文不引入独立重排序模型，也不让 agent 在全部 top20 候选上做大规模 rerank。粗召回后直接按检索分数取 top5（文本+图片），agent 只在这 top5 的小观察窗口内做证据筛选，并专注于检索质量诊断与 query rewrite。

## 4. 方法总览

整体链路为：

```text
原始 query + 可选 query image
        |
        v
固定粗召回 top20（文本库 + 图像库）
        |
        v
按检索分数取 top5（文本 + 图片）
        |
        v
agent：从 top5 筛选有用证据 + 决定 ACCEPT / REWRITE
        |
        +-- ACCEPT -> generator 基于选中证据 -> answer
        |
        +-- REWRITE -> 新 query -> 重新召回 -> 再取 top5 -> 再筛选 -> generator -> answer
```

agent 每一步同时输出两部分：证据筛选结果和动作决策。

```json
{"selected_evidence": ["E1", "E3"], "action": "ACCEPT"}
```

```json
{"selected_evidence": ["E2"], "action": "REWRITE", "rewrite_query": "..."}
```

其中 `selected_evidence` 是 agent 从当前 top5 中挑选出的、更有助于回答的证据子集；`action` 决定是用选中证据直接作答（ACCEPT），还是改写 query 重新召回（REWRITE）。

## 5. 训练数据构造

训练数据不使用 benchmark 样本。Benchmark 仅用于统计问题类型分布和最终测试。

训练样本来自：

- 自有向量库中的 image-text pair；
- 原始 PDF；
- PDF 图片、图注和邻近正文；
- 已有元数据中的器官/部位、病理/类型标签。

每条训练样本必须是四选一选择题：

- `question`
- `options: A/B/C/D`
- `answer`
- `answer_text`
- `query_image_path`
- 简单 source provenance

API 模型只负责生成或润色 `question` 和 `options`，不得从零判断 gold answer。正确答案必须由源数据元信息或确定性构造逻辑决定。

## 6. 训练框架

### 6.1 训练目标

训练目标不是让 agent 成为 reranker，而是让它学习：

1. 当前 top5 evidence 是否足够；
2. 如果不足，应该如何改写 query；
3. 改写后的 query 是否能通过重新召回带来更高 answer utility。

### 6.2 固定前端

训练过程中以下模块全部冻结：

- retriever；
- organ/site filter，如保留；
- modality routing，如保留；
- generator。

系统不含可训练或独立的重排序模块；top5 由粗召回分数直接确定。这样可以将性能变化归因于 agent 的证据筛选与 query rewrite。

### 6.3 GPT Agent 预验证

在训练小模型之前，先使用 GPT 作为 rewrite agent 验证该任务是否有可学习收益。

对于同一条样本，比较两条路径：

```text
原始 query -> recall -> rerank top5 -> generator -> reward_original
```

```text
GPT rewrite query -> recall -> rerank top5 -> generator -> reward_rewrite
```

如果 GPT rewrite 不能稳定优于 no-rewrite，则不应直接训练小模型，应优先检查数据、召回、reranker 和 generator/reward。

### 6.4 GRPO 训练

小模型训练采用 GRPO。每个 query 形成一个 group，包含多个 agent 输出（证据筛选 + 动作组合）：

- 不同证据筛选下的 `ACCEPT`
- 多个不同 `REWRITE` 候选

每个候选动作都实际执行完整链路：

```text
action
 -> 如果 REWRITE，则用新 query 重新召回并按分数取 top5
 -> generator 基于选中证据回答
 -> 计算 reward
```

GRPO 在组内比较不同动作的 reward，推动模型更倾向于产生高 answer-utility 的动作。

### 6.5 Reward 设计

主 reward：

```text
R = U(rewrite_query) - U(original_query)
```

其中 `U` 表示 frozen generator 在给定 top5 evidence 后的 answer utility。

可用的 answer utility 包括：

- 是否答对四选一选择题；
- 正确选项 log probability；
- 相对 empty evidence 的 correct option logprob lift。

训练时可以加入轻量约束项：

- 无效 JSON 扣分；
- 泄露答案字母扣分；
- 明显偏离原问题扣分；
- 过长或无意义 rewrite 扣分。

论文主叙事中应强调 answer-utility reward，避免把工程约束项写成主要贡献。

### 6.6 Agent 输入

agent 输入包括：

- question；
- options；
- query image 是否存在；
- 当前按检索分数取出的 top5 evidence；
- 每条 evidence 的 modality、文本、图片路径或图片摘要、粗召回检索分数。

agent 不接收 top20/top20 原始候选，只在 top5 观察窗口内做筛选与决策。

### 6.7 Agent 输出

agent 输出必须为 JSON，同时包含证据筛选与动作：

```json
{"selected_evidence": ["E1", "E3"], "action": "ACCEPT"}
```

或：

```json
{"selected_evidence": ["E2"], "action": "REWRITE", "rewrite_query": "..."}
```

推理时不给 gold answer。

训练数据构造阶段可以使用 gold answer 辅助生成候选题目，但不能让最终 agent prompt 依赖 gold answer。

## 7. 实验设计

### 7.1 主实验

主表比较：

| Method | Selection | Rewrite | Trainable | Overall |
|---|---|---|---|---|
| Closed-book generator | No | No | No | TBD |
| Retrieve top5（全部证据直接作答） | No | No | No | TBD |
| GPT selection + rewrite | Yes | Yes | No | TBD |
| GRPO agent（selection + rewrite） | Yes | Yes | Yes | TBD |

重点证明：在相同固定前端下，训练后的 agent（证据筛选 + query rewrite）优于 no-rewrite 和非训练基线。

### 7.2 消融实验

推荐消融：

- w/o evidence selection：agent 只改写 query，不筛选证据（top5 全部给 generator）；
- w/o rewrite：agent 只筛选证据，不允许改写 query；
- w/o answer-utility reward：替换为格式或相关性 reward；
- one rewrite round vs two rewrite rounds；
- GPT agent vs trained small agent。

### 7.3 分析指标

除最终 accuracy 外，建议报告：

- rewrite trigger rate；
- rewrite success rate；
- original-query reward vs rewrite-query reward；
- invalid JSON rate；
- answer leakage rate；
- average inference rounds；
- top5 evidence 中有效证据比例变化。

## 8. 贡献写法

建议贡献点写成：

1. 提出一种 answer-utility guided agent，在固定多模态 RAG 前端中同时完成证据筛选与 query rewrite，用于检索失败恢复。
2. 构建不使用 benchmark 训练样本的 source-grounded 医学多模态选择题构造流程。
3. 通过 GRPO 直接优化 agent policy，使其学会在 top5 观察窗口内筛选证据，并在 ACCEPT 与 REWRITE 之间做决策。
4. 在固定 retriever、generator 且无独立 reranker 的设置下验证 agent 的增益，避免将收益归因于前端模块变化。

## 9. 论文叙事边界

本文不主打：

- 新 retriever；
- 新 reranker（本文不含独立 reranker 模块）；
- 新 generator；
- 器官识别模块；
- 图文权重模块。

这些可以作为固定前端或工程组件保留，但不应进入核心贡献。

本文主打：

> 固定多模态 RAG 检索栈上的可学习 evidence-selection + query-rewrite controller。

这一定位能降低“堆模块”的审稿风险，也使消融更加清晰。
