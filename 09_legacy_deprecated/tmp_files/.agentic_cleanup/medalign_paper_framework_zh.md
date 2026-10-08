# MedAlign-RAG 设计思路（Agent 统一控制器版）

> 版本 v0.4（2026-07-12）。本文是 agentic 部分的权威设计来源。主链路不包含独立
> reranker/PPR；证据效用判断、检索充分性判断和查询改写统一由一个可训练多模态 Agent 完成。

## 1. 论文定位

MedAlign-RAG 研究医学多模态 RAG 中的一个具体问题：

> 当原始查询信息量低、图像依赖强，并且相似度初检返回“视觉相似但临床不一致”的候选时，
> 如何训练一个 Agent 判断哪些证据真正有助于正确作答，并在证据不足时主动改写查询恢复检索。

核心贡献不是新的 Retriever、独立 Reranker 或 Generator，而是
answer-utility-aware evidence controller。它联合执行：

1. 对初检候选逐条 keep/drop；
2. 判断当前证据是否充分（ACCEPT）；
3. 证据不足时生成 rewrite_query 并重新检索（REWRITE）。

这里不能表述为“把 reranker 藏进模型”。传统 reranker 学习 query-document 相关性并生成全序；
本方法的 Agent 直接优化最终答案效用，输出证据子集和下一步动作，不要求完整排序。

## 2. 系统边界

| 模块 | 作用 | 是否训练 | 说明 |
|---|---|---|---|
| 文本/图像双路初检 | 构造候选证据池 | 否 | 固定前端 |
| 密度感知路由 | 分配文本/图像召回预算 | 否 | 支撑模块，runtime 尚未接线 |
| Agent policy | keep/drop + ACCEPT/REWRITE | 是 | 核心贡献 |
| Generator | 基于 kept evidence 回答并提供 logprob | 否 | 冻结评估器 |

独立 PPR/Qwen reranker 不进入主链路。PPR 负结果和外部 reranker 只保留为实验基线，用于证明
answer-utility 控制器与相关性排序的差异。

## 3. 方法总览

~~~text
原始低信息 query + query image
      |
      v
冻结双路初检（text/image candidate pool）
      |
      v
Agent policy（读取问题、选项、query image、候选文本和候选图像像素）
      |
      +-- keep/drop：选择 answer-useful evidence
      |
      +-- ACCEPT：kept evidence 足以作答
      |      |
      |      v
      |   冻结 Generator
      |
      +-- REWRITE：生成判别性 query
             |
             v
          抑制已见证据并重新检索，最多 T 轮
~~~

初检 score 作为 Agent 的观察字段，但不直接作为 reward。Agent 需要能否决高相似度但医学上
误导的证据，因此必须看到 query image 和候选图像证据的真实像素。

## 4. Agent 动作空间

Agent 是单个 VLM policy，一次自回归输出结构化 JSON。所有证据索引统一为 0-based：

~~~json
{
  "keep": [0, 2, 3],
  "drop": [1, 4],
  "action": "ACCEPT",
  "rewrite_query": "",
  "reason": "kept evidence 已足够支持作答"
}
~~~

或：

~~~json
{
  "keep": [0],
  "drop": [1, 2, 3, 4],
  "action": "REWRITE",
  "rewrite_query": "结合腔道形态与黏膜特征判断具体消化道亚部位",
  "reason": "当前候选缺少能区分选项的视觉和解剖证据"
}
~~~

约束：

- keep 与 drop 必须覆盖候选池且不重叠；
- rewrite_query 不得包含答案字母或直接泄露 gold answer；
- Agent 不生成最终答案；
- REWRITE 后在检索层抑制 dropped 和已见证据，避免多轮重复；
- 到达最大轮数后必须用当前 kept evidence 结束。

## 5. Answer-utility Reward

唯一主奖励来自冻结 Generator 对正确选项的概率提升：

~~~text
r(a) = P_G(a* | q, I_q, E_a) - P_G(a* | q, I_q, empty)
~~~

其中：

- a* 是 gold option token；
- E_a 是动作产生的 kept evidence；
- ACCEPT 使用当前 kept evidence；
- REWRITE 使用重新检索并再次选择后的 kept evidence；
- 概率从 vLLM chat completion 的 token logprobs 获取。

轻量约束项包括 invalid JSON、答案泄露、不必要改写和过长查询。gold document/page 是否命中
只作为离线分析指标，不进入主奖励。

该奖励使 policy 学习“证据是否帮助回答”，而不是“证据是否与 query 相似”，是方法与普通
query rewriting 或 learned reranking 的主要区别。

## 6. 训练框架

### 6.1 GPT Go/No-Go 预验证

先让 GPT 作为 Agent 在冻结链路上执行真实 keep/drop 和 rewrite。若 GPT 的平均
answer-utility 无法稳定优于 no-op 初检，则优先修复 QA 数据、检索候选池或 Generator 概率接口，
不启动小模型训练。

### 6.2 轻量 Cold-start SFT

SFT 只负责结构化输出和合理初始行为：

- keep/drop 标签来自冻结 Generator 的逐证据边际效用探针；
- ACCEPT/REWRITE 和 rewrite_query 来自经过 answer-utility 排序的 GPT rollout；
- 只保留 reward 为正、无泄露且格式合法的轨迹；
- 保留足够 ACCEPT 样本，避免 policy 退化为总是 REWRITE。

### 6.3 Online GRPO

同一 QA 构造一组动作：

~~~text
ACCEPT 全候选
ACCEPT keep 子集
REWRITE 候选 1..N
~~~

每个动作都真实运行检索、Agent 选择和 Generator 打分，组内使用：

~~~text
A_i = (r_i - mean(r)) / (std(r) + epsilon)
~~~

GPT rollout bank 只作为 cold-start/replay；主训练增益必须来自小模型在线动作和
answer-utility GRPO。实验中必须报告 SFT-only 与 SFT+GRPO。

## 7. 数据构造

Stage 1 只构造可靠医学多模态 QA，不构造虚假的 rewrite 监督：

~~~text
image-text pair + query image + 邻近文本 + PDF 相关 1-2 页
  -> API 生成候选 question/options/candidate answer（必须传图）
  -> 独立盲答 Verifier（必须传图，不看到 candidate answer）
  -> 来源约束、答案一致性和质量门
  -> qa_gold.jsonl
~~~

训练数据不得使用最终 benchmark 样本。benchmark 只用于题型分布参考、开发期冻结评估和最终测试。
Stage 1 v1 只使用 image_text_pair；text_only 留作后续对照。

详细 schema、题型和过滤逻辑见 data_construction/data_construction_design_zh.md。

## 8. 模型与部署

| 组件 | 当前选择 | 状态 |
|---|---|---|
| Agent policy | Qwen3.5-4B 多模态模型 | 待训练 |
| Generator | Qwen3-VL-8B-Instruct + vLLM logprobs | runtime 已封装 |
| 文本初检 | BGE-M3 | 已有本地索引 |
| 图像初检 | Qwen3-VL image encoder | 已有本地索引 |
| 密度路由 | Mengzi-BERT 双头 | 配置保留，runtime 待接线 |

建议四张 A6000 的初始分配：

- GPU 0-1：冻结 Generator；
- GPU 2：Agent rollout/训练；
- GPU 3：图像检索编码或训练副本；
- 文本检索按显存与吞吐动态放置。

实际分配应以显存 profiling 为准，不在论文中写成固定要求。

## 9. 实验设计

### 9.1 主基线

| Method | Evidence selection | Rewrite | Trainable |
|---|---|---|---|
| Closed-book Generator | 无 | No | No |
| First-stage retrieval | top-k 原顺序 | No | No |
| Retrieval + PPR | 外部相关性排序 | No | No |
| Retrieval + external Qwen reranker | 外部相关性排序 | No | No |
| Agent keep/drop only | answer-utility 子集 | No | Yes |
| Agent rewrite only | 不筛证据 | Yes | Yes |
| MedAlign-RAG | keep/drop | Yes | Yes |

PPR 和外部 reranker 是对照，不是部署组件。

### 9.2 必做消融

- keep/drop on/off；
- rewrite on/off；
- suppression on/off；
- answer-utility reward vs evidence-hit/relevance reward；
- query image only vs query image + candidate evidence images；
- SFT-only vs SFT+GRPO；
- 最大轮数 T；
- 候选池大小与模态预算；
- 密度路由 on/off（完成接线后）。

### 9.3 指标

- 最终 QA accuracy；
- answer-utility lift；
- rewrite trigger/success rate；
- 平均轮数与平均 kept evidence 数；
- evidence-hit/Recall@K（仅分析）；
- 无效 JSON 与 answer leakage rate；
- 延迟、吞吐和视觉 token 成本；
- 跨 Generator 泛化，检查 reward overfitting。

## 10. 论文叙事边界

建议表述：

> We remove standalone relevance reranking from the deployed pipeline. A trainable multimodal
> policy jointly estimates answer-conditioned evidence utility and retrieval sufficiency, then
> selects evidence and decides whether to rewrite the query.

不声称：

- 发明了多轮 Agentic RAG；
- 提出了新的 Retriever/Reranker/Generator；
- keep/drop 是一个隐藏的传统 reranker；
- PPR 是有效组件。

主张：

- 医学多模态场景中的 answer-utility-aware evidence control；
- 将证据选择、充分性判断和 query rewrite 统一到单一 policy；
- 通过在线 GRPO 优化最终答案效用。

## 11. 当前实现状态

已完成：

- Stage 1 prepare/route/API generate/盲答 verify/validate；
- 双路初检适配；
- GPT Agent keep/drop + ACCEPT/REWRITE；
- 候选图像像素传入 Agent；
- 多轮 suppression；
- Generator logprobs 与 answer-utility；
- GPT rollout smoke 与 GRPO 组内优势。

待完成：

- 密度感知路由接入 runtime；
- keep/drop 软标签批处理；
- 本地 Agent policy 推理适配；
- SFT 与 GRPO 训练器；
- 跨 Generator 和完整基线实验。

