# Agentic RAG 相关工作调研与 MedAlign-RAG 方案修正建议

本文档调研 ReasonRAG、MA-RAG（Multi-Round Agentic RAG）、Doctor-RAG、Search-R1 以及 Agentic RAG SoK，并对当前 MedAlign-RAG 的数据构造和训练叙事做针对性修正。

## 1. 当前方案需要修正的点

当前 v2 数据构造已经从“证据描述题”改成了图像问题，但仍存在两个核心问题：

1. `organ_tags` 可能包含多个器官/部位，不能简单取第一个非“通用”标签作为 gold answer。
2. `image_content_type_identification` 这类“图像内容类型识别”题不贴近真实 benchmark 使用场景，也不给 rewrite agent 提供足够真实的检索失败/改写空间。

因此，后续数据构造不应继续机械使用标签生成固定问题，而应改成：

```text
自有图文对 + query image + PDF 两页上下文
 -> 参考 benchmark taxonomy / 问题风格
 -> API 生成图像依赖的四选一问题
 -> API/规则二次验证答案是否被图像与来源证据支持
 -> accepted 才进入训练池
```

benchmark 只用于学习题型分布和问题风格，不把 benchmark 样本内容放入训练。

## 2. ReasonRAG: 过程监督比单纯 outcome reward 更稳

论文：Process vs. Outcome Reward: Which is Better for Agentic RAG Reinforcement Learning / ReasonRAG

核心思想：

- 传统 outcome reward 只看最终答案是否正确，训练信号稀疏。
- ReasonRAG 主张构造 process-level rewards，覆盖 query generation、evidence extraction、answer generation 三类动作。
- 它用 MCTS 探索 agentic RAG 决策树，用 SPRE（Shortest Path Reward Estimation）给中间步骤估计奖励。
- 最终构造 RAG-ProGuide 数据集，并用 preference optimization 训练模型。

关键数据：

```text
Questions: 4603
Actions: 13289
Query Generation: 3295
Evidence Extraction: 4305
Answer Generation: 5689
Avg. Iteration: 2.7
```

对我们的启发：

- 我们当前只做 `ACCEPT/REWRITE`，比 ReasonRAG 的动作空间更窄，这是差异化优势：更聚焦、更低成本。
- 但我们不能只依赖最终 `U(rewrite)-U(original)`，否则会有 sparse reward 问题。
- 可以引入轻量 process signals：
  - rewrite 是否包含正确部位/病变/视觉特征；
  - rewrite 是否避免答案字母泄露；
  - rewrite 是否真正改变检索方向；
  - top5 evidence 是否包含 source-supported 证据。
- SFT warm-up 不应只是 GPT 轨迹 imitation，而应保留 rejected/accepted rewrite 对，形成偏好样本。

建议写法：

```text
Unlike ReasonRAG, which trains a general agent to decide among query generation, evidence extraction, and answer generation, MedAlign-RAG isolates a narrower multimodal query-rewrite controller on top of a frozen retrieval-reranking-generation stack. This constrained action space reduces policy search complexity and enables answer-utility-guided optimization in a medical multimodal setting.
```

## 3. MA-RAG: 利用语义冲突作为检索触发信号

论文：From Conflict to Consensus: Boosting Medical Reasoning via Multi-Round Agentic RAG

核心思想：

- MA-RAG 是医学 QA 方向的 multi-round agentic RAG。
- 每轮先由 Solver Agent 采样多个 candidate responses。
- 如果候选回答之间存在 semantic conflict，Retrieval Agent 将冲突转成 actionable queries 去检索外部证据。
- Ranking Agent 选择高质量历史推理痕迹，避免长上下文退化。
- 它把 self-consistency 从“投票选答案”扩展成“如果不一致，就触发检索”。

对我们的启发：

- 我们当前 GPT rewrite 判断依据是 `top5 evidence 是否足够`，这很好，但可以补充一个更强信号：generator 多次回答或不同 evidence 下回答是否冲突。
- 如果原始 query 的 generator 多采样结果分散，说明当前证据不足，可以作为 REWRITE 触发信号。
- 我们不应声称“提出 multi-round agentic RAG”，因为 MA-RAG 已经做了医学 multi-round agentic RAG。
- 我们应强调差异：
  - MA-RAG 处理文本医学 QA reasoning；
  - 我们处理医学多模态 RAG 中低信息图像 query 的检索失败恢复；
  - 我们训练的是轻量 rewrite controller，而不是多 agent test-time scaling 系统。

建议借鉴：

```text
原始 query -> top5 evidence -> generator 多次回答
若答案分布冲突/置信不稳，则更倾向触发 REWRITE。
```

可作为实验指标：

```text
answer conflict rate before rewrite
answer conflict rate after rewrite
rewrite success on high-conflict cases
```

## 4. Doctor-RAG: 失败诊断和局部修复

论文：Doctor-RAG: Failure-Aware Repair for Agentic Retrieval-Augmented Generation

核心思想：

- DR-RAG 不盲目重跑整个 agentic RAG，而是先做 failure diagnosis。
- 诊断阶段判断 evidence sufficiency、failure type，并定位最早失败点。
- repair 阶段只在失败点做局部修复，尽可能复用有效前缀和证据。

对我们的启发：

- 我们的 rewrite agent 也应被描述成 failure-aware controller，而不是泛泛 query rewriter。
- 当前 `ACCEPT/REWRITE` 可以解释为对 retrieval failure 的二分类诊断：
  - `ACCEPT`: top5 evidence sufficient；
  - `REWRITE`: evidence insufficient or misleading，需要修复 query。
- 但我们不要声称做了复杂 failure localization；Doctor-RAG 已经覆盖这一方向。
- 我们可以加入轻量 failure taxonomy：
  - missing organ/site;
  - missing lesion/type;
  - visual feature absent;
  - distractor confusion;
  - evidence off-topic.

这有利于数据构造和论文分析。

## 5. Search-R1 / ReSearch / R1-Searcher: outcome RL 搜索代理

这些工作将搜索引擎作为工具，让模型通过 RL 学习何时搜索、如何搜索、多轮搜索。

对我们的关系：

- 它们通常是通用文本搜索或开放域 QA。
- 动作空间更大：搜索、阅读、推理、回答。
- reward 多为最终答案正确性。

我们的差异：

- 检索前端、reranker、generator 全冻结。
- policy 只负责 rewrite/accept，不负责阅读所有 top-k 或生成最终答案。
- 场景是医学多模态图像 QA，不是开放域文本搜索。
- reward 是 frozen generator 的 answer utility improvement，而不是模型自己最终回答是否正确。

可借鉴点：

- 训练时要惩罚不必要的工具调用/重写。
- 需要报告平均调用轮次和 rewrite trigger rate。

## 6. Agentic RAG SoK: 避免叙事过宽

SoK 将 Agentic RAG 形式化为有限视野 POMDP，并总结：

- planning mechanisms;
- retrieval orchestration;
- memory paradigms;
- tool invocation;
- verification and self-correction;
- cost/latency/reliability trade-offs。

对我们的启发：

- 我们的贡献不能写成“通用 agentic RAG 框架”。
- 应写成一个具体、窄而清晰的 controller：

```text
A multimodal answer-utility-guided query rewrite controller for low-information medical image queries under a frozen RAG stack.
```

- 评估不能只看最终 accuracy，还要看 trajectory-level 指标：
  - rewrite trigger rate;
  - accepted vs rejected rewrite;
  - evidence utility change;
  - invalid JSON rate;
  - answer leakage rate;
  - inference cost/rounds;
  - top5 evidence support rate。

## 7. 与我们当前成文逻辑的冲突风险

### 7.1 “Multi-round agentic RAG” 不是我们的主贡献

MA-RAG 已经是医学 multi-round agentic RAG，并且明确使用 conflict-to-query 机制。

我们的主贡献应避免写成：

```text
提出医学多轮 Agentic RAG 框架
```

应改成：

```text
提出固定多模态 RAG 前端上的低信息图像 query rewrite controller。
```

### 7.2 “过程监督 agentic RAG” 不是我们的主贡献

ReasonRAG 已经主打 process-level reward 和 process-supervised data。

我们可以借鉴，但不要声称全面过程监督。我们的差异是：

- 动作空间更窄；
- 医学多模态；
- answer utility 来自 frozen generator；
- 训练目标是 retrieval failure recovery，而非完整 agentic reasoning。

### 7.3 “失败定位修复” 不是我们的主贡献

Doctor-RAG 已经做 failure diagnosis + local repair。我们可以说自己是 evidence sufficiency-aware rewrite，不要包装成完整 failure localization。

## 8. 新的数据构造建议

你提出的方向是正确的：benchmark 已经整理了困难问题类型，我们应学习它的题型风格，而不是直接拿标签生成问题。

建议新链路：

```text
EndoBench taxonomy / question style statistics
        |
        v
自有 image-text pair + query image + PDF 两页上下文
        |
        v
API question generator:
  生成 1-3 个 benchmark-style image-dependent MCQ candidates
  生成 answer_text 与 options
  明确指出答案依据来自图像/PDF局部证据
        |
        v
API verifier:
  判断问题是否图像依赖、答案是否唯一、证据是否支持、选项是否合理
        |
        v
local validator:
  schema / leakage / duplicate / provenance / split check
        |
        v
accepted MCQ pool
```

## 9. 新题型建议

优先保留贴近 benchmark 的题型：

### 9.1 Organ / anatomical site identification

示例：

```text
这张内镜图像主要显示哪个部位？
图中所示区域最可能属于哪一段消化道？
```

答案不再由 `organ_tags[0]` 直接决定，而是：

```text
organ_tags + 图像 + 图文描述 + PDF两页 -> API 确认唯一 gold
```

如果多个部位都可成立，则拒绝。

### 9.2 Lesion / finding identification

示例：

```text
图中病变最符合下列哪种表现？
这张图像最主要显示哪种异常改变？
```

答案应来自图文描述/PDF，而不是固定标签。

### 9.3 Procedure / treatment step recognition

示例：

```text
图中正在进行的内镜操作最可能是？
该图像对应下列哪个治疗步骤？
```

这比“图像内容类型识别”更真实，因为选项是具体操作，而不是“检查操作/治疗操作”这种元标签。

### 9.4 Visual discrimination / option distinction

示例：

```text
与其他选项相比，图中最突出的识别依据是？
根据图像特征，最能支持该判断的是哪一项？
```

这类题可直接训练 rewrite agent 提取视觉鉴别特征。

不建议继续使用：

```text
这张医学图像主要对应下列哪类内容？
```

它太像内部标签分类，不像真实医学 QA。

## 10. 新 API generator 输出建议

每条自有图文对输入：

```json
{
  "query_image_path": "...",
  "image_description": "...",
  "pdf_context_text": "...",
  "organ_tags": [...],
  "primary_knowledge_type": "...",
  "benchmark_style_pool": [...]
}
```

API 输出：

```json
{
  "candidates": [
    {
      "query_type": "organ_identification / lesion_identification / procedure_step_recognition / visual_discrimination",
      "question": "...",
      "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
      "answer": "A",
      "answer_text": "...",
      "evidence_basis": "简述答案依据，不进入最终 question",
      "image_dependency": "high/medium/low",
      "reject_if": "..."
    }
  ]
}
```

生成阶段要求：

- 不能把图文描述直接写进 question。
- 不生成需要 bbox/坐标的题。
- 不生成答案依赖常识而非图像/PDF局部证据的题。
- 不生成多个答案都合理的题。

## 11. 新 API verifier 输出建议

在原 verifier 基础上增加：

```json
{
  "benchmark_style_match": true,
  "image_dependency": "high",
  "gold_source": "image+caption/pdf_context",
  "multi_organ_ambiguity": false,
  "question_naturalness": true,
  "rewrite_value": "high/medium/low"
}
```

其中 `rewrite_value` 判断该题是否有训练 rewrite agent 的价值：

- high: 原始问题低信息，必须靠图像/检索补充部位、病变、视觉特征。
- medium: 有一定图像依赖。
- low: 文本题或标签题，训练价值低。

最终只保留：

```text
image_dependency in {high, medium}
rewrite_value in {high, medium}
answer_correct == true
ambiguous == false
multi_organ_ambiguity == false
```

## 12. 对训练链路的建议

### 12.1 GPT rewrite 预验证

保留原计划，但输入数据应换成 API-generated benchmark-style MCQ，而不是标签模板题。

### 12.2 SFT warm-up

从 GPT rewrite 预验证日志中筛选：

- ACCEPT 正样本：原始 top5 已足够，generator 答对，rewrite 无必要。
- REWRITE 正样本：rewrite 后 answer utility 明显提升。
- 负样本：rewrite 泄露答案、偏题、无提升、增加噪声。

可形成偏好对：

```text
same state: good rewrite > bad rewrite
same state: accept > unnecessary rewrite
same state: rewrite > accept when original evidence insufficient
```

这比单纯 SFT imitation 更接近 ReasonRAG 的 process preference 思路。

### 12.3 GRPO reward

建议 reward 改为：

```text
R = answer_utility_delta
    + evidence_support_delta
    - unnecessary_rewrite_penalty
    - answer_leakage_penalty
    - off_topic_penalty
    - extra_round_cost
```

其中主项仍然是 answer utility delta。

## 13. 建议修改论文贡献表述

旧表述风险：

```text
训练一个 agentic RAG rewrite agent
```

过宽，容易和 ReasonRAG、MA-RAG、Search-R1 撞车。

建议新表述：

```text
We study low-information medical multimodal queries, where the initial textual query is too sparse to retrieve discriminative evidence. On top of a frozen multimodal RAG stack, we train a lightweight answer-utility-guided query rewrite controller that decides whether the current top-k evidence is sufficient and, if not, generates a targeted query emphasizing anatomical site, visual findings, and option-discriminative cues.
```

中文：

```text
本文关注低信息医学多模态 query 的检索失败恢复问题。在冻结的多模态 RAG 前端上，训练一个轻量级 answer-utility-guided query rewrite controller，使其判断当前 top-k 证据是否足够，并在证据不足时生成强调部位、视觉发现和选项区分信息的 targeted query。
```

## 14. 下一步建议

不要继续全量跑当前 v2 verifier。建议改为：

1. 删除/废弃 `image_content_type_identification` 模板题作为训练来源。
2. 保留 v2 脚本作为 smoke，不作为最终数据主线。
3. 新增 `generate_benchmark_style_mcq_with_api.py`：
   - 输入自有图文对 + PDF两页 + benchmark taxonomy；
   - 输出 benchmark-style MCQ candidates。
4. 新增/改造 verifier：
   - 核验答案；
   - 核验图像依赖；
   - 核验 rewrite training value。
5. 先跑 50-100 条 API generation + verification，人工检查。
6. 通过后再批量生成 4k accepted 数据。

## 15. 参考文献与链接

- ReasonRAG / Process vs. Outcome Reward: https://arxiv.org/abs/2505.14069
- MA-RAG / From Conflict to Consensus: https://arxiv.org/abs/2603.03292
- Doctor-RAG: https://arxiv.org/abs/2604.00865
- SoK Agentic RAG: https://arxiv.org/abs/2603.07379
- Search-R1: https://arxiv.org/abs/2503.09516
- ReSearch: https://arxiv.org/abs/2503.19470
