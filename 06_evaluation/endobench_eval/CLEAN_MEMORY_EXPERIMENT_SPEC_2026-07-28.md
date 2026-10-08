# Clean Memory-Aware Experiment Specification

日期：2026-07-28  
优先级：P0，服务器立即执行  
目标：公平验证“迭代检索”和“正负记忆”是否分别带来准确率与检索效率收益。

## 0. 结合现有全量结果后的最终执行结论

以下三组已经具有全量结果，不重复运行：

| 组别 | 状态 | Overall Accuracy |
|---|---|---:|
| One-shot domain-adaptive RAG（Classification + Weight + RAG） | 已完成，6,832题 | 39.98% |
| Single-round Agent | 已完成，6,832题 | 40.12% |
| 3-Round Memory-Aware Agent | 已完成，6,832题 | 42.81% |

One-shot RAG 的四个 category 结果为：49.18%、31.51%、39.17%、43.25%，Overall 为 39.98%。

**现在只需要新增一组全量实验：**

> **3-Round Iterative Agent without Memory：关闭 M+ 和 M-，其他所有设置与现有 3-Round Memory-Aware Agent 完全一致。**

运行前先审计现有 42.81% 结果的日志与脚本。必须确认新旧两组使用相同的：

- 6,832 个 qid 及顺序；
- 检索前端、候选缓存和 top-k；
- controller 与 generator checkpoint；
- 最大检索次数；
- ACCEPT、预算耗尽和 ParseFail 回退；
- 解码参数与随机种子。

如果上述条件一致，只运行无记忆组即可；如果现有 42.81% 使用了不同回退、强制 M+、不同前端或不同候选缓存，则不能称为 clean paired comparison，需要先统一该差异。

### 为什么统一设为3轮

本文把“3轮”严格定义为：

```text
Round 1: 初始查询检索
Round 2: 第一次 REWRITE 后检索（仅未 ACCEPT 时）
Round 3: 第二次 REWRITE 后检索（仅仍未 ACCEPT 时）
```

- 最多进行 3 次检索调用，而不是“初始检索后再 rewrite 3 次”。
- 每一轮均允许 controller 提前 ACCEPT；不是强制跑满3轮。
- 第3轮结束仍未 ACCEPT 时按统一规则强制终止。
- Memory 与 No-Memory 两组使用完全相同的3轮上限。

选择3轮是为了给控制器提供两次纠正初始检索方向的机会，同时限制临床问答中的检索成本和误差累积。论文比较的不是“记忆组拥有更多轮次”，而是**在相同3轮预算下，历史信息能否提高准确率或减少平均检索调用数**。

### 本轮唯一新增命令应表达的差异

```text
base_config = existing_3round_full_config
use_mplus = false
use_mminus = false
max_retrieval_calls = 3
early_stop_on_accept = true
```

除这两个 memory 开关外，禁止修改其他参数。

## 1. 本轮只回答两个问题

1. 在相同检索基座下，允许控制器迭代改写是否优于一次决策？
2. 在相同三轮预算下，加入 M+ / M- 是否优于无记忆迭代？

旧版 A/B/E/F 结果仅作为实现诊断，不进入主结果表。旧版 Full 中 M+ 基本为空、M- 直接扰动检索分布，不能作为记忆假设的有效检验。

## 2. 共享且冻结的检索基座

所有配置严格复用同一候选缓存、模型、索引和检索参数：

- 解剖路由：`scene -> body_site_main`，Milvus expr 预过滤。
- 文本塔：BGE-M3，每轮 top-20。
- 图像塔：Qwen3-VL-8B-Instruct 图像编码器，每轮 top-20。
- 自适应融合：同一 mengzi 密度/依赖模块与归一化方式。
- 候选池：每塔 top-6，共 12 个候选。
- Controller：同一版本 checkpoint。
- Generator：同一冻结模型、提示词和解码参数。
- 数据顺序、图像、问题翻译缓存和随机种子完全一致。

禁止为任一配置单独改变器官过滤、通道权重、候选数量或生成器回退。

## 3. 四个主配置

### R0: One-shot RAG

- 初始查询只检索一次。
- 不调用 controller。
- 融合排序后的 top-5 直接交给 generator。
- 检索调用数恒为 1。

### R1: One-decision Agent

- 初始检索得到 `C_1`，controller 只决策一次。
- `ACCEPT`：使用 `KEEP_1`；若为空，使用当前池 top-5。
- `REWRITE`：执行一次新检索得到 `C_2`，随后强制结束并使用 `C_2` 的 top-5。
- 最多允许一次 rewrite，检索调用数为 1 或 2。
- 不使用 M+、M-。

### R3-NM: Iterative Agent without Memory

- 最多 3 次检索调用，即初始检索加最多 2 次 rewrite。
- 每轮 controller 自主选择 `ACCEPT` 或 `REWRITE`。
- `ACCEPT`：使用当前轮 `KEEP_t`；若为空，使用当前轮 top-5。
- `REWRITE`：上一轮候选与查询不传入下一轮；只使用新 query 重新检索。
- 第 3 次检索后仍未 ACCEPT：使用当前轮 `KEEP_t`；若为空，使用当前轮 top-5。
- 不使用 M+、M-。

### R3-MEM: Memory-Aware Iterative Agent

- 检索次数上限、controller、generator及其余设置与 R3-NM 完全一致。
- Controller 动作必须允许：`REWRITE + partial KEEP`。
- `M_t+` 保存此前轮次中 controller 明确保留、但不足以立即作答的证据。
- `M_t-` 保存失败查询及简短失败原因，供下一轮 controller 和 query rewrite 参考。
- M- 不得直接过滤候选、修改融合分数、屏蔽向量邻居或改变 Milvus 检索逻辑。
- `ACCEPT`：最终证据为 `dedup(M_t+ union KEEP_t)`，按保留优先级截断到 5 条。
- 第 3 次检索后仍未 ACCEPT：使用 `dedup(M_t+ union KEEP_t)`；若为空，使用当前轮 top-5。
- ParseFail：采用当前轮已积累的 M+ 加当前 top-5，去重后截断到 5 条；若 M+ 为空则只使用当前 top-5。

## 4. 记忆的严格更新规则

每轮输出统一为：

```json
{
  "action": "ACCEPT | REWRITE",
  "keep": [0, 2],
  "rewrite_query": "...",
  "failure_reason": "missing anatomical region evidence"
}
```

更新规则：

```text
K_t       = candidates selected by keep
M+_{t+1}  = dedup(M+_t union K_t)
M-_{t+1}  = append(M-_t, {query_t, rejected direction, failure_reason})
```

约束：

- 只有 `REWRITE` 时更新 M-。
- `REWRITE` 允许 `keep` 非空，不得自动清空。
- M+ 最多保存 5 条，按 controller 保留顺序和原始融合分数稳定截断。
- M- 最多保留最近 2 条轨迹，避免上下文膨胀。
- 候选内容、M+ 和 M- 使用不同字段，禁止拼接成不可区分的自然语言段落。
- 相同文档 ID 只保留一次。

## 5. 公平终止与回退

- “3轮”统一定义为最多 3 次检索调用，不是 3 次 rewrite。
- 所有 controller 配置遇到预算耗尽均在当前状态结束，不额外发起检索。
- R1、R3-NM、R3-MEM 的空 KEEP 均回退到当前轮 top-5。
- 禁止某组回退 closed-book、另一组回退 top-5。
- ParseFail 不得改变动作预算，也不得为某一配置额外增加检索机会。
- 每道题使用相同 seed；结果必须按 `qid` 对齐比较。

## 6. Controller 训练要求

现有单轮 u50 checkpoint 可用于代码 smoke test，但不能作为最终 Memory-aware 结论。

最终 R3-MEM controller 的训练状态必须包含：

- 当前问题、图像与候选池。
- 当前轮数和剩余检索预算。
- 非空与空 M+ 的样本。
- 非空与空 M- 的样本。
- `REWRITE + partial KEEP` 的监督轨迹。
- 在证据充分时 ACCEPT、证据部分有效时 KEEP 后 REWRITE、方向错误时改写并记录原因。

最低可行训练顺序：

1. 从现有 RFT checkpoint 初始化。
2. 用多轮轨迹做 memory-aware SFT/RFT，使动作格式和 partial KEEP 先稳定。
3. 再做短程 GRPO；奖励使用最终生成效用并加入检索成本。

建议奖励：

```text
R = U(E_T) - lambda_round * (retrieval_calls - 1)
             - lambda_parse * parse_fail
```

训练和评测均不得使用 gold answer 构造推理期 M+ 或 M-。

## 7. 运行顺序

### Stage 0: 32题代码验收

从现有 Memory 结果中取同一 32 题，仅运行 No-Memory 配置并逐题对齐。通过条件：

- 结果为 32 条，qid 与 Memory 组完全一致。
- No-Memory 与 Memory 的检索调用数均不超过 3。
- No-Memory 日志中的 M+、M- 始终为空。
- 除 memory prompt/state 字段外，两组初始候选必须完全一致。
- 无重复 qid、无断点重复写入。

### Stage 1: 200题配对验收

在同一200题上比较现有 Memory 与新增 No-Memory：

- Accuracy 与逐题正确性转移。
- 平均检索调用数和各轮 ACCEPT 率。
- Memory 组的 M+ 非空率及最终使用率。
- ParseFail 率。
- 预算耗尽率及回退来源。

若发现两组回退逻辑不同、初始候选不同，或现有 Memory 结果使用了强制注入而论文声称模型自主 KEEP，则停止并先统一实验定义。

### Stage 2: 全量 6,832题 No-Memory

- 只运行 R3-NM。
- 与现有 R3-MEM 逐 qid 配对。
- 输出 paired bootstrap 95% CI 和 McNemar 检验。
- 同时输出 Accuracy、平均检索调用数、各轮 ACCEPT 率、预算耗尽率和 ParseFail。

## 8. 每题必须保存的字段

```text
qid, index, scene, category, task, gold
config, pred, correct, logp_gold
retrieval_calls, stop_round, stop_reason
actions_by_round, queries_by_round
keep_by_round, candidate_ids_by_round
mplus_ids_by_round, mminus_by_round
final_evidence_ids, final_evidence_source
parse_ok_by_round, fallback_reason
elapsed_total, elapsed_retrieval, elapsed_controller, elapsed_generator
```

## 9. 最终主表

| Method | Memory | Accuracy | Delta | Avg. retrieval calls | R1/R2/R3 accept | Budget exhausted | ParseFail |
|---|---:|---:|---:|---:|---:|---:|---:|
| One-shot RAG | - | | | 1.00 | - | - | - |
| One-decision Agent | - | | | | | | |
| Iterative Agent | No | | | | | | |
| Memory-aware Iterative Agent | M+ / M- | | | | | | |

论文的有效结论判据：

- `R3-NM > R1`：多轮迭代查询精炼有效。
- `R3-MEM > R3-NM`：记忆带来额外准确率收益。
- 若准确率接近但 R3-MEM 的平均调用更少：记忆带来检索效率收益。
- 必须同时报告准确率与成本，不以单个子集的最好数字替代全量配对结果。

## 10. 暂不运行的旧消融

以下实验先暂停，待主比较成立后再补：

- M+ only / M- only。
- Selection-only / Rewrite-only。
- Top-1 / Top-3 / Top-5 memory capacity。
- 强制每轮 rewrite 的 Sequential。
- SFT / RFT / GRPO 训练阶段对比。

这些属于解释性消融，不能替代 R3-MEM 与 R3-NM 的核心比较。
