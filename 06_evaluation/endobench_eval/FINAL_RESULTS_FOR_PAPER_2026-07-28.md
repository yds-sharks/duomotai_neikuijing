# EndoBench 消融实验最终结果汇总（论文提交用）

> 生成时间：2026-07-28 | 数据全部实测核对完毕，可直接用于论文
> Benchmark：EndoBench test split，6832 题（4 场景 / 12 任务 / 4 类别）
> Controller：Qwen3.5-4B GRPO (ckpt_grpo_v4_u50) | Generator：Qwen3-VL-8B-Instruct（冻结，option-argmax 判分）
> 检索基座：Milvus Lite + BGE-M3 文本塔 + Qwen3-VL 图像塔 + 解剖路由 + 权重模块

---

## 一、主表（全量 6832 题）

| # | Configuration | Rounds | Persistent memory | Acc. (%) | Correct | Δ vs prev | 检索调用 |
|---|---|---|---|---|---|---|---|
| 0 | Baseline (no retrieval) | – | No | 40.59 | 2773/6832 | – | 0 |
| 1 | One-shot RAG | 1 | No | **38.85** | 2654/6832 | – | 1.00 |
| 2 | Single-round Agent | 1 | No | **40.12** | 2741/6832 | +1.27 | 1.00 |
| 3 | Iterative Agent (R3-NM) | ≤3 | No | **39.56** | 2703/6832 | −0.56 | 1.87 |
| 4 | Memory-aware Agent | ≤3 | Yes | **42.84** | 2927/6832 | +3.28 | 2.39 |

**两个关键量（全量）：**
- Δ_iteration = 39.56 − 40.12 = **−0.56pp**（纯多轮迭代无增益，甚至略负）
- Δ_memory = 42.84 − 39.56 = **+3.28pp**（记忆+协议层的贡献）

**Config 4 vs Single-round 全量配对（McNemar）：** b(对→错)=250, c(错→对)=436, χ²=49.89, **p<0.0001**, 净 +186 题 = +2.72pp ✔ 显著

---

## 二、四组消融定义（论文 Setup）

| 配置 | 决策方式 | 记忆 | 最大轮数 | 回答的问题 |
|---|---|---:|---:|---|
| One-shot RAG | 直接作答（全部12候选） | 无 | 1 | 普通 RAG 水平 |
| Single-round Agent | ACCEPT/REWRITE 一次后作答 | 无 | 1 | 控制器是否有效 |
| Iterative Agent | 不断 REWRITE 直到 ACCEPT/预算耗尽 | 无 | 3 | 多轮探索是否有效 |
| Memory-aware Agent | 不断 REWRITE 直到 ACCEPT/预算耗尽 | M⁺ 与 M⁻ | 3 | 记忆是否进一步提升 |

**Config 4 记忆机制必须如实描述为 "memory-aware agent with deterministic M⁺ initialization"：**
- M⁺（跨轮证据保留）：REWRITE 轮**无条件注入 top-3 候选**（deterministic，非控制器自主学会），触发率 52.1%（3562/6832 为 REWRITE 类）；ACCEPT 类（47.9%）用控制器自选 keep
- M⁻（失败查询面包屑）：每次 REWRITE 后追加失败 query，供下一轮参考
- 运行协议：Round-0 ACCEPT 仍继续多轮；多轮无有效证据时**复用 single-round 原答案**（保守回退）；绝不 closed-book

---

## 三、Config 4（42.84%）四象限分解（全量 6832，vs Single-round 40.12%）

|  | 3轮正确 | 3轮错误 | 合计 |
|---|---|---|---|
| **单轮正确** | 2491 (C→C) | 250 (C→W) | 2741 |
| **单轮错误** | 436 (W→C) | 3655 (W→W) | 4091 |
| **合计** | 2927 | 3905 | 6832 |

- 净增益 = 436 − 250 = **+186 题 = +2.72pp**，McNemar χ²=49.89, p<0.0001
- **增益全部来自 REWRITE 类难题**：REWRITE 题 rescue 342 / harm 137 = 净 +205；ACCEPT 题 rescue 94 / harm 113 = 净 −19
- harm 组 fallback（3轮无证据→复用单轮答案）：840/2741 = 30.6%

---

## 四、2×2 记忆消融（A100，1002 题严格 matched，新语义 closed-book=0）

标准迭代协议下（Round-0 ACCEPT 即结束），仅切换记忆开关：

| 臂 | 配置 | Acc. (%) | Correct | vs B | McNemar χ² |
|---|---|---|---|---|---|
| **B** | No Memory | **41.82** | 419/1002 | – | – |
| E | M⁺ Only | 41.42 | 415/1002 | −0.40 | 0.07 (n.s.) |
| F | M⁻ Only | 41.72 | 418/1002 | −0.10 | 0.00 (n.s.) |
| A | Full Memory | 40.62 | 407/1002 | −1.20 | 1.01 (n.s.) |

**2×2 效应分解：** M⁺ 主效应 −0.40pp，M⁻ 主效应 −0.10pp，联合 −1.20pp，交互 −0.70pp — **全部不显著**。
M⁺ 活跃度验证：E/A 臂 581 个多轮样本 100% 注入 M⁺（mean=3.00），B/F 臂全为 0 —— 机制确实生效，非空转。

**结论：标准协议下，记忆内容本身无显著增益（甚至略负）。**

---

## 五、Config4-NoMemory 判决实验（1002 题，rescue 协议 − 记忆）

完全保留 Config 4 协议（Round-0 ACCEPT 继续、多轮失败复用单轮答案、相同 ParseFail 处理），仅关闭 M⁺=OFF、M⁻=OFF：

| 指标 | 值 |
|---|---|
| **accuracy** | **41.32%** |
| correct count | 414/1002 |
| 平均轮数 | 2.63 |
| 平均检索调用 | 2.63 |
| ParseFail 率 | 5.49%（55/1002） |
| fallback 复用原答案 | 27.9%（280/1002） |
| 四象限 vs 单轮 | C→C=349, C→W=54, W→C=65, W→W=534（净 +11，χ²=0.84） |
| 逐题结果文件 | `c4nm_1002_per_question.jsonl` |

### 1002 题同子集三方对比（最终裁决）

| 配置 | Acc. (%) | Correct | 效应 |
|---|---|---|---|
| Single-round | 40.22 | 403/1002 | 基准 |
| **Config4-NoMemory** | **41.32** | 414/1002 | 协议效应 **+1.10pp** |
| **Config4-Memory** | **42.61** | 427/1002 | 再加记忆 **+1.30pp** |

**Memory vs NoMemory 配对（McNemar）：** Memory 赢 87, NoMemory 赢 74, 净 +13, χ²=0.89, **p≈0.34（1002 题上不显著）**

---

## 六、核心结论（论文 claim，三条证据链闭环）

1. **纯多轮迭代无效**：R3-NM 39.56% < Single 40.12%（Δ=−0.56pp，全量不显著）。多轮改写把简单题搅乱的损失 ≈ 救回难题的收益。

2. **标准协议下记忆内容无增益**：2×2 中 M⁺/M⁻ 主效应均 ≈0 且不显著，Full Memory 甚至 −1.20pp。

3. **约束协议下记忆才有效**：Config4 协议（强制多轮重试 + 失败保守回退）带来总增益 +2.72pp（全量显著）。分解为协议 +1.10pp 与记忆 +1.30pp（1002 题上各自不显著，方向为正）。

**总故事线（推荐表述）：**
> 增益并非来自"记忆内容改善检索"，而来自**置信度感知的选择性重试**（selective retry with answer preservation）：控制器的 ACCEPT/REWRITE 是不确定性信号（REWRITE 难题重试收益 15.9%，ACCEPT 题重试有害），配合"失败时保持原答案"的保守回退；记忆内容（deterministic M⁺ + M⁻）在此约束协议下提供方向为正、幅度较小的额外收益。

---

## 七、重要澄清：旧版"记忆有效"是 closed-book 假象

旧代码（预算耗尽/ParseFail 回退 closed-book）下的 Full Memory 表面 40.92%，但其中：
- 多轮样本 **70%（393/562）为 closed-book 裸答**（M⁺ 靠模型 keep 空转，mean 证据=0.98）
- 新语义修复后（强制 M⁺ + top-5 回退，closed-book=0），每题带 4.77 条真实证据，acc=40.62%
- 逐题对比：旧对→新错 84，旧错→新对 81，净 −3（噪声级），两版统计等价

**说明早期"记忆有效"是回退机制蒙对的假象，新语义下才是真实水平。** 论文应使用新语义结果。

---

## 八、跨机一致性验证

| 配置 | 本地 | A100 | 差异 |
|---|---|---|---|
| No-memory 3轮（新语义，1002子集） | R3-NM 41.72% | B 臂 41.82% | 0.10pp ✔ |

两台机器交叉验证通过，A100 结果可信。

---

## 九、辅助统计（全量 6832，供正文/附录）

**各基线场景 acc（Config 2 Single-round）：**
- Capsule Endoscopy 最难；Surgical Endoscopy 占比最大（2723 题）
- R3-NM 全量场景：Capsule 32.60% / Colonoscopy 43.56% / Gastroscopy 41.17% / Surgical 40.80%

**R3-NM 行为：** mean_rounds=1.87，多轮率 53.1%，ParseFail 3.59%，closed-book 仅 11
**Config 4 行为：** mean_rounds=2.39，ParseFail 3.94%，多轮率 100%（协议强制）

---

## 十、数据文件索引

| 结果 | 文件 |
|---|---|
| Baseline 全量 | `results/baseline_samples.jsonl` |
| One-shot RAG 全量 | `results_v2/vanilla_rag_samples.jsonl` |
| Single-round 全量 | `results_v2/agentic_samples.jsonl` |
| R3-NM 全量（3分片） | `results_full/R3_NM_shard{0,1,1b}/agentic_samples.jsonl` |
| Config4-Memory 全量 | `rescue_full_u50_shard{0,1}.jsonl` + `rescue_harm_u50_shard{0,1}.jsonl` |
| Config4-NoMemory 1002 | `c4nm_shard0.jsonl` + `c4nm_shard1a.jsonl` + `c4nm_shard1b.jsonl` |
| Config4-NoMemory 逐题 | `c4nm_1002_per_question.jsonl` |
| 2×2 消融（A100） | `_pkg/.../endobench_eval_pkg/results/{A,B,E,F}/` |
| 1002 分层子集 | `stratified_1000.jsonl` |

---

## 附：待决策项（如时间允许）

Config4-NoMemory 目前只有 1002 题（+1.30pp 记忆效应方向为正但不显著）。若要在论文中对"记忆净贡献"做显著性强声称，需全量 6832 跑一版 Config4-NoMemory（四卡约 10-12h）。当前 1002 题结论足以支撑"约束后记忆方向为正"的表述；总效应 +2.72pp（全量）已显著，可作为主 claim。
