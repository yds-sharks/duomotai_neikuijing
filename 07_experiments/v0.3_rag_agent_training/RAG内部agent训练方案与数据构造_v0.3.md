# RAG 内部 Agent 训练方案与数据构造（v0.3 规格）

> 状态：设计定稿，待实施
> 适用架构：`10_harness/` 中央 Agent Harness（大脑 + rag_search 单工具）
> 训练对象：**RAG 内部 agent**（rag_search 工具内部的检索改写 + 证据筛选 + 检索记忆）；大脑暂时冻结
> 训练路线：**SFT → GRPO**（不做 DPO）
> 基座：Qwen3.5-4B（策略 = 冻结生成器 = 图像编码视觉塔，全链路统一）；文本检索 BGE-M3 冻结

---

## 0. 一句话总结

旧训练数据存在**图像自命中泄露**，而且**没有文本检索路**，agent 学到的是捷径，学不到改写能力。v0.3 重建检索环境和题库，让 RAG 内部 agent 接收大脑给出的**疑惑点（info_need）**，在工具内部做**多轮改写探索**，并用**检索记忆（M+ 好文段 / M- 错误改写方向）**串起各轮。正负标签全部由冻结生成器的 answer-utility 自动计算。训练先做 SFT，再接 GRPO。

---

## 1. 背景：旧数据的问题诊断

### 1.1 旧数据是怎么来的
1. **出题**（`rerank_image_and_text/agentic/outputs/mcq_image_v2_4000/`）：取 `multimodal_samples.db` 的图像-文本对，用模板出题，答案取自图像说明文字。共 4000 题（train 3200 / dev 400 / internal_test 400），只有 2 种题型（`image_organ_identification`、`image_content_type_identification`）。provenance 显示 `generated_by=manual_template_image_query`，`api_verified=false`。
2. **跑轨迹**（`outputs/stage2_calibration/agent_context_v11_train3200.jsonl`）：GPT-4o 扮演 controller 跑冻结 RAG，输出 keep/drop + ACCEPT/REWRITE + rewrite_query。
3. **派生训练集**：SFT 3327 / RFT 2069 / DPO 偏好对 / GRPO 1672 状态；reward 由 Qwen3-VL-8B 生成器计算。

> 注：设计文档中的 `qa_gold_4000`（gpt-5.4 生成 + verifier 验证，4 类题型）在 `outputs/` 下没有找到成品，实际训练用的是上面的模板题库。

### 1.2 问题一：图像自命中泄露（致命）
题图本身就在向量库里。图像路用题图去检索，最先召回的是这张图自己，连带召回**出题用的、写着答案的说明文字**。

真实样例（`medalign_img_train_00002`）：
```
题目：该内镜图像主要观察到的是哪个部位？  A食管 B胃 C十二指肠 D胆胰   答案：B
题图来源：《消化内镜应用提升技巧》p53，sample_id=imgtxt_9abdcb3d_044374

候选[0] score=1.0018  ← 就是题图本身（sample_id 相同）
        “在NBI放大内镜观察下，胃黏膜呈现粉红色……”   ← 答案写在里面
候选[1] score=0.9630  同一本书的相邻图
候选[2] score=0.9459  同一本书的相邻图
```
统计（train3200 全量）：
- 候选含题图本身：**3200/3200**
- top1 score > 0.999：**2143/3200**
- gold sample_id 出现在候选中：**3200/3200**

后果：agent 学到的规律是“留下得分约等于 1 的那条，答案就在里面”。EndoBench 的图来自外部数据集，库里没有，这条捷径在评测时失效。

### 1.3 问题二：没有文本检索路，改写学不到东西
每轮 12 个候选**全部来自图像路，文本路 0 条**。图像路只认题图，换什么 query 结果都一样，改写没有任何学习信号。实测 REWRITE 只占 4.2%（134/3200），90.5% 的题一轮就结束。

### 1.4 问题三：题型与 EndoBench 不对齐
`content_type_identification`（“属于基础概念/病变特征/诊断评估/…”）是元问题，EndoBench 没有这类题，也不需要检索。

---

## 2. 架构与职责划分

### 2.1 两级查询链
```
大脑（题目语义层）——冻结，本阶段不训练
  读题 + 选项 + 图像 + 已有证据 → 分析“还不清楚什么”
  输出 rag_search(info_need="A 胃与 C 十二指肠在 NBI 下黏膜纹理的区别")
        │
        ▼
RAG 内部 agent（向量库语义层）——本阶段训练对象
  info_need + 题目上下文 + 检索记忆
    → 改写检索式 → 双路检索 → 判断正负 → 保留好文段 → 写记忆 → 继续改写或停止
  返回：本次新增证据 + 诊断（diag）+ 记忆摘要
```
- 大脑负责“需要什么”（疑惑点）；RAG 内部 agent 负责“怎么查到”（从正负样本中学到的向量库检索方向）。
- 大脑的动作空间不变（rag_search / submit_answer），只是参数从 `query` 改为 `info_need`。

### 2.2 与 v0.5 controller 的关系
v0.5 的 controller（keep/drop + ACCEPT/REWRITE + M+/M-）基本整体**回到 RAG 工具内部**，成为新的 RAG 内部 agent：

| v0.5 controller | v0.3 RAG 内部 agent |
|---|---|
| 输入原始 query | 输入大脑给的 info_need |
| ACCEPT / REWRITE | STOP / CONTINUE（工具内部的轮次控制） |
| rewrite_query | next_query |
| M+：程序保留 keep 的证据 | M+：agent 显式写入好文段并附理由 |
| M-：程序在 REWRITE 时记录 query（无原因） | M-：agent 显式写入错误改写方向并附**失败原因** |
| 候选只有图像路 | 文本路 + 图像路，留一检索 |

---

## 3. 检索记忆机制（M+ / M-）

### 3.1 作用
把每一轮检索的经验沉淀下来，供后续轮次的改写方向探索使用：
- **M+ 好文段**：已确认有帮助的证据，后续轮次不重复检索，改写可以围绕它补齐缺失信息
- **M- 错误改写方向**：召回不佳的 query 及**失败原因**，后续轮次避开同一方向

### 3.2 作用域
- **会话级**，挂在 `AgentSession` 上，在**同一题的所有轮次**之间共享：既包括一次 rag_search 内部的多轮改写，也包括大脑多次调用 rag_search
- **M+ 与会话证据集统一**：M+ 就是最终交给生成器的证据集（`session.collected`），不再维护两份
- 已检索过的候选 id 放入抑制集（沿用 v0.5 `candidate_id` + suppressed 逻辑），后续轮次不再重复出现

### 3.3 数据结构
```json
{
  "m_plus": [
    {"cid": "s:imgtxt_xxx", "summary": "胃窦 NBI 下腺管开口呈圆形小凹", "supports": "B", "why": "直接描述胃黏膜腺管特征", "from_query": "胃黏膜 NBI 腺管开口"}
  ],
  "m_minus": [
    {"query": "消化道黏膜 内镜 图像", "why": "过于宽泛，召回多为结肠和病理切片，偏离部位鉴别",
     "stats": {"n_recalled": 8, "n_kept": 0, "top_docs": ["结肠镜图谱", "病理学"]}}
  ],
  "tried_queries": ["消化道黏膜 内镜 图像", "胃黏膜 NBI 腺管开口"]
}
```
- `summary / supports / why / query / m_minus.why`：**由 agent 写**（学习目标）
- `stats / tried_queries / cid`：**由程序统计**（客观字段，不让模型编造）

### 3.4 写入规则
agent 每轮的输出中带 `memory` 字段，程序校验后再写入：
- `good` 中的编号必须是本轮候选中真实存在的编号
- `bad_direction.query` 必须等于本轮实际执行的 query
- 解析失败时回落 v0.5 规则：本轮 keep 为空且 CONTINUE → 自动把本轮 query 记入 M-（原因留空）

### 3.5 渲染给 agent 的形式（下一轮输入的一部分）
```
检索记忆
已确认的好文段（M+，共 2 条，无需重复检索）：
  [M+1] 胃窦 NBI 下腺管开口呈圆形小凹 —— 支持 B（来自「胃黏膜 NBI 腺管开口」）
  [M+2] ……
已失败的改写方向（M-，请避开）：
  - 「消化道黏膜 内镜 图像」：过于宽泛，召回多为结肠和病理切片（召回 8 / 保留 0）
已尝试的 query：……
```

### 3.6 对大脑的可见性
rag_search 返回给大脑的内容：本次新增证据 + diag + **M- 摘要**（让大脑避免提出同一方向的疑惑点）。大脑**不写记忆**，只读摘要。

---

## 4. RAG 内部 agent 的协议

### 4.1 工具内循环
一次 rag_search 调用内部最多 `K_inner = 3` 次检索，**不计入大脑预算**。
```
第 0 步：agent 读 info_need + 题 + 图 + 记忆（候选为空）→ 输出 next_query
第 r 步：用 next_query 双路检索（留一 + 抑制已见）→ 候选
         agent 读 info_need + 题 + 图 + 记忆 + 本轮候选
           → 输出 keep / memory / action(CONTINUE|STOP) / next_query / diag
         程序更新记忆；CONTINUE 且未到 K_inner 则进入下一步
结束：返回本次新增的 M+ 文段 + diag + M- 摘要
```

### 4.2 输出 JSON（每步一个对象）
```json
{
  "keep": [2, 5],
  "memory": {
    "good": [{"id": 2, "summary": "…", "supports": "B", "why": "…"}],
    "bad_direction": null
  },
  "action": "CONTINUE",
  "next_query": "十二指肠 绒毛结构 内镜表现",
  "diag": "胃部特征已覆盖（2 条），十二指肠对照证据不足，继续检索"
}
```
- 第 0 步候选为空时，`keep=[]`、`memory` 为空，只输出 `next_query`
- `action=STOP` 时 `next_query` 留空
- 键名保持英文（代码解析依赖），文本内容用中文

---

## 5. 数据构造（7 步）

### 第 0 步：检索环境修正（前置条件，必须先做）
- **留一检索**：检索时排除 `image_path == 题图`、`sample_id == gold`，以及**同一本书**（`doc_id` 相同）的全部候选
  - 主设定：同书排除（最接近 EndoBench 外部图像的情况）
  - 对照设定：只排除自身样本（用于消融，量化同书泄露的影响）
- **打开文本路**：候选为 BGE-M3 文本 + 视觉塔图像的混合，文本路 top-k 不得为 0
- **验收**：抽查 100 题，自命中 = 0，同书命中 = 0，文本候选占比 > 30%

### 第 1 步：题库重建
- 源池：`multimodal_samples.db` 的图像-文本对，沿用按 doc_id 的划分（train 19717 / dev 1530 / internal_test 2753）
- 出题：LLM 根据图像说明文字出多选题，说明文字只用于确定答案，**不进题面**；再用独立模型盲答验证（看图 + 题，不看说明文字），去掉仅凭题面就能答对或答案有歧义的题
- 题型（对齐 EndoBench）：解剖部位识别、病灶/发现识别、操作识别、病变定位等；**弃用** content_type 元问题
- 原 `mcq_image_v2_4000` 中的 organ 题（2000 道）可保留，在新检索环境下重新检索
- **支持度检查**：同书排除后，对每题用 oracle query（答案词 + 说明文字关键词）检索，计算最佳单文段 u；按结果把题分成：
  - 有支持（存在 `u_i > τ+` 的文段）：主体，目标 ≥ 70%
  - 无支持：保留 10% 左右，用来教 agent 如实停止并报告“语料覆盖弱”
- 目标规模：训练约 3000 题，dev / internal_test 各约 300 题

### 第 2 步：合成 info_need（模拟大脑的疑惑点）
- 教师模型（GPT）看题 + 选项 + 图，写 1~3 个疑惑点，要求：围绕选项之间的鉴别点、具体可检索、不直接写出答案
- 例：“A 胃与 C 十二指肠在 NBI 下黏膜纹理的区别”、“图中隆起病变是黏膜层还是黏膜下来源”
- 每个（题, info_need）对是一条 RAG 内部 agent 的训练起点
- 副产物：这批数据就是后续大脑 SFT 的种子数据

### 第 3 步：第 1 轮改写探索
- 每个（题, info_need，记忆为空）采样 **N = 8** 个检索式：6 个来自 4B 温度采样（T=0.9），2 个来自教师模型
- 每个检索式真实检索一次（第 0 步的环境），记录候选

### 第 4 步：正负标注
冻结生成器（Qwen3.5-4B）计算 answer-utility：
```
u(E) = logP(gold | 题, 图, E) − logP(gold | 题, 图, ∅)
例：无证据 P(B)=0.40，加入证据后 P(B)=0.75  →  u = ln0.75 − ln0.40 = +0.63
```
- **单文段标注**：对 8 个检索式召回候选的**并集**逐条计算 `u_i = u({p_i})`
  - 正类：`u_i > τ+`（初值 0.2）
  - 有害负类：`u_i < τ−`（初值 −0.2），最需要学会丢弃
  - 无关负类：`|u_i| ≤ 0.2`
- **检索式标注**：每个检索式取其召回中的正类组成集合，计算 `u_set`，用来给 8 个检索式排名
- **记忆标签**：
  - M+ 目标：本轮候选中的正类
  - M- 目标：`u_set < ε`（初值 0.1）的检索式记为错误方向；失败原因由教师模型根据候选摘要与标签写一句话，写不出时用模板：“召回 N 条无正类，主要来自 X”

### 第 5 步：多轮分支（多轮探索与记忆数据的核心）
从第 1 轮的 8 个检索式中挑两类作为“已执行”的第 1 轮，分别向后展开：
- **失败分支**：执行最差检索式 → 写入 M-（及偶然命中的 M+）→ 以该记忆为条件再采样 8 个第 2 轮检索式 → 检索（抑制已见）→ 标注 → 最佳者作为第 2 轮目标
  - 学的是：**看到失败方向后，往哪里改**
- **成功分支**：执行最佳检索式 → 写入 M+ → 同样采样第 2 轮 → 计算边际增益 `Δu = u(M+ ∪ 新正类) − u(M+)`
  - `max Δu ≥ ε_c`（初值 0.15）→ 标为 CONTINUE，目标为增益最大的检索式
  - 否则 → 标为 STOP
  - 学的是：**什么时候已经够了**
- 第 3 轮按同样规则只在 CONTINUE 分支上继续展开（K_inner = 3）

### 第 6 步：组装样本
每个决策状态一条样本，分四类：

| 类型 | 状态 | 目标 | 比例（目标） |
|---|---|---|---|
| T1 首轮规划 | 记忆为空、候选为空 | next_query（第 1 轮最佳） | 25% |
| T2 筛选 + 记忆 + 继续 | 有候选，覆盖不足 | keep 正类 + 写记忆 + CONTINUE + next_query | 35% |
| T3 筛选 + 记忆 + 停止 | 有候选，已覆盖 | keep 正类 + 写记忆 + STOP | 30% |
| T4 无支持 | M- 累积、无正类 | keep=[] + 写 M- + STOP + diag“语料覆盖弱” | 10% |

样本格式：
```json
{
  "qid": "...", "split": "train", "round_idx": 1, "sample_type": "T2",
  "input": {
    "info_need": "A 胃与 C 十二指肠在 NBI 下黏膜纹理的区别",
    "question": "...", "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
    "query_image_path": "...",
    "memory": {"m_plus": [...], "m_minus": [...], "tried_queries": [...]},
    "candidates": [{"id": 1, "origin": "text", "text": "...", "doc_id": "..."}, ...]
  },
  "target": {
    "keep": [2, 5],
    "memory": {"good": [...], "bad_direction": null},
    "action": "CONTINUE",
    "next_query": "十二指肠 绒毛结构 内镜表现",
    "diag": "..."
  },
  "labels": {
    "passage_utility": {"1": -0.31, "2": 0.41, "5": 0.33},
    "query_utility_group": [0.63, 0.12, -0.05, ...],
    "delta_u_next": 0.22
  }
}
```
`labels` 不参与 SFT 损失，留作分析和 GRPO 初始化参考。

### 规模与成本估算
- 约 3000 题 × 1.5 个 info_need ≈ 4500 个起点
- 第 1 轮：4500 × 8 次检索；单文段标注按并集约 30 条 → 约 13.5 万次 4B 前向
- 第 2、3 轮分支约再增加 1 倍
- 合计约 30 万次 4B 前向（短输入，单卡可批量），检索约 8 万次
- 降本开关：单文段标注只覆盖每个检索式的 top-5；多轮分支只对 50% 起点展开

---

## 6. 训练方案：SFT → GRPO

### 6.1 为什么不做 DPO
- DPO 的作用是“好坏对比”。GRPO 的组内相对优势本身就是在策略自己的分布上做好坏对比，而且是在线的，比离线偏好对更贴近真实检索
- 负类文段的信息通过两条路径学进去：SFT 的 keep 目标只含正类；GRPO 对保留负类的行为扣分
- 少一个阶段，就少一套偏好对数据和一组超参，链路更简洁

### 6.2 阶段 A：SFT（冷启动）
- **目标**：学会协议格式、基础改写能力、正负判断和记忆写法
- **数据**：第 6 步的 T1–T4 样本；dev 按 doc 划分，用于早停
- **训练**：沿用已跑通的 Qwen3.5-4B 全参 FSDP SFT 流水线（wrap `Qwen3_5DecoderLayer` / `Qwen3_5VisionBlock`，`enable_thinking=False`），只对 target 部分计算损失
- **输入**：图像 + 文本（与 harness 的多模态消息格式一致）
- **验收**（dev）：
  - JSON 格式合法率 > 98%
  - keep 正类精度 > 0.7
  - 首轮 next_query 平均 u_set 高于 4B 零样本基线
  - STOP/CONTINUE 判断准确率 > 0.75
  - next_query 与 M- 方向重复率 < 10%
- **GRPO 前提**：格式合法率不达标就不进入 GRPO（格式错误会导致整组 rollout 作废）

### 6.3 阶段 B：GRPO（在线探索）
**单次 rollout**：对一个（题, info_need），策略从第 0 步开始完整跑一次工具内循环（最多 3 次检索，带记忆），得到最终 M+。

**分组**：每个起点采样 G = 8 条 rollout（T = 0.9），组内计算相对优势：
```
A_i = (R_i − mean(R)) / max(std(R), 0.5)
```
（沿用 `train_ctrl_grpo.py` 的组优势逻辑，从单步改为多轮 episode，优势作用于该 episode 内全部决策 token）

**奖励**：
```
R = u(M+_final)                          主项：最终证据集对正确答案的帮助
  + α · keep_precision                   保留文段中正类占比（只对保留的文段在线计算单文段 u）
  + β · m_minus_acc                      写入 M- 的方向是否确实失败（该轮新增正类的 u_set < ε，可验证）
  − γ · repeat_rate                      next_query 与已有 M- 方向的 BGE-M3 余弦相似度 > 0.9 的比例
  − δ · n_retrievals                     检索次数成本
  格式非法：R = −1
初值：α = 0.3, β = 0.2, γ = 0.5, δ = 0.05（在 dev 上调）
```
- `u(M+_final)` 让模型学会召回真正有用的证据
- `keep_precision` 惩罚保留负类
- `m_minus_acc` 和 `repeat_rate` 训练记忆机制：如实记录失败方向，并在后续轮次避开
- `δ` 让模型在已经够用时尽早 STOP

**稳定性**：
- KL 约束到 SFT 参考策略（系数初值 0.02）
- 每 50 步在 dev 的 24 题探针集上记录 reward 和格式合法率；格式合法率下跌 > 5% 时回滚
- 题目按支持度分层采样，避免无支持题过多导致组内 reward 全为 0、没有梯度

**验收**（internal_test，未见过的书）：
- 平均 `u(M+_final)` 高于 SFT
- 生成器正确率高于 SFT
- 第 2 轮相对第 1 轮的平均增益 > 0（多轮探索确实有用）
- M- 重复率低于 SFT

### 6.4 阶段 C：接回 harness 联调
- 训好的 agent 替换 `backends/rag_backend.py` 的 `AgentEvidenceFilter`，大脑保持冻结
- 在 internal_test 和 EndoBench 子集（50–100 题，中文翻译缓存已就绪）上跑，对比 v0.2.2 基线
- 之后再训练大脑（种子数据来自第 2 步，再加 harness 真实轨迹）

---

## 7. 评测与消融

### 7.1 指标
| 维度 | 指标 |
|---|---|
| 答案 | 生成器正确率；平均 u(M+_final) |
| 检索 | 每题正类召回数；文本候选占比 |
| 筛选 | keep 正类精度；有害负类误保留率 |
| 记忆 | M- 标记准确率；next_query 与 M- 重复率；第 2/3 轮边际增益 |
| 成本 | 平均内部检索次数；STOP 准确率 |

### 7.2 消融（论文贡献证据）
1. **记忆消融**：完整 M+/M- vs 去掉 M-（沿用 v0.5 `no_mminus` 开关思路）vs 去掉 M+ vs 都去掉
2. **记忆写法消融**：agent 写失败原因 vs 只记录 query（v0.5 原版）
3. **泄露对照**：旧数据（自命中）训练 vs 新数据训练，在 EndoBench 上对比，量化泄露带来的虚高
4. **检索环境**：同书排除 vs 只排除自身样本
5. **训练阶段**：零样本 vs SFT vs SFT+GRPO

---

## 8. 旧资产复用清单

| 资产 | 位置 | 处理 |
|---|---|---|
| `multimodal_samples.db` 源池 + doc_id 划分 | `data_house/` | ✅ 直接复用 |
| organ 题 2000 道 | `outputs/mcq_image_v2_4000/` | ✅ 题目保留，在新环境下重新检索 |
| content_type 题 2000 道 | 同上 | ❌ 弃用 |
| 出题与盲答验证流程 | `data_construction/verify_*_with_api.py` 等 | ✅ 流程复用，按新题型重新跑 |
| M+/M- 记忆逻辑、`candidate_id`、抑制集 | `code/trajectory_runtime.py` | ✅ 复用，并升级为 agent 显式写入 + 失败原因 |
| best-of-N 改写方法（k3：best-of-8 使 logP 从 −2.95 提升到 −1.66） | `outputs/stage2_calibration/verify_bestofN_rewrite.py` | ✅ 作为第 3、5 步的核心方法 |
| KB 支持度验证（k1 oracle query） | `outputs/stage2_calibration/verify_kb_support.py` | ✅ 用于第 1 步支持度检查 |
| answer-utility 计算 | `10_harness/backends/transformers_backend.py`（4B 版） | ✅ 复用 |
| GRPO 组优势逻辑 | `train/train_ctrl_grpo.py` | ✅ 复用，由单步扩展为多轮 episode |
| 全参 FSDP SFT 流水线 | 已在 Qwen3.5-4B 上跑通 | ✅ 复用，更换数据 schema |
| train3200 轨迹与标签 | `outputs/stage2_calibration/` | ❌ 不用于训练；保留作泄露对照实验 |
| 8B 生成器 reward bank | `train/rewards_sample_v1_*` | ❌ 数值作废（生成器已换成 4B） |
| SYSTEM_PROMPT_V11 | — | ❌ 按第 4 节协议重写 |
| DPO 数据与脚本 | `train/build_dpo_pairs.py` | ❌ 本方案不使用 |

---

## 9. Harness 侧改动清单

1. `tools/rag_search.py`：参数 `query` 改为 `info_need`（保留 `query` 作直接检索模式，供消融）
2. `runtime/session.py`：新增 `RetrievalMemory`（m_plus / m_minus / tried_queries），m_plus 与 collected 统一
3. `backends/rag_backend.py`：`AgentEvidenceFilter` 升级为 `RagInternalAgent`，实现工具内循环（K_inner = 3）、记忆读写与校验、规则回落
4. `FirstStageRetriever` 调用：增加留一 / 同书排除参数，并确认文本路打开
5. `prompts/`：新增 RAG 内部 agent 提示词（中文，键名英文）；大脑策略段改为“分析疑惑点、描述信息需求”
6. rag_search 返回值增加 diag 与 M- 摘要，渲染进大脑的状态消息

---

## 10. 实施顺序

| 步骤 | 内容 | 验收 |
|---|---|---|
| P0 | 第 0 步检索环境修正 + 100 题抽样 | 自命中 = 0、同书命中 = 0、文本候选 > 30% |
| P1 | 支持度抽样检查（100 题，同书排除） | 得到有支持题比例，决定源池是否够用 |
| P2 | 题库重建 + info_need 合成 | 约 3000 题通过盲答验证 |
| P3 | 第 3–5 步 rollout 与标注（先跑 200 题小样，校准 τ+ / τ− / ε / ε_c） | 正负类比例合理、STOP/CONTINUE 分布合理 |
| P4 | 全量标注 + 组装样本 | T1–T4 比例接近目标 |
| P5 | SFT | 第 6.2 节验收 |
| P6 | GRPO | 第 6.3 节验收 |
| P7 | harness 联调 + EndoBench 评测 + 消融 | 第 7 节指标 |

---

## 11. 待定事项
1. **同书排除**：建议同书排除作主设定，只排除自身样本作对照
2. **题型范围**：建议对齐 EndoBench 题型，参考原规划的 4 类（解剖部位 35% / 病灶识别 30% / 操作识别 20% / 空间区域 15%）
3. **教师模型**：建议 info_need 与失败原因用 GPT，改写采样以 4B 自采样为主（GRPO 本来就要在自身分布上探索）
4. **阈值**：τ+ / τ− / ε / ε_c 的初值需在 P3 小样上校准
