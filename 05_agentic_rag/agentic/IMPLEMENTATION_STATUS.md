# AgenticRL 实现进度与实验审计文档

> 最后更新：2026-07-22（GRPO v4 训练进行中，约 66%）
> 状态标记：✅ 已完成 / 🔄 进行中 / ⬜ 未开始（TODO）

---

## 1. Controller 具体定义

### 1.1 模型与参数量

| 项 | 值 |
|---|---|
| 基座 | Qwen3.5-4B（本地快照 `models--Qwen--Qwen3.5-4B`） |
| 参数量 | ~4B，**全参数训练**（非 LoRA） |
| 当前训练初始化 | `train/ckpt_ctrl_rft_v3`（SFT→RFT 后的检查点） |
| 输出目录 | `train/ckpt_grpo_v4`（中途 ckpt：u50/u100 已存盘，save-every=50） |

### 1.2 输入格式（与训练时逐字一致）

- System：`SYSTEM_PROMPT_V11`（中文，定义 keep/drop 价值判断 + ACCEPT/REWRITE 决策规则）
- User（多模态，按序排列）：
  1. `USER_TEMPLATE` 文本：qid、query_type、原始 query、当前检索 query、问题、选项 JSON、**0-based 编号的候选证据块**（每条含来源/doc 名/页码/首阶段分数/文本截断 700 字符）
  2. `查询图像：` + query image（长边 768px）
  3. `候选证据图像 [idx]：` + 各候选的证据图像（长边 384px，最多 7 张，总图像上限 max_images=8）
- 渲染：`processor.apply_chat_template(..., enable_thinking=False)`

### 1.3 输出格式（动作 schema —— ✅ 已固定）

```json
{
  "keep": [0, 2],           // 0-based 候选下标，≤3 条，训练 prompt 明确约束
  "drop": [1, 3],           // 与 keep 互补
  "action": "ACCEPT" 或 "REWRITE",
  "rewrite_query": "...",   // ACCEPT 时为空字符串
  "reason": "≤80字"
}
```

- 解析器：`train/grpo_reward.py:parse_action`（正则提取 JSON → 下标越界/重复清洗 → drop 自动补全；解析失败 fallback：action=ACCEPT, keep=[], parsed=False）
- 该 schema 自 SFT → RFT → GRPO 三阶段未变，**已冻结**；评测（含 GPT-4o 基线）使用完全相同的 schema 与解析器。

### 1.4 M+ / M-（跨轮记忆）

**⬜ 当前实现中不存在 M+/M-。** 需要如实说明：

- M+/M-（正面/负面证据记忆）是 MedAlign-RAG 控制器 v2.1 旧设计的概念，**未被带入当前 AgenticRL 实现**。
- 当前控制器是**单轮决策**：对每个状态（一次初始检索的结果）做一次 keep/rewrite 决策，REWRITE 触发一次重检索后即终止，没有第二轮，因此没有跨轮记忆的载体。
- 训练数据字段里唯一与"历史"相关的是 prompt 中允许出现的 `search_history` 面包屑候选（上一轮检索线索），当前训练数据中该字段为空。
- **对论文的含义**：当前投稿版本是单轮 agentic 控制；多轮 + M+/M- 属于未来工作，不要在论文中暗示已实现。

### 1.5 M- 是否保证不含 gold utility / 答案标签

- 既然 M+/M- 不存在，此问题对当前实现不适用。
- 但需要记录**奖励侧的防泄漏机制**（✅ 已实现，`train/grpo_reward.py` + `reward_model.py`）：
  - reward 只通过冻结生成器对 gold 选项的 logP 计算，**控制器永远看不到 gold 字母**；
  - 硬泄漏惩罚：`leak_penalty=0.5`，仅针对 controller 输出文本中出现"答案是X"/"最终答案"模式（`hard_leakage`）；answer_text（答案文本词）命中仅监控不惩罚；
  - prompt 约束：rewrite_query 不得包含答案字母、不得说"正确答案是…"。

---

## 2. RAG 环境

### 2.1 组件清单（`code/agentic_runtime_config.json`）

| 组件 | 实现 |
|---|---|
| 向量库 | Milvus Lite，`data_house/milvus/multimodal_vector_indexes.db`，nprobe=64 |
| 文本编码器 | BGE-M3（dense+sparse 混合检索） |
| 图像编码器 | Qwen3-VL-8B-Instruct（作为图像 embedding 模型） |
| 生成器（冻结） | Qwen3-VL-8B-Instruct（bf16），训练/评测中全程冻结 |
| 重排器 | ⚠️ 配置中存在（Qwen3-VL-Embedding-2B + PPR），**但训练与评测循环均未使用**——候选排序纯按首阶段检索分数（`select_top_evidence`） |

### 2.2 检索单元

- **文本单元**：中文消化内镜书籍语料切分后的 passage（chunk），每条带 doc_id/page_idx/block_id；检索后回查配对图像，形成"图文对"（`rebuild_obs_candidates.py` 修正过的逻辑：文本命中补全 image_path）
- **图像单元**：书页/插图图像及其图注文本，与文本单元同库存储、独立集合检索
- 语料：自建中文消化内镜书籍语料（`data_house/multimodal_samples.db` 主库），与 EndoBench 无任何样本级重叠

### 2.3 预算参数（✅ 已固定）

| 参数 | 值 | 说明 |
|---|---|---|
| 初始检索 | text top-20→选 6 + image top-20→选 6 = **12 候选** | 与 `rebuild_obs_candidates.py` 配方一致 |
| 最大轮数 | **1**（单轮决策；REWRITE = 一次重检索后终止） | 见 §1.4 |
| keep 上限 | ≤3 条（prompt 硬约束） | |
| REWRITE 重检索 | text top-**5**（`--rewrite-topk 5`） | 仅文本通道（改写只影响文本 query，query 图不变） |
| 送入 generator 的证据 k_ctx | ACCEPT：keep 的 0~3 条；REWRITE：重检索 5 条；vanilla 基线：全部 12 条 | generator 图像上限 7+1 |

### 2.4 REWRITE 是否真实重新查库

✅ **是。** 训练时 `--rewrite-retrieval 1`：每个 REWRITE rollout 的 rewrite_query 真实执行 BGE-M3 文本检索（Milvus 在线查询，无缓存/代理/模拟），检回 top-5 后由冻结生成器打分。这就是论文"closing the loop between rewriting, retrieval, and reward computation"的实现。日志确认 `rw_retr=1`。

---

## 3. 奖励与训练

### 3.1 Utility 完整定义（✅ 已固定）

```
u(evidence) = logP_options(gold | question, image, evidence)
            − logP_options(gold | question, image, ∅)
```

- **logP 计算**：生成器在 prompt 末尾"…请只输出唯一正确选项的字母（A/B/C/D）。答案："位置做**单次前向**，取下一 token 分布在各选项字母（含空格前缀变体，logsumexp 合并）上的 logits，**仅在有效选项集合上做 softmax**（options-normalized），取 gold 字母的 log 概率。
- **长度归一化**：不需要——打分位置是单一 next-token（字母），不是序列生成，无长度维度。
- **Baseline**：**no-evidence（∅）**，不是 current-state。即衡量"该证据集合相对完全无证据的增量"。
- 实现：`train/gen_scorer.py:AnswerScorer`（`judge()` 一次前向同时给出 4 选项分布、argmax 预测、logp_gold），与评测同源。

### 3.2 奖励粒度

- **逐 rollout（=逐动作）**：组内每个 rollout 独立产生动作 → 形成各自证据集 → 独立计算 u。单轮决策，无"终局"概念。
- GRPO 组相对优势：`A_i = (u_i − mean(u)) / max(std(u), adv_std_floor=0.5)`；
- 噪声门控：组内 `max−min < min_spread=0.3` 跳过更新；`gate_eps=0.05` 预检（none/top3/all 三档证据下 logP 跨度不足直接跳过该状态）。

### 3.3 三阶段训练配置（✅ 全部记录）

| 阶段 | 数据来源 | 样本量 | 关键超参 |
|---|---|---|---|
| SFT（冷启动） | GPT-4o teacher（`GPTContextAgent`，相同 v11 prompt）在自建 3200 题上的轨迹 | train 3327 / val 176 | 全参数，手动 CE |
| RFT（拒绝采样微调） | Stage-D 奖励筛选的策略自采样：rewrite-win / selection / trivial 三源各取一 | train 2069 / val 85 | 同上 |
| GRPO（当前） | 3200 题 → tier 过滤（selection+rewrite+15%trivial）→ **1672 状态**（尾部 24 留作 probe） | G=8/组 | lr=3e-6（Adafactor），β_kl=0.02，T=1.4，top_p=0.95，G=8，forced-rewrite=2/组（prefix 不进梯度），grad_accum=8，pg_batch=4，clip=5.0，epoch=1，seed=42，梯度检查点开 |

- GRPO 启动命令、PID、日志：`train/grpo_v4.log`（首行 `[cfg]` 有完整配置快照）
- 当前进度（07-22 20:00）：s≈1100/1672（66%），u≈115 更新；probe mean_u：+0.82→+1.88→…→+1.78；KL≈0.31（监控中）；预计剩余 ~14h。

### 3.4 奖励消融状态

| 消融 | 状态 |
|---|---|
| binary reward（终局对错 0/1） | ⬜ TODO（需重训一组 GRPO，奖励换成 argmax 对/错） |
| s3-style reward | ⬜ TODO |
| state-relative ΔU（baseline 换成 current-state） | ⬜ TODO |
| 在线检索反馈消融（REWRITE 不重检索，u 恒 0 = v3 行为） | ✅ **已有天然对照数据**：v3 训练日志（REWRITE 恒 0 → 探针随机游走）vs v4（真实检索 → 探针 +0.96），可作为消融证据写入论文 |
| 门控消融（无 adv_std_floor / 无 min_spread） | ⬜ TODO（可选，优先级低） |

---

## 4. 数据与污染审计

### 4.1 训练数据

- 来源：自建中文消化内镜书籍语料 → `data_construction/` 管线生成图像接地 MCQ（API 生成 + API 校验两轮），题库 `mcq_image_v2_4000`，实际取 **3200 题**（`agent_context_v11_train3200_sep.jsonl`）。
- 题目与语料均来自中文医学书籍，语言中文；EndoBench 为英文公开基准。

### 4.2 划分

| 集合 | 用途 | 规模 |
|---|---|---|
| 3200 题 → SFT/RFT 轨迹 + GRPO 1672 状态 | 训练 | 如上 |
| GRPO 尾部 24 状态 | 训练内 held-out 探针（不进梯度） | 24 |
| SFT/RFT val | ckpt 选择（keep-F1） | 176 / 85 |
| **EndoBench test（6832）** | **仅测试** | 6832 |

### 4.3 EndoBench 污染审计（如实状态）

- ✅ QA pairs：训练集为独立生成的中文题，**未使用任何 EndoBench QA**（摘要已明确声明 "constructed without using EndoBench question-answer pairs for training"）。
- ✅ 图像来源：EndoBench 图像来自公开内镜数据集（Kid 等），我方语料为中文书籍扫描图，来源不同，预期无重叠。
- ⬜ **图像级 hash 去重：未做** —— TODO：对训练语料图像与 EndoBench 图像做感知哈希（pHash）近重复检测，报告排除数量（预期为 0，但需要数字支撑）。
- ⬜ 模板/近重复文本审计：训练题模板与 EndoBench 题型（器官识别/病灶分型等）存在任务类型相似（领域使然），无文本级重叠；TODO：n-gram 重叠率报告。
- ⬜ 数据/索引版本记录：Milvus db 文件路径已固定，但 **未记录文件 hash/版本号** —— TODO：记录 `multimodal_vector_indexes.db`、`multimodal_samples.db` 的 md5 + 生成日期，写进论文附录。

---

## 5. 实验结果现状

### 5.1 主结果表

| 行 | 状态 | 说明 |
|---|---|---|
| closed-book（无检索） | 🔄 脚本就绪待跑 | `eval/eval_endobench.py --mode baseline` |
| frozen RAG（12 候选全塞，无控制器） | 🔄 脚本就绪待跑 | `--mode vanilla_rag` |
| 专有模型控制器（GPT-4o，同 prompt 同动作空间） | 🔄 脚本就绪待跑 | `--mode gpt4o`，`GPTContextAgent` |
| **AgenticRL（本文）** | 🔄 脚本就绪待跑 | `--mode agentic --ctrl-model train/ckpt_grpo_v4` |
| rewrite-rerank pipeline | ⬜ TODO | mwx 旧管线有近似实现，需对齐检索栈 |
| selection-only（禁用 REWRITE） | ⬜ TODO（消融，ckpt 同一份，评测时强制 ACCEPT） |
| rewrite-only（禁用选择） | ⬜ TODO（消融，评测时强制 REWRITE） |

- 评测打分原语 = 训练奖励原语（`AnswerScorer.judge`，选项 argmax），零采样零解析误差。
- 聚合脚本 `eval/build_paper_table.py`：主表（overall+4 场景，含 LaTeX 行）、行为拆解、utility 对比、per-task 对比。
- 预计耗时：baseline ~3h / vanilla ~5h / agentic ~10h / gpt4o ~15h（6832 题全量）。

### 5.2 预算固定（✅）

- 检索预算：初始 12 候选（图6+文6），重检索 top-5，全方法统一
- generator 调用：每题 1 次前向（judge），所有方法一致
- 候选曝光：控制器最多见 12 候选 + 8 图，所有方法一致

### 5.3 统计严谨性

- 当前训练：单 seed（42）。⬜ 3-seed GRPO 复训：TODO（视时间决定，至少对关键消融做 2 seed）
- ⬜ bootstrap 95% CI：评测逐样本 JSONL 已落盘，TODO 写 CI 脚本（按 index 重采样即可）
- 训练内探针：24 状态 held-out，每 20 更新一次 greedy 评测（已实现并在产出）

---

## 6. 行为分析指标

> 注：FRR/FWRR/UERR/HESR/RFDR/RUG 这组缩写在当前代码库中**无既有定义**。以下按语义给出建议定义，并标注可计算性。

| 指标（建议定义） | 可从日志直接算？ | 数据来源 |
|---|---|---|
| **Rewrite Trigger Rate**（改写触发率） | ✅ | 评测 JSONL `action` 字段 |
| **FRR**（Failure Recovery Rate：初始检索失败→REWRITE 后答对的比例） | ✅ 需联合 baseline+agentic 两行 | baseline 错 & agentic REWRITE & 最终对 |
| **FWRR**（Failure-Wrong Rewrite Rate：REWRITE 后从对变错的比例，改写伤害率） | ✅ 同上 | baseline 对 & agentic REWRITE & 最终错 |
| **HESR**（Helpful Evidence Selection Rate：keep 的证据带来正 utility 的比例） | ✅ | 评测 JSONL 的 `logp_gold` − baseline 同行 `logp_gold` > 0 的占比（按 ACCEPT 样本） |
| **UERR**（Utility-Error Reduction Rate：相对 vanilla RAG 的负 utility 样本减少率） | ✅ | 三行联合 |
| **RFDR**（Rewrite False Discovery Rate：REWRITE 但未带来 utility 提升的比例） | ✅ | agentic 行内 REWRITE 子集 |
| **RUG**（Reward/Utility Gain：平均 ΔlogP vs baseline） | ✅ | 聚合脚本已实现（utility 对比表） |
| mean keep / 解析失败率 / accept vs rewrite 各自准确率 | ✅ | 聚合脚本已实现 |

**逐样本日志字段（✅ 评测时保存）**：index、scene/task/category/dataset、gold/pred/correct、logp_gold、4 选项概率、action、keep、rewrite_query、parse_ok、n_evidence、evidence doc 列表。
**训练侧逐 rollout 日志**：⬜ 当前 GRPO 日志只有组级聚合（u 的 min/max 区间），无逐 rollout (action, u) 对 —— TODO：下次训练加 `u_rw=` 字段（已列入计划）。

**成功恢复 / 失败案例可视化**：⬜ TODO——从评测 JSONL 筛选（baseline 错→agentic REWRITE 对）与（baseline 对→agentic REWRITE 错）各抽 5-10 例，保存 query 图、keep 证据图、rewrite_query、概率变化，做论文 Figure。

---

## 7. 人工证据审计

| 项 | 状态 |
|---|---|
| qrels / 人工 supportive labels | ⬜ 无 |
| 抽样标注计划 | ⬜ 建议方案已成型，未执行 |
| 双标注者 + κ/α | ⬜ 未执行 |

**建议执行方案（可直接落地）：**

1. 抽样：从 agentic 模式评测 JSONL 分层抽 250 题（ACCEPT 200 + REWRITE 50，覆盖 4 场景）
2. 标注界面：题目+query 图+控制器最终送入 generator 的证据（图+文），四分类标签：
   - `supportive`（证据足以支撑正确作答）
   - `related-only`（相关但不能支撑）
   - `conflicting`（误导/矛盾）
   - `wrong-organ`（部位错误）
3. 标注者：2 名（医学背景优先），各自独立标注；重叠 100 题计算 Cohen's κ（目标 κ≥0.7）
4. 产出：控制器证据质量的人工校准（vs 自动 utility），作为论文 Appendix 的证据质量审计表
5. 工时估算：250 题 × ~40s/题/人 ≈ 3h/人

---

## 8. 时间线（按当前节奏）

| 时间 | 事项 |
|---|---|
| 07-23 上午 | GRPO v4 训练完成（预计 ~14h 后），保存最终 ckpt |
| 07-23 下午 | 评测冒烟（每模式 64 题）→ 全量跑 baseline + vanilla_rag |
| 07-24 | agentic + gpt4o 全量；`build_paper_table.py` 出主表 |
| 07-25 | 消融（selection-only / rewrite-only 评测态；binary reward 重训若时间允许）；行为分析图；失败案例 Figure |
| 07-26 前 | 污染审计补数（pHash、db md5）；人工标注启动（可选）；全文数据回填 |
