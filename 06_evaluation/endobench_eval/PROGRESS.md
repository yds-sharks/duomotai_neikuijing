# EndoBench AgenticRL 评测进度总结

> 更新时间：2026-07-26（P0.3统计显著性完成，消融实验进行中）

## 1. 已完成的全量评测（6832 题）

| 模式 | 准确率 | 说明 |
|---|---|---|
| baseline（无检索） | 40.59% | 冻结生成器，纯选项 argmax |
| vanilla_rag（固定检索） | 38.85% | img6+txt6，无控制器 |
| agentic single-round（u50） | 40.12% | u50 控制器，单轮 keep/drop |
| agentic mr_full（u50，2轮） | 39.80% | u50 控制器，多轮 M+/M- |
| **agentic 3-round rescue（u50）** | **42.81%** | **u50 控制器，3轮 M+/M- + ParseFail回退** |

**显著性检验**（paired bootstrap CI + McNemar，6832题）：

| 对比 | 差值 | CI95 | McNemar χ² | p 值 | 显著？ |
|---|---|---|---|---|---|
| single vs vanilla | +1.27pp | [0.31, 2.25] | — | 0.012 | **是** |
| mr_full vs single | -0.32pp | [-0.80, 0.16] | — | 0.19 | 否 |
| single vs baseline | -0.47pp | [-1.13, 0.18] | — | 0.15 | 否 |
| **3-Round vs Baseline** | **+2.22pp** | **[+0.95, +3.54]** | **11.30** | **7.76e-04** | **是 ***** |
| **3-Round vs Vanilla RAG** | **+3.97pp** | **[+3.04, +4.87]** | **68.58** | **<0.000001** | **是 ***** |
| **3-Round vs Single-round** | **+2.69pp** | **[+1.95, +3.44]** | **48.96** | **2.61e-12** | **是 ***** |

**四象限计数**（3-Round vs Single-round）：

| 象限 | 数量 | 占比 |
|---|---|---|
| 维持正确 | 2491 | 90.9% |
| 被伤害 (H) | 250 | 9.1% |
| 被救回 (R) | 434 | 10.6% |
| 仍然错误 | 3657 | 89.4% |

**Per-category 显著性**（3-Round vs Single-round）：

| Category | Single | 3-Round | Diff | McNemar p |
|---|---|---|---|---|
| Anatomical Structure | 40.94% | 41.42% | +0.48pp | 0.568 |
| Lesion Analysis | 30.73% | 31.68% | +0.95pp | 0.255 |
| Spatial Localization | 42.85% | 46.33% | +3.48pp | 1.36e-07 *** |
| Surgical Workflow | 44.20% | 48.64% | +4.44pp | 2.71e-07 *** |

## 1.1 历史全量消融评测（含部位模块版本对比）

> 以下为早期全量评测结果，包含部位模块（Classification Module）3个训练版本及权重模块消融。

| Category | # Samples | Baseline | CM+Rag (v1) | CM+Rag (v2) | CM+Rag (v3) ★ | CM+Weight+Rag |
|---|---|---|---|---|---|---|
| Anatomical Structure Recognition | 1043 | 31.80% | 47.43% | 45.75% | 46.50% | 49.18% |
| Lesion Analysis and Grading | 1575 | 32.65% | 29.56% | 29.53% | 29.51% | 31.51% |
| Spatial Localization and Region Understanding | 2413 | 37.70% | 37.76% | 38.86% | 38.73% | 39.17% |
| Surgical Workflow and Operation Analysis | 1801 | 46.13% | 43.04% | 43.42% | 43.96% | 43.25% |
| **Overall** | **6832** | **37.91%** | **38.71%** | **38.93%** | **39.13%** | **39.98%** |

- **CM+Rag v1/v2/v3**：部位模块（Classification Module）的三组不同训练版本，架构相同，训练配置不同
- **★ v3 为最终采用版本**（Overall 39.13%）
- **CM+Weight+Rag**：在 v3 基础上加入权重模块（Weight Module），Overall 39.98%，为历史最优配置
- 此处 Baseline（37.91%）与第1节 EndoBench agentic 评测的 Baseline（40.59%）存在差异，系不同评测阶段/框架所致

## 2. M+（跨轮保留证据）失效根因

### 2.1 训练数据审计

| 数据集 | 样本数 | 多轮状态 | retained(M+) | REWRITE keep!=[] |
|---|---|---|---|---|
| SFT (sft_ctrl_train) | 3327 | 0 | **0** | 97.3% |
| GRPO (agent_context_v11) | 3200 | 0 | **0** | 96.4% |

**训练数据中 M+ 样本为零**，模型从未见过 `origin="retained"` 的候选。

### 2.2 评测行为偏差

| 行为 | SFT 教师 | u50 评测 |
|---|---|---|
| REWRITE 时 keep!=[] | 97.3% | **0%** |
| ACCEPT 时 keep!=[] | 100% | 100% |

模型在 EndoBench OOD 数据上 REWRITE 时 100% 输出 keep=[]（全丢），导致 M+ 永远为空。

### 2.3 origin 标签兼容性

| origin 标签 | SFT 训练出现次数 | 说明 |
|---|---|---|
| `(image)` | 38,776 (97.1%) | 模型熟悉 |
| `(text)` | 1,148 (2.9%) | 模型熟悉 |
| `(search_history)` | 11 (0.03%) | 几乎未见 |
| `(retained)` | **0** | **从未见过** |

**救援测试采用原始 `(image)`/`(text)` 标签注入 M+ 候选**，避免模型被陌生标签搞混。

## 3. GRPO v4 checkpoint 对比

### 3.1 GRPO v4 probe 曲线（24 held-out 状态 greedy utility）

| checkpoint | probe mean_u | bad_json | 保存？ |
|---|---|---|---|
| u0 (RFT v3) | +0.822 | 2 | - |
| **u20** | **+1.884** | **0** | 未保存 |
| u40 | +1.552 | 0 | 未保存 |
| u50 | ~+1.59 | ~1 | **已保存** |
| u60 | +1.631 | 2 | 未保存 |
| u80 | +1.746 | 3 | 未保存 |
| **u100** | **+1.780** | **2** | **已保存** |
| u120 | +1.551 | 4 | 未保存 |
| u140 | +0.959 | 7 | 未保存 |
| u150 | ~+0.9 | ~10 | 已保存 |
| u180 (final) | +1.529 | 15 | 已保存 |

- **u20 最优**（+1.884），但 save-every=50 未保存
- **u100 是已保存中最好的**（+1.780）
- u150/u180 bad_json 爆炸（>10/24），不可用

### 3.2 u50 选择理由

64 题小样评测：u50 准确率 64.06%（0 解析失败）> u100 62.50%（7 次解析失败）。
但 GRPO probe 上 u100 更高。**需要 200 题救援测试对比来最终选型。**

## 4. 三轮救援测试

### 4.1 设计

对 single-round 做错的题目，强制注入 M+/M- 后跑最多 3 轮：

- **ACCEPT+错**：M+ = 模型保留的证据（用原始 image/text 标签）
- **REWRITE+错**：M+ = 强制注入 top-3 原始候选 + M- = 失败检索词 breadcrumb
- 每轮：M+ + M- + 新检索 → 控制器 keep/drop → 累积最优证据
- 最终用第 3 轮优化后的证据给生成器

### 4.2 关键文件

| 文件 | 说明 |
|---|---|
| `rescue_3round.py` | 三轮救援评测脚本 |
| `wrong50_subset.jsonl` | 50 题错题子集（25 ACCEPT + 25 REWRITE） |
| `wrong200_subset.jsonl` | 200 题错题子集（100 ACCEPT + 100 REWRITE） |
| `rescue50_results.jsonl` | 50 题救援结果（u50） |
| `rescue200_u50.jsonl` | 200 题救援结果（u50，进行中） |
| `rescue200_u100.jsonl` | 200 题救援结果（u100，进行中） |

### 4.3 50 题 pilot 结果（u50）

| 指标 | 数值 |
|---|---|
| 总数 | 50 |
| **救回** | **11 (22.0%)** |
| ACCEPT+错救回 | 4/25 (16.0%) |
| REWRITE+错救回 | 7/25 (28.0%) |

**救回机制分析**：

| 路径 | 数量 | 机制 |
|---|---|---|
| 模型主动保留 M+ | 5/11 | 模型 keep 了 M+ 项 + 新检索 → 更好证据 |
| 累积兜底 | 6/11 | 模型 keep=[] 但 fallback 用了全部累积候选 → 生成器答对 |

### 4.4 200 题 u50 vs u100 对比测试（已完成）

**总体救回率**：

| 模型 | 救回 | 救回率 |
|---|---|---|
| u50 | 37/200 | 18.5% |
| u100 | 38/200 | 19.0% |

**Paired 对比**（McNemar）：

| 类别 | 数量 |
|---|---|
| 两者都救回 | 28 |
| 仅 u50 救回 | 9 |
| 仅 u100 救回 | 10 |
| 都未救回 | 153 |
| p 值 | 1.0000 |
| **差异** | **+0.5pp（不显著）** |

**按 fresh action 分层**：

| Action | u50 | u100 |
|---|---|---|
| ACCEPT | 15/85 (17.6%) | 18/97 (18.6%) |
| REWRITE | 22/115 (19.1%) | 20/103 (19.4%) |

**行为差异**：

| 指标 | u50 | u100 |
|---|---|---|
| REWRITE 比例 | 115/200 (57.5%) | 103/200 (51.5%) |
| 解析失败率 | 4/200 (2.0%) | 13/200 (6.5%) |
| 轮数分布 | 2轮=120, 3轮=80 | 2轮=134, 3轮=66 |

**并集救回**（两模型至少一个救回）：47/200 (23.5%)

**结论**：u50 与 u100 在救援任务上几乎等效（+0.5pp 不显著）。u100 GRPO probe 更高（+1.780 vs ~+1.59），但解析失败率也更高（6.5% vs 2.0%）。u50 更保守（REWRITE 更多），u100 更果断（ACCEPT 更多）。

## 5. 评测基础设施

### 5.1 脚本清单

| 脚本 | 用途 | 关键参数 |
|---|---|---|
| `eval_endobench.py` | 主评测（baseline/vanilla/agentic/gpt4o） | `--mode`, `--max-rounds`, `--no-mplus`, `--no-mminus` |
| `compare_multiround.py` | 多组对比 + paired bootstrap CI + McNemar | 输入多个 samples.jsonl |
| `rescue_3round.py` | 三轮救援测试（错题） | `--subset`, `--max-rounds 3`, `--milvus-db-path` |
| `run_single_resume.sh` | 恢复 single-round 全量（GPU 0+1） | CKPT 路径 |
| `run_mr_full_resume.sh` | 恢复 mr_full 全量（GPU 2+3） | CKPT + shard1 Milvus |

### 5.2 并行评测方案

Milvus Lite 本地文件锁：两个进程不能同时打开同一个 DB 文件。

解决方案：
- 进程 1：默认 DB `multimodal_vector_indexes.db`
- 进程 2：`--milvus-db-path multimodal_vector_indexes_shard1.db`

### 5.3 GPU 分配（4x A6000）

| 进程 | ctrl | gen | retr_text | retr_image |
|---|---|---|---|---|
| u50 rescue | cuda:0 | cuda:1 | cuda:1 | cuda:0 |
| u100 rescue | cuda:2→0 | cuda:3→1 | cuda:3→1 | cuda:2→0 |

注：CUDA_VISIBLE_DEVICES 重映射后，进程内设备编号为 cuda:0/cuda:1。

## 6. 全量 4091 题救援测试（u50，3轮）

### 6.1 设计

对 single-round 评测全部 4091 道错题（ACCEPT 1934 + REWRITE 2157）执行三轮救援。
2 路并行（GPU 0+1 / GPU 2+3），每路 ~2046 题。

### 6.2 总体结果

| 指标 | 数值 |
|---|---|
| 总错题数 | 4091 |
| **成功救回** | **434 (10.6%)** |
| 仍然错误 | 3657 (89.4%) |
| 错误/异常 | 0 |
| 解析失败(任一轮) | 149 (3.6%) |

### 6.3 按原始 Action 分层

| Action | 错题数 | 救回 | 救回率 |
|---|---|---|---|
| ACCEPT | 1934 | 92 | 4.8% |
| REWRITE | 2157 | 342 | 15.9% |

REWRITE 救回率显著高于 ACCEPT（15.9% vs 4.8%），说明强制注入 top-3 候选作为 M+ 对 REWRITE 场景更有效。

### 6.4 轮数分布

| 轮数 | 数量 | 占比 |
|---|---|---|
| 2轮（ACCEPT 提前终止） | 2523 | 61.7% |
| 3轮（跑满） | 1568 | 38.3% |

### 6.5 按场景分层

| 场景 | 错题数 | 救回 | 救回率 |
|---|---|---|---|
| Surgical Endoscopy | 1587 | 231 | 14.6% |
| Colonoscopy | 1040 | 111 | 10.7% |
| Gastroscopy | 339 | 25 | 7.4% |
| Capsule Endoscopy | 1125 | 67 | 6.0% |

### 6.6 准确率影响

| 指标 | 数值 |
|---|---|
| 原始 single-round 准确率 | 2741/6832 = 40.12% |
| 救援后准确率 | 3175/6832 = **46.47%** |
| **提升** | **+6.35pp** |

### 6.7 与 200 题 pilot 对比

| 测试 | 题数 | 救回率 | 说明 |
|---|---|---|---|
| 200题 pilot | 200 | 18.5% | 偏向易救回题目 |
| 全量 4091题 | 4091 | 10.6% | 全量更真实 |

200 题 pilot 高估了救回率（18.5% vs 10.6%），全量结果更可靠。

### 6.8 代码修正（ParseFail 确定性回退）

`rescue_3round.py` 已修改，关键变更：

| 变更 | 说明 |
|---|---|
| ParseFail 回退 | 解析失败时 `continue` 而非 `break`；全部失败则回退到原始单轮预测 |
| 消融参数 | 新增 `--no-mplus`、`--no-mminus`、`--mplus-topk` CLI 参数 |
| 逐轮日志 | 新增 `query`、`latency_s`、`keep_docs`、`fallback_to_orig` 字段 |
| 14个案例修补 | `rescue_full_u50.jsonl` 中 ParseFail 受影响案例已修正（436→434） |

### 6.9 关键文件

| 文件 | 说明 |
|---|---|
| `wrong_full_subset.jsonl` | 全量 4091 道错题 |
| `wrong_full_shard0.jsonl` | shard0 (2045题) |
| `wrong_full_shard1.jsonl` | shard1 (2046题) |
| `rescue_full_u50_shard0.jsonl` | shard0 救援结果 |
| `rescue_full_u50_shard1.jsonl` | shard1 救援结果 |
| `rescue_full_u50.jsonl` | 合并后全量结果 |
| `run_rescue_full_u50.sh` | 并行救援脚本 |
| `logs/rescue_full_u50_shard0.log` | shard0 日志 |
| `logs/rescue_full_u50_shard1.log` | shard1 日志 |

## 7. 2741题伤害测试（u50，3轮，已完成）

### 7.1 设计

对 single-round 评测全部 2741 道正确题执行相同的三轮救援机制，测量伤害数 H。
真实准确率 = (2741 - H + 434) / 6832。

### 7.2 最终四象限（全量6832题）

| 象限 | 数量 | 占比 | 说明 |
|---|---|---|---|
| 维持正确 (TN) | 2491 / 2741 | 90.9% | 原本正确，救援后仍正确 |
| **被伤害 (H)** | **250 / 2741** | **9.1%** | 原本正确，救援后变错 |
| **被救回 (R)** | **434 / 4091** | **10.6%** | 原本错误，救援后变对 |
| 仍然错误 | 3657 / 4091 | 89.4% | 原本错误，救援后仍错 |

### 7.3 真实准确率

| 指标 | 数值 |
|---|---|
| 原始准确率 | 2741/6832 = 40.12% |
| **真实准确率** | **(2491+434)/6832 = 42.81%** |
| **净提升** | **+2.69pp** |

### 7.4 统计检验

| 检验 | 结果 | 显著？ |
|---|---|---|
| McNemar | χ²=48.96, p<0.000001 | **是** (α=0.05) |
| Paired Bootstrap CI | +2.70pp, 95% CI [1.95, 3.43] | **是** (CI不含0) |

### 7.5 伤害率 by orig_action

| Action | 总量 | 被伤害 | 伤害率 |
|---|---|---|---|
| ACCEPT | 1336 | 113 | 8.5% |
| REWRITE | 1405 | 137 | 9.8% |

REWRITE 伤害率略高于 ACCEPT（9.8% vs 8.5%），与救回率趋势一致——REWRITE 场景下证据变动更剧烈，双向风险均更高。

### 7.6 救回率 by orig_action（对照）

| Action | 错题数 | 救回 | 救回率 |
|---|---|---|---|
| ACCEPT | 1934 | 92 | 4.8% |
| REWRITE | 2157 | 342 | 15.9% |

### 7.7 伤害率 by scene

| 场景 | 总量 | 被伤害 | 伤害率 |
|---|---|---|---|
| Colonoscopy | 808 | 92 | 11.4% |
| Capsule Endoscopy | 553 | 56 | 10.1% |
| Surgical Endoscopy | 1136 | 88 | 7.7% |
| Gastroscopy | 244 | 14 | 5.7% |

### 7.8 Fallback 统计

| 测试 | Fallback 数 | 占比 |
|---|---|---|
| 伤害测试 (2741题) | 840 | 30.6% |
| 救援测试 (4091题) | 14 | 0.3% |

伤害测试中 30.6% 的题目因 ParseFail 或无有效证据回退到原始预测，这些题目不会被伤害（保持原始正确）。

### 7.9 关键文件

| 文件 | 说明 |
|---|---|
| `correct_full_shard0.jsonl` | shard0 (1370题) |
| `correct_full_shard1.jsonl` | shard1 (1371题) |
| `rescue_harm_u50_shard0.jsonl` | shard0 伤害测试结果 |
| `rescue_harm_u50_shard1.jsonl` | shard1 伤害测试结果 |
| `run_harm_test_u50.sh` | 并行伤害测试脚本（含自动汇总） |

## 8. 后续计划（论文实验清单）

### P0（必须完成）
- [x] baseline/vanilla/single/mr_full 全量公平对比
- [x] paired bootstrap CI + McNemar 显著性检验（单轮对比）
- [x] u50 vs u100 救援测试对比（18.5% vs 19.0%，不显著）
- [x] 扩大救援测试到全量 4091 道错题（10.6% 救回率，+6.35pp）
- [x] ParseFail 确定性回退修复（14个案例修补，436→434）
- [x] 两套 Milvus DB 一致性验证（文件大小、实体数完全相同）
- [x] 2741题伤害测试（H=250, R=434, 真实准确率42.81%, +2.69pp）
- [x] 四象限分析 + McNemar检验 + paired bootstrap CI（p<0.000001, CI[1.95,3.43]）
- [x] **P0.1 公平重跑消融Full配置（200题，更新ParseFail）** — `run_fair_full_200.sh`（已完成）
  - 公平 Full: 11/200 (5.5%) vs 旧基线: 37/200 (18.5%)
  - 旧版 ParseFail 夸大基线 3.4 倍，公平版现在与其余消融变体可比
  - Fallback: 52/200 (26.0%)，ParseFail: 10/200 (5.0%)
  - ACCEPT: 2/94 (2.1%), REWRITE: 9/106 (8.5%)
- [ ] **P0.2 前端消融（4组×1002题）** — `run_frontend_ablation_1000.sh`
  - Full front-end / w/o modality mixer / w/o anatomical router / w/o both
  - 全部使用3轮完整记忆配置，仅变前端模块
- [x] **P0.3 统计显著性（6832题全量）** — `run_stat_significance.py`
  - 3-Round vs Baseline: +2.22pp, CI[+0.95,+3.54], p=7.76e-04 ***
  - 3-Round vs Vanilla RAG: +3.97pp, CI[+3.04,+4.87], p≈0 ***
  - 3-Round vs Single-round: +2.69pp, CI[+1.95,+3.44], p=2.61e-12 ***

### P1（强烈建议）
- [ ] **P1.4 记忆消融扩展到含正确题集合（1002题）** — `run_memory_ablation_1000.sh`
  - Full memory / no-memory / M+ only / M- only
  - 报告 Accuracy、Correct→Wrong、Wrong→Correct、平均轮数、ParseFail
- [ ] **P1.5 轮数与成本（1002题）** — `run_memory_ablation_1000.sh`（合并批次2）
  - 2-round 配置，比较 Accuracy、平均轮数、检索/控制器调用次数、耗时
- [ ] **P1.6 Qwen2.5-VL-7B 生成器迁移（513题）** — `run_qwen25vl_migration.sh`
  - Closed-book / Vanilla / Single-round / 3-Round，复用保存的检索轨迹
  - 扩展门槛：3-Round vs Closed-book ≥ +2pp；3-Round vs Vanilla ≥ +1pp

### P2（有预算再做）
- [ ] 三阶段训练消融（SFT / SFT+RFT / SFT+RFT+GRPO）
- [ ] 动作机制分析（Selection-only / Rewrite-only / 联合）

### 已完成的消融实验
- [x] 200题记忆消融（6变体）— `run_ablation_200.sh`（旧版ParseFail，仅用于筛选）

### 论文最终保留
- 表1：主结果递进（Baseline → Vanilla → Single → 3-Round）
- 表2：公平版本的记忆消融（Full / no-mem / M+ only / M- only / 2-round）
- 图3：EndoBench模型性能与规模
- 图4：单轮到三轮的配对变化及动作条件纠正率
- 图5：Category、Scene、Task临床分析
- 正文一段效率与统计显著性
