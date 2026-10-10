# P2a EndoBench 分题型盲答基线报告（v0.3）

> 状态：**已完成**（2026-10-10），480 题（12 task × 40 分层抽样）0 错误，207 秒
> 动机：P2 出题前核实"训练题 vs benchmark"的难度与题型差距，避免出题过简或题型错位导致训练收益无法迁移
> 脚本：`data_construction/p2a_endobench_baseline.py`；产物：`data_construction/p2a_baseline_out/`

---

## 1. 总体结果：用户担心的"过简"不成立，但题型错位是实锤

| 口径 | 4B 看图盲答正确率 |
|---|---|
| EndoBench 均匀分层抽样（480 题） | **38.75%**（186/480） |
| **EndoBench 全量 task 分布加权** | **36.2%** |
| 旧模板题（mcq_image_v2_4000，P1 同口径） | 41%（organ 40% / content_type 42%） |

难度本身接近（41% vs 36–39%），**旧模板题不是"过于简单"，而是题型分布与 EndoBench 完全错位**：

- 旧训练题 100% 是 organ/content_type 两类，而它们在 EndoBench 里只占 **15.3%**（Organ + Landmark Identification，1043 题）；
- EndoBench 的大头是 Spatial Localization（35.3%）、Surgical Workflow（26.4%）、**Lesion Analysis（23.1%）**——旧训练题完全没有覆盖病灶分级/定量。

## 2. 分 task 正确率（按难度升序）

| task | 4B 盲答 | logp_gold 中位 | 全量占比 |
|---|---|---|---|
| Lesion Quantification（病灶定量） | **17.5%** | −2.57 | 4.7% |
| Lesion Severity Grading（病灶分级） | **17.5%** | −2.59 | 2.8% |
| Macro Phases Identification（手术阶段） | 27.5% | −1.36 | 10.8% |
| Organ Identification（器官识别） | 27.5% | −1.80 | 6.1% |
| Lesion Type Identification（病灶类型） | 30.0% | −1.81 | 15.5% |
| Visual Grounding（视觉定位） | 30.0% | −1.30 | 9.8% |
| Region Recognition（区域识别） | 32.5% | −1.20 | 19.2% |
| Landmark Identification（解剖标志） | 47.5% | −0.95 | 9.2% |
| Micro Operation Analysis（微观操作） | 47.5% | −0.90 | 9.8% |
| Region Selection（区域选择） | 57.5% | −0.43 | 6.3% |
| Instrument Management（器械管理） | 65.0% | −0.56 | 4.3% |
| Preoperative Assessment（术前评估） | 65.0% | −0.43 | 1.5% |

分 category：Lesion Analysis and Grading **21.7%**（最弱）< Anatomical Structure 37.5% < Spatial Localization 40.0% < Surgical Workflow 51.2%。

选项数效应：2 选项 66.7% / 4 选项 39.6% / 5 选项 20.0% / **6 选项 17.5%**（EndoBench 有 767 道 5–6 选项题，占 11.2%）。

## 3. 对 P2 出题方向与难度的调整建议

**核心判断：把训练权重往"4B 最弱且语料最有支持"的题型打。** 病灶分级/定量题最弱（17.5%）且恰好依赖"诊断标准、分级表格"类知识——这正是 PDF 文献语料最丰富、检索增益潜力最大的部分。

题型配额调整（v0.3 计划 → 建议）：

| 训练题型 | 原配额 | **建议配额** | 依据 |
|---|---|---|---|
| 病灶识别+严重程度分级+定量 | 30% | **35%** | 4B 最弱（17.5–30%）；EndoBench 占 23%；分级/定量知识语料支持最强 |
| 手术流程与操作（阶段识别/操作分析） | 20% | **25%** | EndoBench 占 26%；4B 中等（27–47%）；阶段/操作知识可检索 |
| 空间定位（区域识别/选择/视觉定位） | 15% | **25%** | EndoBench 最大头（35%）；Region Recognition 32.5% 值得覆盖 |
| 解剖部位/标志识别 | 35% | **15%** | 4B 相对不弱（27–47%）；旧模板题已大量覆盖同质能力；EndoBench 仅占 15% |

难度工程（并入三层验证协议执行）：

1. **L2 看图盲答 logP<−0.5 的阈值有依据了**：EndoBench 易题（65% 类）logp_gold 中位 −0.43，难题 −2.6；旧模板题 −1.1～−1.3 居中。阈值 −0.5 恰好把"模型有把握"的题滤掉；
2. 出题按 logP 分层保留难度梯度（−0.5 ~ −3 各档配额），避免全变超难题导致 GRPO 组内全错；
3. **选项数混合**：~85% 四选项 + ~15% 五/六选项，对齐 EndoBench 的 11.2% 多选项题；分级题干扰项用同病灶的其他级别；
4. 分级/定量题的干扰项设计在 GPT-5.6 出题 prompt 中显式约束（同病灶异级别、相邻 T/N 分期等），防止选项间语义距离过远被排除法猜中。

## 4. 风险与后续

- 病灶分级/定量题的 oracle 支持（检索能否搜回诊断标准文段）尚未验证——P1 只对 organ/content_type 验证过；P2 出题后用 L3（oracle 文段可解）把关即可，若该类题 L3 通过率低，说明语料分级标准文段不足，需调整出题来源（倾向含"诊断/分级/标准"关键词的源样本）；
- 评测口径提醒：本基线为 argmax 一次前向口径，与 harness 生成式作答存在小差异（v0.2.x smoke 生成式 3/10 vs 本次 Organ Identification 27.5%，量级一致）；
- 出题教师模型 GPT-5.6（endpoint/key 走本地 `api_config.local.json`，已 gitignore，不入库）。

## 5. 产物

- 明细：`data_construction/p2a_baseline_out/p2a_baseline_samples.jsonl`（480 条：qid/task/pred/gold/correct/logp_gold）
- 汇总：`data_construction/p2a_baseline_out/p2a_baseline_report.json`（含 by_task / by_category / by_scene / by_n_options）
- 全量题库导出：`10_harness/runs/queries_endobench_full.jsonl`（6832 题，含 task/category/scene 字段）
