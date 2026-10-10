# 内窥镜 Agentic 项目 — Git 上传与更新日志

> 本文件记录本仓库每一轮上传到 Git 的完整记录，以及每轮的更新与优化内容。
> **记录规范**：每完成一轮开发/优化，在「更新记录」最上方追加一节；Git 提交推送后回填提交哈希。

## 仓库信息
- 远程：https://github.com/yds-sharks/duomotai_neikuijing.git
- 分支：main
- 本地路径：`/mnt/data_1/yds/多模态/内窥镜agentic`
- 定位：医学多模态 RAG 全项目归档 + Agentic RAG 论文迭代主线仓库

---

## 更新记录

### Round 2 — 2026-10-09：论文重构启动：基线盘点 + 本日志建立（进行中）
**本轮目标**：论文全面调整前的三项准备——(1) 基座模型确认与 Qwen3.5 系列切换；(2) 训练数据构造链路梳理与重训；(3) 中央 Agent Harness（RAG 作为工具模块）搭建。

**已完成**：
- [x] 建立本日志文件（GIT_LOG.md）
- [x] 基座模型盘点（结论见「基线状态快照」）
- [x] 训练数据构造链路盘点（结论见「基线状态快照」）
- [x] 决策确定：全链路统一 **Qwen3.5-4B**（控制器 = 生成器 = 图像编码视觉塔，复用本地已有权重 `851bf6e8`，无需新下载）；文本检索保持 BGE-M3
- [x] README 重构：由「四阶段流水线」改为「真实 Agent 场景」架构（中央 Agent 大脑 + RAG 工具模块 + answer-utility 奖励），新增「模型资产与复现指南」节（记录 HF 下载来源与本地路径，保证跨机器复现）；门控与重排序降级为消融基线
- [x] 规划确认：后续通过 **OPD（Online Policy Distillation）** 内化模型学到的知识
- [x] **中央 Agent Harness v0.1 搭建完成**（`10_harness/`，20 个文件）：
  - 工具层 `tools/`：`text_retrieve`（BGE-M3）/ `image_retrieve`（视觉塔）/ `keep_evidence`（keep/drop）/ `submit_answer`（ACCEPT），统一 JSON 工具调用协议
  - 大脑 `brain/agent_brain.py`：观察→决策→行动多轮循环，无效调用消耗预算可恢复，最终触发生成器作答并计算 answer-utility `u_set`
  - 后端层 `backends/`：检索后端动态加载 v0.5 `FirstStageRetriever`（候选字段与训练链路一致）；大脑/生成器走 OpenAI 兼容 API（vLLM serve Qwen3.5-4B）；全部可注入 Mock
  - 轨迹 `runtime/`：保留 v0.5 训练链路字段（`obs_candidates` 等）+ 新增 `rounds[]` 逐步记录；`run_harness.py` 批量出轨迹 jsonl + summary
  - 设计要点：v0.5 的 ACCEPT/REWRITE 不再显式存在——REWRITE 退化为 agent 自主换 query 再检索，ACCEPT 即 submit；冒烟 `run_smoke.py` 全绿（工具分发/预算/跨轮去重/无效恢复/奖励）

- [x] **工具粒度反转（用户反馈）**：完整 RAG 封装为**单一工具** `rag_search`（输入 query → 直接输出筛选后文段），原四工具细粒度设计废弃；v0.5 的证据选择 agent 下沉为 RAG 工具内部处理（`AgentEvidenceFilter`，失败回落 TopK）；大脑行为空间精简为 `rag_search` + `submit_answer` 两个动作，只负责查询规划；仓库根 README 与 10_harness README 架构图同步更新

- [x] **Harness v0.2 真实链路冒烟通过**（EndoBench 真实题 ×10，任务：Organ Identification）：全链路 0 错误、10/10 正常提交、平均 2.4 轮/题、平均 5.4 条证据。关键落地：
  - `make_queries.py`：从 HF 缓存导出真实题（qid=eb_N、选项、答案、图像路径、中文检索提示）；
  - `backends/transformers_backend.py`：进程内 transformers 推理（vLLM 0.11 不支持 Qwen3_5 架构，原生与 transformers 0.11 后端均不可用），与训练同款 `AutoModelForImageTextToText` 加载，大脑/证据筛选/生成器共享单卡实例（cuda:0）；检索双模型 BGE-M3 cuda:1 / Qwen3-VL-8B cuda:2（图像索引仍为 8B 建，重建前不得换）；
  - 修复：FirstStageRetriever 需完整 `{"paths","retrieval"}` 配置结构；LLM 输出剥 `<think>` + `chat_template_kwargs(enable_thinking=false)`；观察中注入中文检索提示（题面英文/语料中文）。
  - **已知短板（v0.2.1 待办）**：生成器 prompt 为纯文本，未输入查询图像——Organ Identification 这类看图题全靠文段描述，导致正确率仅 1/10；下一步给生成器/大脑加图像输入。

- [x] **Harness v0.2.1 图像输入修复，正确率 1/10 → 3/10**：生成器与大脑均接入查询图像（同一写法对齐 v0.5 生成器：`{"type":"image"}` 占位 + `processor(text, images=...)`）；验证大脑确实在看图（thought 开始描述内镜图像内容，v0.2 做不到）。逐题对比：eb_3/eb_4/eb_8 由错变对，eb_0 退步（4B 看图判别力边界，v0.2 系纯文本撞对）。落地：
  - `transformers_backend.py`：`chat/decide/generate/answer_utility` 全部增加 `image_path` 参数，最后一条 user 消息转多模态 content；
  - `reward_backend.py`/`llm_backend.py`：Protocol 与全部实现同步签名；OpenAI 后端走 base64 data URI 视觉格式（vLLM serve 后直接可用）；
  - `brain_prompt.py`：系统提示明示图像附在每条消息中；首轮反馈文案改为显式 "FIRST ROUND"（修复大脑误读为“上次检索无结果”）。
  - **新暴露问题（v0.2.2 待办）**：末轮大脑倾向继续搜索而非提交（2/10 budget_exhausted，虽 generator 兑底答案仍计分且均答对）——末轮提示强制提交可修复。

- [x] **Harness v0.2.2 中文 query + 末轮强制提交**：检索语料为中文而大脑生成英文 query，召回质量受限；改为 POLICY 强制中文 query（基于 RETRIEVAL HINT + 图像所见改写）。效果：单轮新增证据从常为 +1 提升至 +4/+5（召回明显改善）；submit_rate 0.8 → 1.0（末轮 FINAL ROUND 提示生效，无 budget_exhausted）；正确率 3/10 持平（±2 题波动，10 题样本噪声范围内；新对 eb_1/eb_5，翻错 eb_4/eb_8）。**瓶颈转移结论**：检索侧已收敛（每题稳定积累 4-5 条证据），当前瓶颈在 4B 模型对 Organ Identification 看图题的视觉判别力——需扩大样本（v0.3）或回主线训练。落地：`brain_prompt.py` POLICY 中文要求 + RETRIEVAL HINT 文案强化 + `final_round` 参数；`agent_brain.py` 传入末轮标志。

- [x] **v0.3 RAG 内部 agent 训练方案定稿**（`07_experiments/v0.3_rag_agent_training/RAG内部agent训练方案与数据构造_v0.3.md`）。旧数据诊断：(1) **图像自命中泄露**——题图本身在库中，图像路召回题图及写着答案的说明文字（train3200 中 3200/3200 题候选含题图，2143 题 top1 score>0.999），agent 学到的是捷径，EndoBench 外部图像上失效；(2) 候选全部来自图像路、文本路 0 条，改写无学习信号（REWRITE 仅 4.2%）；(3) 实际题库为模板生成的 `mcq_image_v2_4000`（未验证，2 种题型），`qa_gold_4000` 成品在 outputs 下未找到。新方案要点：
  - 职责：大脑输出疑惑点 `info_need`，RAG 内部 agent 负责改写检索式 + 正负筛选 + 工具内多轮（K_inner=3）
  - **恢复并升级检索记忆 M+/M-**（v0.5 `trajectory_runtime.py` 机制）：会话级共享，M+ 与证据集统一；agent 显式写入好文段及理由、错误改写方向及失败原因，程序校验
  - 数据构造 7 步：留一 + 同书排除检索 → LLM 出题 + 盲答验证 → 合成 info_need → best-of-8 改写 → 4B 生成器单文段 utility 正负标注 → 失败/成功多轮分支 → T1–T4 四类样本
  - 训练：**废弃 DPO，SFT → GRPO**；GRPO 奖励 = u(M+_final) + keep 精度 + M- 准确率 − 重复失败方向 − 检索次数

- [x] **v0.3 完整设计总览定稿**（`07_experiments/v0.3_rag_agent_training/00_完整设计总览_v0.3.md`），补齐大脑与自进化设计：
  - 大脑两种模式：答题模式（提出疑惑点 info_need + 判断提交）/ 复盘模式（分析器：分析完整轨迹的冗余与可优化步骤，写成技能）
  - 三层经验沉淀：单题内检索记忆 M+/M- → 跨题技能库 → OPD 内化进模型参数
  - 分析器读程序标注后的轨迹（每步新增正类、边际增益、截断回放损失、重复度、成本），冗余/过度检索/过早停止等有客观判定规则
  - 技能生命周期：候选 → A/B 验证 → 生效（注入提示词）→ OPD 内化（老师带技能、学生不带，在线逐 token 反向 KL）→ 归档；未验证技能不得内化
  - 自进化循环：跑题 → 标注 → 复盘 → 验证 → 内化 → 回归；EndoBench 严格不参与技能挖掘

**待办**：
- [ ] 执行生成器切换：修改 eval/grpo 的生成器配置指向 Qwen3.5-4B；如图像编码同步切换，重建图像向量索引
- [ ] 按 v0.3 方案重构训练数据并重训（SFT → GRPO，不做 DPO；先做 P0 检索环境修正 + P1 支持度抽样）
- [ ] Harness v0.3：EndoBench 批量评测（50-100 题，全量中文翻译缓存 6832 条已就绪）+ 与 v0.5 pipeline 对照
- [ ] OPD 方法实现（按 v0.3 总览第 5 节：技能库 + A/B 验证 + 在线蒸馏内化 + 自进化循环）

**提交**：`0b51d3c` 建立日志；`95f6724` 回填哈希；`7911f58` 架构重构+统一Qwen3.5-4B+模型复现指南；`1fa91d3` 中央 Agent Harness v0.1；`db6f56b` 工具粒度反转：单一 rag_search；`bde0f7c` v0.2 真实链路冒烟；`22b2277` v0.2.1 图像输入修复；`3e48993` v0.2.2 中文query+末轮提交；`e217f1a` 提示词中文化+EndoBench引用论文清单；`26301a4` v0.3 RAG内部agent训练方案定稿（均已推送）

---

### Round 1 — 2026-10-08：项目全量归档与首次上传 ✅
| 提交 | 说明 |
|---|---|
| `1d4cb64` | init: 医学多模态RAG项目代码整理归档（625 文件 / 52M 入库，main 分支） |
| `eb2d6d3` | docs: README 补充本归档仓库远程地址 |

**内容**：工作区全部代码按流水线阶段分类为 `01_data_processing` ~ `09_legacy_deprecated` + `third_party`；数据/权重/实验结果通过 `.gitignore` 本地保留不入库；原工作目录只读未动；大数据（1.1T 语料、202G ckpt、23G milvus、53G 发布资产等）留在原位，对照表见 README。

**插曲记录**：首版 .gitignore 误纳入 4 个 594M 的 pkl 与 legacy A 树自带 `.git`（gitlink）；已用 `git update-index --force-remove` 移出索引并 `git gc --prune=now` 清理对象库（.git 604M → 19M），fsck 无错误。

---

## 基线状态快照（2026-10-09 盘点）

### 1. 基座模型现状
| 角色 | 模型 | 位置/快照 | 状态 |
|---|---|---|---|
| **Agent 控制器**（策略，被训练对象） | **Qwen3.5-4B**（原生多模态，AutoModelForImageTextToText） | `/mnt/data_1/yds/models/hf_hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e8...` | 全参 FSDP SFT → DPO（起点 `ckpt_qwen35_ctrl_full_v1`）→ GRPO；FSDP wrap：`Qwen3_5DecoderLayer, Qwen3_5VisionBlock`；chat template 需 `enable_thinking=False` |
| **答案生成器**（冻结，reward 定义来源） | Qwen3-VL-8B-Instruct | `/mnt/data_10/mwx/huggingface_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd0...` | 冻结；answer-utility `u(keep)=1[pred==gold]` 由其 logprobs 定义 |
| 文本检索编码 | BGE-M3 | `.../BAAI/bge-m3` | 冻结 |
| 图像检索编码 | Qwen3-VL-8B-Instruct（视觉塔） | 同生成器 | 冻结 |
| 重排序 embedding | Qwen3-VL-Embedding-2B | `/mnt/data_1/yds/models/Qwen/Qwen3-VL-Embedding-2B` | 冻结 |

**结论**：控制器（策略）已经是 Qwen3.5-4B；**尚未切换到 Qwen3.5 的是生成器**（现为 Qwen3-VL-8B-Instruct），以及可选的图像编码/重排序模型。Qwen3.5 系列切换方案需确定：目标型号与尺寸、原生多模态接口（`AutoModelForImageTextToText`）、reward 重算方式。

### 2. 训练数据构造链路现状
```
题目/环境池（05_agentic_rag/agentic/data_construction/）
  multimodal_samples.db 图文样本库
    → 模板 MCQ 候选（build_agentic_mcq_dataset.py / build_agentic_image_mcq_dataset.py）
    → schema + 泄漏校验（validate_agentic_mcq_dataset.py）
    → gpt-5.4 API 校验过滤（verify_*_with_api.py）
    → qa_gold_4000（4 类题型：病灶识别 / 操作识别 / 解剖部位 / 空间区域理解）
  ⚠ EndoBench benchmark 样本全程 held-out，不作训练

策略训练数据（05_agentic_rag/agentic/train/）
  GPT 教师 agent 跑 Stage-2 轨迹（gpt_agent_adapter.py + rollouts_*.jsonl）
  ├─ SFT  build_sft_from_trajectory.py：每轮轨迹 → 行为克隆样本
  │       （system v11 提示 + 编号候选块 + query 图像 + 证据图像 → 教师 keep/drop、ACCEPT/REWRITE、rewrite_query JSON）
  ├─ RFT  build_rft_dataset.py：Stage-D 奖励分层（rewrite-win / selection / trivial）按 u_set 选最优成员为目标
  ├─ DPO  build_dpo_pairs.py：同状态候选 keep-action 按 u(keep)=1[pred==gold] 排序构成偏好对（teacher vs 更差）
  └─ GRPO grpo_reward.py：在线 rollout + 生成器 logprobs 的 answer-utility 奖励
```

**重训注意**：若生成器切换为 Qwen3.5 系列，reward 定义（u_set / u(keep)）将整体变化，SFT/RFT/DPO 数据需用新生成器重算；qa_gold_4000 环境池可复用。

---

## 记录模板（复制到「更新记录」最上方使用）
```markdown
### Round N — YYYY-MM-DD：主题
**本轮目标**：
**完成**：
**改动文件**：
**提交**：`hash`（已推送）
**备注/遗留问题**：
```
