# MedAlign-RAG 操作与设计总纲（含断电丢失分析与重建方案）

> 生成时间：2026-07-18
> 目的：记录当前真实磁盘状态、系统架构设计、断电导致的丢失清单，以及完整重建方案。
> 权威架构来源：`medalign_paper_framework_zh.md`、`WORK_SUMMARY.md`、`CHANGES_LOG.md`。

---

## 0. 一句话现状

2026-07-18 服务器断电，导致 7/17–7/18 期间新写但未落盘的**全部训练产物与脚本丢失**（ext4 延迟分配 + 正在写盘的训练进程）。工作区现等价于 git 提交 `6d85732`（本会话之前的代码库快照）。**幸存**：全部架构代码、Stage1 benchmark 数据、设计文档。**丢失且不可找回**：GPT 轨迹、SFT 数据、训练脚本、checkpoint。

---

## 1. 断电丢失分析（forensics）

### 1.1 根因
- **原因：断电（unclean shutdown），非 git、非人为误删。**
- 机制：ext4 默认 `data=ordered` + delalloc。文件元数据每 ~5s 入日志，数据块可在 page cache 滞留更久。断电时 7/17–7/18 新写文件的数据尚未落盘；正在写盘的训练进程（全参 SFT）必然未 fsync。
- 重启日志重放后：
  - 元数据已入日志、数据未落盘 → **0 字节空壳**：`train/train_ctrl_sft_full.py`、`train/fsdp_qwen35.yaml`（典型 delalloc 指纹）。
  - 目录项未入日志 → **整条消失**：其余所有 train/ 数据、轨迹、checkpoint、脚本、日志。
- 7/16 及更早文件早已落盘 → 完好。

### 1.2 git 操作与丢失无关
- reflog：`... ed54213(commit) → reset 到 8f431a3 → 6d85732(commit)`。你做的 reset+commit 只影响**已跟踪**的代码库；我们的产物是**未跟踪文件**，`git reset` 不删未跟踪文件。

### 1.3 恢复途径（全部无效）
| 途径 | 结果 |
|---|---|
| git 已提交历史 | ❌ 从未提交（`6d85732`/`ed54213` 均 76 个旧文件） |
| git 悬空对象 | ❌ 唯一悬空 `ed54213` 仅代码库同步，无数据 |
| ext4 `lost+found` | ❌ 空（fsck 未抢救出 inode） |
| 磁盘孤立大文件 | ❌ 近两日 >50MB 残留：无 |

**结论：不可找回，必须重建。**

---

## 2. 系统架构设计（v0.3，权威，已幸存）

### 2.1 数据流
```
query(+图) → 冻结检索(BGE-M3文本 + Qwen3-VL图像, Milvus)
           → 可训练控制器 Controller(keep/drop + ACCEPT/REWRITE)
           → [REWRITE 则用 rewrite_query 重检索，召回层抑制 dropped/已见，最多 max_rounds 轮]
           → 冻结生成器 Qwen3-VL-8B 产答案
```
- **无独立 reranker**：检索结果直接给控制器做"价值判断"（keep/drop），而非相关性排序。
- **控制器一次输出 schema**：
```json
{"keep":[0,2,4], "drop":[1,3], "action":"ACCEPT|REWRITE", "rewrite_query":"...", "reason":"..."}
```

### 2.2 奖励（answer-utility，护城河）
```
r(a) = P_G(a* | q, I_q, E_a) − P_G(a* | q, I_q, ∅)
```
- 用冻结生成器 logprob 测"证据是否抬高正确答案概率"，而非 evidence_hit。
- 避免退化成"纯检索 query 改写器"，保持医学特异性。
- GRPO 组内相对优势 A=(r−mean)/std；轻量约束：leakage/unnecessary_rewrite/length penalty。

### 2.3 训练两阶段
- **SFT 冷启**：逐轮行为克隆（模仿 GPT-agent 的 keep/drop + ACCEPT/REWRITE + rewrite_query）。
- **GRPO**：用 answer-utility 奖励在线优化，突破 SFT 天花板。

### 2.4 面包屑（search_history memory）设计【本会话新增，需重建】
- 触发：某个 REWRITE 轮 `keep=[]`（什么都没留）。
- 动作：把该轮检索 query 作为一条合成 `search_history` 记忆项，注入**下一轮**候选，并**强制放进 keep**（索引通过 n_bc 偏移重映射），给下一轮改写方向信号。
- 规模（丢失前）：3988 条 SFT 样本中 84 条含面包屑，面包屑固定出现在候选 [0]。

---

## 3. 磁盘现状清单（幸存 vs 丢失）

### 3.1 幸存（在 HEAD `6d85732`，无需重建）
| 类别 | 路径 |
|---|---|
| 设计文档 | `medalign_paper_framework_zh.md`、`agentic_rl_workbench_spec_zh.md`、`WORK_SUMMARY.md`、`CHANGES_LOG.md` |
| 推理/奖励代码 | `code/{agentic_rag_pipeline,gpt_agent_adapter,generator_adapter,reward_model,evidence_selection,retrieval_adapter,rag_prompting,schemas}.py` |
| 运行脚本 | `code/run_local_qwen3vl_server.sh`、`code/run_gpt_agent_rollout_smoke.py`、`code/run_agentic_rag_smoke.sh` |
| 配置 | `code/agentic_runtime_config.json` |
| Stage1 数据构造 | `data_construction/` 全套 |
| Benchmark 源数据 | `outputs/mcq_image_v2_4000/{train,dev,internal_test}.jsonl`、`outputs/qa_stage1_prepared/*`、`outputs/mcq_template_4000/*` |

### 3.2 丢失（不可找回，需重建）
| 类别 | 路径 | 说明 |
|---|---|---|
| GPT 轨迹 | `outputs/stage2_calibration/agent_context_v11_train3200.jsonl` | 3200 条逐轮轨迹 |
| SFT 数据 | `train/sft_ctrl_train.jsonl`(3787)、`train/sft_ctrl_val.jsonl`(201) | 逐轮样本，含 84 面包屑 |
| 轨迹运行时 | `code/trajectory_runtime.py` | 逐轮记录 + 面包屑 |
| 轨迹驱动 | `code/run_agent_context.py` | 跑 GPT agent 产轨迹 |
| SFT 构建 | `train/build_sft_from_trajectory.py` | 轨迹→逐轮 SFT + 面包屑重映射 |
| 全参训练器 | `train/train_ctrl_sft_full.py` | FSDP 全参 |
| FSDP 配置 | `train/fsdp_qwen35.yaml` | accelerate 配置 |
| LoRA 训练器 | `train/train_ctrl_sft_lora.py` | 已被全参取代（可选） |
| 加载校验 | `train/verify_qwen35_load.py` | 模型 load smoke |
| Checkpoint | `train/ckpt_qwen35_ctrl_full_v1`(+`_epoch1/2`, smoke) | 训练权重 |

---

## 4. 待重建脚本的设计规格（重建蓝图）

### 4.1 SFT 数据规格（build_sft_from_trajectory.py 产出）
- **逐轮一条样本**：每个检索轮 = 一条行为克隆样本。丢失前统计：轨迹 3199 条（单轮 2373 + 两轮 826），总检索轮 4025；剔除 19 条 misaligned（37 轮）后 = **3988 样本**（train 3787 / val 201）。对账：4025 − 37 = 3988。
- 非终轮标 REWRITE，终轮按 agent 决策标 ACCEPT/REWRITE。
- 面包屑：keep=[] 的 REWRITE 轮，下一轮注入 search_history 到候选 [0] 并强制 keep（n_bc 偏移，kept[n_bc:] 对齐原候选）。
- 每样本字段：`system`(v11 prompt)、`user_text`、`query_image_path`、`evidence_image_paths`(候选序)、`target`(keep/drop/action/rewrite_query/reason JSON)、`has_search_history`。

### 4.2 全参 FSDP 训练器规格（train_ctrl_sft_full.py）
- 环境：venv `qwen35-train`（transformers 5.13.1 / accelerate 1.14.0 / torch 2.8.0），`accelerate launch --config_file train/fsdp_qwen35.yaml`。
- 模型：Qwen3.5-4B（`AutoModelForImageTextToText`，dtype bf16，`model_type=qwen3_5`）。全参可训。
- **忠实多模态输入**（用户选定）：system(v11) + user[user_text, "查询图像："+query图, ("候选证据图像 [idx]："+证据图)×≤8]，复刻 `gpt_agent_adapter.decide()` 布局；面包屑候选无图自然跳过。
- **三大省显存关键**（3×48GB 才能装下全参）：
  1. **Adafactor** 优化器（分解二阶矩，优化器态从 ~12GB/卡 降到几百 MB）。
  2. **logits_to_keep**：只对尾部 assistant 目标 token 算 logits + 手写 cross_entropy，消除 LM head 全序列×152k 词表的 ~6GB 瞬时峰值。
  3. **reentrant 梯度检查点**：`gradient_checkpointing_enable(use_reentrant=True)` + `enable_input_require_grads()`（非 reentrant 会触发 Qwen3.5 attention 的 CheckpointError）。
- 超参：epochs 2、lr 1e-5、grad-accum 8、warmup 0.03、query-edge 768、ev-edge 384、max-images 8。
- 逐 epoch 存点：`save_ckpt(out_dir + _epoch{n})`，最终存 `out_dir`。稳定态 ~38GB/48GB，~10h。
- FSDP 配置要点：3 进程、FULL_SHARD、TRANSFORMER_BASED_WRAP=`Qwen3_5DecoderLayer,Qwen3_5VisionBlock`、`fsdp_use_orig_params=true`、`fsdp_cpu_ram_efficient_loading=true`、`fsdp_activation_checkpointing=false`（用 HF reentrant 代替）、bf16。

---

## 5. 环境与硬件

- **venv**：
  - `qwen3vl-rerank`（/mnt/data_1/yds/venvs/qwen3vl-rerank）：检索器 + 生成器 + 奖励。
  - `qwen35-train`（/mnt/data_1/yds/venvs/qwen35-train）：策略训练；无 deepspeed，用 accelerate+FSDP。
- **GPU**：4× RTX A6000（48GB）。GPU3 常被他人占用（勿动）。当前 GPU0/1/2 也被他人进程占用（PID 1124524–1124527），需等空闲。
- **冻结生成器 vLLM**：`bash code/run_local_qwen3vl_server.sh`（GPU_IDS 默认 0，端口 8888，Qwen3-VL-8B-Instruct，`--max-model-len 16384`）。轨迹生成 / GRPO 前需重启。

---

## 6. 重建方案（按序执行）

1. **重写脚本**（不依赖 GPU，可立即做）：
   `code/trajectory_runtime.py`、`code/run_agent_context.py`、`train/build_sft_from_trajectory.py`、`train/train_ctrl_sft_full.py`、`train/fsdp_qwen35.yaml`、`train/verify_qwen35_load.py`。写完立即 `git add && commit`。
2. **重跑 3200 GPT 轨迹**（需空闲 GPU）：重启 vLLM（GPU0）+ 检索器（GPU1）→ 跑 `run_agent_context.py` 覆盖 benchmark 3200 题 → 产 `agent_context_v11_train3200.jsonl`。
3. **重建 SFT 数据**（CPU，几分钟）：`build_sft_from_trajectory.py` → `sft_ctrl_{train,val}.jsonl`。
4. **全参 SFT**（3 卡，~10h）：`accelerate launch` → `ckpt_qwen35_ctrl_full_v1`。逐 epoch 存点校验。
5. **GRPO**（后续）：answer-utility 在线优化。

### 防再丢失纪律
- 每产出关键数据/脚本，**立即 git commit**（大数据文件也应至少 commit 脚本 + 用外部备份存数据）。
- 长训练开启逐 epoch 存点；重要 checkpoint 训完立刻 `cp` 到另一目录或 `git-lfs`/外部盘。
- 训练脚本入 git，数据产物做定期 `rsync` 备份。

---

## 7. 待办状态（重建视角）
- [ ] 重写 7 个丢失脚本（可立即，无需 GPU）
- [ ] 重跑 3200 GPT 轨迹（等 GPU）
- [ ] 重建 SFT 数据
- [ ] 全参 SFT 训练
- [ ] GRPO
