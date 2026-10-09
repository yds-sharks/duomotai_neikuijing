# 内窥镜 Agentic RAG — 项目归档与论文主线仓库

本目录是对 `/mnt/data_1/yds/多模态` 工作区全部项目代码的重新分门别类整理（原工作目录**只读未动**，本目录为完整副本，大数据/权重除外）。
同时作为论文迭代主线仓库：论文架构正从「固定流水线」转向「真实 Agent 场景」，全部模型统一为 **Qwen3.5-4B**，后续通过 **OPD（Online Policy Distillation）** 内化模型学到的知识。每轮更新与优化记录见 [GIT_LOG.md](GIT_LOG.md)。

## 一、系统架构（真实 Agent 场景）

**核心思想**：不再使用固定多阶段流水线，而是由**中央 Agent（Qwen3.5-4B）作为大脑**，
把 RAG 的检索/证据管理封装为可调用的**工具模块**，Agent 在多轮循环中自主决策，直到提交最终答案。

```
                        ┌─────────────────────────────────────────┐
                        │      中央 Agent 大脑（Qwen3.5-4B）         │
                        │  行为空间：                                │
                        │   · keep/drop —— 候选证据筛选              │
                        │   · ACCEPT   —— 证据充分，提交生成          │
                        │   · REWRITE  —— 改写检索 query 后再次检索   │
                        └───────┬────────────────────┬────────────┘
                    工具调用     │                    │   状态输入
              ┌─────────────────▼───────┐   ┌────────▼──────────────┐
              │   检索工具层（RAG 模块）    │   │ 生成器（同一 Qwen3.5-4B）│
              │  · text_retrieve (BGE-M3) │   │ 以 logprobs 定义        │
              │  · image_retrieve (视觉塔) │   │ answer-utility 奖励      │
              └─────────────────────────┘   └───────────────────────┘

训练闭环：GPT教师轨迹 SFT → DPO → GRPO（在线） → OPD 在线蒸馏内化
评测：EndoBench 端到端（06）
```

**与旧流水线的关系**：密度门控（03）与 ODSC/PPR/SEV 重排序（04）不再是主线组件，降级为**消融对照与历史基线**；证据选择职责完全由中央 Agent 承担。旧流水线描述保留于 git 历史（Round 1 版 README）。

## 一·五、模型资产与复现指南（在其它机器复现请按此下载）

本项目全部核心角色统一使用 **Qwen3.5-4B**（原生多模态，`Qwen3_5ForConditionalGeneration`，`model_type: qwen3_5`）：

| 角色 | HF 下载来源 | 本地路径 | 备注 |
|---|---|---|---|
| Agent 大脑 / 控制器（训练对象） | https://huggingface.co/Qwen/Qwen3.5-4B | `/mnt/data_1/yds/models/hf_hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | 全参 FSDP SFT→DPO→GRPO 已跑通 |
| 答案生成器 / reward 来源（冻结） | 同上（同一份权重，无需额外下载） | 同上 | Round 2 起由 Qwen3-VL-8B-Instruct 切换为 Qwen3.5-4B |
| 图像检索编码 | 同上（取视觉塔 embedding） | 同上 | 若同步切换需重建图像向量索引 |
| 文本检索 | https://huggingface.co/BAAI/bge-m3 | `/mnt/data_1/yds/RAG/Hybrid_milvus/总版/pretrained_models/BAAI/bge-m3` | 冻结，混合检索 |

复现下载：`huggingface-cli download Qwen/Qwen3.5-4B` 与 `huggingface-cli download BAAI/bge-m3`。
关键环境：`transformers==5.13.1`（Qwen3.5 架构支持）、`peft`；chat template 必须传 `enable_thinking=False` 以保证输出简洁检索 query 而非思考过程。

## 二、目录结构

| 目录 | 内容 | 原始位置 |
|---|---|---|
| `01_data_processing/` | 文本两层存储构建（SQLite→Milvus）、数据清洗（规则+LLM）、PDF 解析 | `code/`、`data_house/` |
| `02_retrieval/` | 多模态双路检索包、BM25+BGE 混合检索 v2 | `retrieval/多模态/`、`insert/` |
| `03_weight_module/` | 权重/密度门控（L0-L4 密度 + R1-R3 图像依赖）、Mengzi-BERT 训练、密度设计文档 | `权重模块/`、`text_density_design/`、顶层 benchmark 散文件 |
| `04_rerank/` | 重排序历史基线：ODSC、PPR/PPR v2、SEV、VCMS（已被 Agent 证据选择取代，仅作消融对照） | `rerank_image_and_text/重排序/`、`核心代码梳理/` |
| `05_agentic_rag/` | **权威 Agentic RAG（v0.5）**：agent 运行时、qa 数据集构建、SFT/RFT/DPO/GRPO 训练、EndoBench 评测、论文草稿 | `rerank_image_and_text/agentic/`（唯一权威 B 树） |
| `06_evaluation/` | EndoBench 端到端评测（多轮/消融/rescue/统计显著性） | `endobench_eval/` |
| `07_experiments/` | 检索基线实验（exp_path3）、权重扫描、消融测试、早期 RAG 测试 | `insert/exp_path3/`、`retrieval/500_0.1-0.4/`、`retrieval/test/` 等 |
| `08_papers/` | BSPC（YpathRAG）投稿材料 + IEEE DataPort 数据集说明 + AAAI27 论文 LaTeX | `BSPC改稿/`、`AgenticRL_AAAI27_Overleaf_Final_2026-07-29_v1/` |
| `09_legacy_deprecated/` | **已废弃旧副本存档**（仅供追溯，勿在此开发） | 见目录内说明 |
| `third_party/` | 第三方工具（clash、BaiduPCS-Go、RAG-Anything、练习代码），**仅本地保留** | 顶层散落文件 |

## 三、大数据 / 权重位置对照（体积过大，未复制，保留原位）

| 数据 | 大小 | 原始位置 |
|---|---|---|
| 原始 PDF 语料 + 各阶段文本库 | 1.1T | `data/`（text_database.db、vector_store.db 等） |
| 多模态向量索引 / 样本库 | 23G+ | `data_house/milvus/`、`data_house/multimodal_samples.db` |
| 188w 测试 ingest | 33G | `data_house/test_ingest_188w_20260508_bgem3/` |
| 原始图文数据 | 641M | `data_house/origin_data/` |
| Agentic 训练 ckpt（SFT/GRPO 等） | 202G | `rerank_image_and_text/agentic/train/ckpt_*` |
| A100 部署包（含模型与 ckpt） | 37G | `rerank_image_and_text/agentic/_a100_pkg/` |
| 权重模块 checkpoints（Mengzi-BERT） | 3.1G | `权重模块/checkpoints/` |
| 发布资产分片 | 53G | `_release_assets/` |
| 原始文档库 | 260M | `total_store/documents/` |

## 四、Git 入库策略

- **入库**：全部代码（py/sh/yaml/tex）、文档（md）、配置（json/yaml）、论文 LaTeX 与图、小型汇总结果（csv/json）。
- **本地保留、不入库**（见 `.gitignore`）：数据库（*.db）、模型权重、实验输出目录（output*/results*/logs/ 等）、大文件（*.jsonl/*.log/*.zip/*.tar.gz）、参考文献 PDF（papers/）、第三方工具（third_party/）、含密钥的本地配置（api_config.local.json）。

## 五、legacy 说明（09_legacy_deprecated）

| 子目录 | 来源 | 废弃原因 |
|---|---|---|
| `agentic_toplevel/` | 顶层 `agentic/`（A 树） | 旧副本，数据未校验；权威版在 `05_agentic_rag/` |
| `agentic2/` | `rerank_image_and_text/agentic2/` | 更旧的中间副本 |
| `pkg_agentic/` | `_pkg/agentic/` | A100 推送用的 git 打包副本（remote: yds-sharks/agentic） |
| `pkg_endobench_eval/` | `_pkg/stage/endobench_eval_pkg/` | 评测部署打包快照 |
| `core_review_agentic_data/` | `核心代码梳理/agentic_data_construction/` | 数据构建旧副本 |
| `rerank_old_copy/` | `rerank/` | 同一仓库（yds-sharks/rerank_image_and_text）的旧提交工作副本 |
| `tmp_files/` | `核心代码梳理/_tmp_*` 等 | 临时开发文件 |

## 六、相关远程仓库

- **本归档仓库**：https://github.com/yds-sharks/duomotai_neikuijing.git
- `rerank_image_and_text`（05/04 的权威来源）：https://github.com/yds-sharks/rerank_image_and_text.git
- `agentic`（A100 推送副本）：https://github.com/yds-sharks/agentic.git

## 七、Agent 场景数据流

```
知识库构建（离线）：
  assetpack.jsonl → 01 文本库(text_database.db) → 向量索引(BGE-M3 文本索引 / 视觉塔图像索引)
  → 01 multimodal_samples.db（图文样本库，兼作题目环境池）

Agent 推理循环（在线）：
  query+图像 → 中央 Agent(Qwen3.5-4B)
    loop: [检索工具调用(text_retrieve / image_retrieve) → 候选证据块
           → Agent 评估 keep/drop → 证据充分? ACCEPT : REWRITE(改写query再检索)]
    → 最终证据集 + 生成器(Qwen3.5-4B) 出答案
    → 06 EndoBench 端到端评测 → 08 论文结果

训练闭环：
  qa_gold_4000 环境池 + GPT 教师轨迹（data_construction / train）
  → SFT(build_sft_from_trajectory) → DPO(build_dpo_pairs) → GRPO(grpo_reward, answer-utility)
  → （规划）OPD 在线蒸馏：将教师/探索中学到的策略知识内化进 Qwen3.5-4B
```
