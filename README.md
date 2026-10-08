# 医学多模态 RAG 系统 — 整理归档

本目录是对 `/mnt/data_1/yds/多模态` 工作区全部项目代码的重新分门别类整理。
原目录**只读未动**，本目录为完整副本（大数据/权重除外，见下文对照表）。

## 一、系统流水线（四阶段）

```
① 双路检索                ② 双门路由              ③ 重排序 / Agent 证据选择        ④ 端到端生成评测 + Agentic RL
BGE-M3(文本) + Qwen3-VL(图像) → Mengzi-BERT 密度/依赖门控 → ODSC / PPR v2 / SEV / Agent(keep/drop) → Qwen3-VL + EndoBench + GRPO/SFT/RFT/DPO
```

## 二、目录结构

| 目录 | 内容 | 原始位置 |
|---|---|---|
| `01_data_processing/` | 文本两层存储构建（SQLite→Milvus）、数据清洗（规则+LLM）、PDF 解析 | `code/`、`data_house/` |
| `02_retrieval/` | 多模态双路检索包、BM25+BGE 混合检索 v2 | `retrieval/多模态/`、`insert/` |
| `03_weight_module/` | 权重/密度门控（L0-L4 密度 + R1-R3 图像依赖）、Mengzi-BERT 训练、密度设计文档 | `权重模块/`、`text_density_design/`、顶层 benchmark 散文件 |
| `04_rerank/` | ODSC 重排序（主力）、PPR/PPR v2、SEV、VCMS baseline | `rerank_image_and_text/重排序/`、`核心代码梳理/` |
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

- `rerank_image_and_text`（05/04 的权威来源）：https://github.com/yds-sharks/rerank_image_and_text.git
- `agentic`（A100 推送副本）：https://github.com/yds-sharks/agentic.git

## 七、模块间数据流

```
assetpack.jsonl (PDF解析)
  → 01 text_database.db → vector_store.db (BGE-M3)
  → 01 multimodal_samples.db → 多模态向量索引 (BGE-M3 文本 + Qwen3-VL 图像)
  → 02 双路检索 / BM25+BGE 混合检索 (top-20/30 候选)
  → 03 权重门控 (density L0-L4 / dependency R1-R3) 加权
  → 04 ODSC/PPR/SEV 重排序 → top-3 证据
  → 05 Agentic 证据选择 (keep/drop + 多轮改写) + GRPO 训练
  → 06 EndoBench 端到端评测 → 08 论文结果
```
