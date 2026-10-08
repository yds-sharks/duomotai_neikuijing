# 图文重排序 — 医学多模态 RAG 重排序研究

医学多模态 RAG 重排序研究（EndoBench），目标发论文。

## 工作台 Agent 体系（统一调度）
本项目由工作台 agent 体系统一管理，日常在工作台目录下发起任务。
- 工作台路径: `/mnt/workspace/yds/工作台/`
- 踩坑记录库: `/mnt/workspace/yds/工作台/config/env_pitfalls.md`
- Agent 注册表: `/mnt/workspace/yds/工作台/agents/`
- 权限矩阵: `/mnt/workspace/yds/工作台/config/permission_matrix.yaml`
- 全局索引: `/mnt/workspace/yds/工作台/config/global_registry.yaml`
- 环境配置: `config/env_setup.md`

## 子模块
- **重排序/** — ODSC 方法（主力）+ PPR baseline

## 核心脚本
- `重排序/code/rerank_odsc.py` — ODSC 重排序（Option-Discriminative Submodular Composition）
- `重排序/code/rerank_multimodal_ppr.py` — PPR baseline（已验证为负提升）

## 关键路径
- 输入数据: 见 `重排序/config/input_manifest.json`
- 方法说明: `重排序/instructions/odsc_method.md`
- Python 环境: `/mnt/data_1/yds/venvs/qwen3vl-rerank/bin/python`

## 规范
- 文件命名以子模块名为前缀
- 数据严格分 `data/tmp/` 和 `data/persistent/`
- 详细规范见 MODULE_SPEC.md
