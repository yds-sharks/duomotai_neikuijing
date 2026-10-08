# MedAlign-RAG Agentic Workspace

本目录是 MedAlign-RAG 数据构造、Agent 运行链路和训练准备的唯一工作入口。
代码与文档以当前 Git 分支为准；旧目录 /mnt/data_1/yds/多模态/agentic 只保留历史输出和
本地密钥，不再作为代码来源。

## 最终架构

~~~text
低信息 query + query image
  -> 冻结双路初检（text/image）
  -> Agent policy
       - 对候选证据执行 keep/drop
       - 判断 ACCEPT/REWRITE
       - REWRITE 时生成新 query 并重新检索
  -> 冻结 Generator
  -> answer-utility reward
~~~

主链路不包含独立 reranker。PPR 和外部 reranker 仅作为实验基线；Agent 学习的是
answer-conditioned evidence utility、检索充分性和改写动作，而不是相关性排序。

## 目录

- medalign_paper_framework_zh.md：架构与论文叙事的权威来源。
- data_construction/data_construction_design_zh.md：Stage 1 到 GRPO 的数据工程设计。
- data_construction/README.md：Stage 1 脚本顺序与运行方式。
- code/README.md：本地检索、Agent、Generator 和 reward 链路。
- WORK_SUMMARY.md：实现状态摘要。
- CHANGES_LOG.md：历史变更记录。
- papers/：相关论文原文。
- outputs/：本地运行产物，禁止提交。

## 工作目录

从仓库中的 agentic 目录启动工作：

~~~bash
cd /mnt/data_1/yds/多模态/rerank_image_and_text/agentic
~~~

API 密钥放在 data_construction/api_config.local.json。该文件被 Git 忽略，不得写入代码、
文档或提交记录。

## 当前状态

- Stage 1 的 prepare、route、API generate、盲答 verify、validate 脚本已经具备。
- Stage 2 的双路检索、GPT Agent、多轮改写和 Generator answer-utility smoke 链路已经具备。
- 密度感知路由配置已保留，但尚未接入 runtime，当前默认关闭。
- 小模型 SFT/GRPO 训练器尚未实现。

