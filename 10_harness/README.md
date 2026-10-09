# 10_harness — 中央 Agent Harness（RAG 作为工具模块）

> Round 2 核心交付：把 RAG 从「固定流水线」重构为「真实 Agent 场景」。
> 中央 Agent（Qwen3.5-4B）作为大脑，RAG 的检索/证据管理全部封装为**可调用的工具**，
> Agent 在多轮循环中自主决策直到提交答案。每步产生可训练轨迹（与 SFT/DPO/GRPO 数据链路兼容）。

## 一、架构

```
┌────────────────────────────────────────────────────────────┐
│                AgentBrain（中央大脑, Qwen3.5-4B）             │
│  每轮输出 JSON 工具调用:                                       │
│    {"thought": "...", "tool": "...", "args": {...}}         │
└──────┬─────────────────────────────────┬───────────────────┘
       │ 工具调用（ToolRegistry 分发）       │ 状态渲染（编号候选+面包屑+预算）
┌──────▼──────────────┐        ┌──────────▼──────────────────┐
│  工具层 tools/        │        │  会话状态 runtime/session.py  │
│  text_retrieve       │        │  · collected_evidence 跨轮累积│
│  image_retrieve      │        │  · last_candidates 本轮编号   │
│  keep_evidence       │        │  · search_history 面包屑      │
│  submit_answer       │        │  · 预算: rounds/tool_calls    │
└──────┬──────────────┘        └─────────────────────────────┘
       │
┌──────▼──────────────────────────────────────────────────────┐
│  后端层 backends/（全部可注入、可 Mock）                          │
│   retrieval_backend → FirstStageRetriever(02_retrieval 同源逻辑)│
│   llm_backend       → OpenAI 兼容 API(vLLM serve Qwen3.5-4B)/GPT│
│   reward_backend    → answer-utility(生成器 logprobs)           │
└──────────────────────────────────────────────────────────────┘
```

## 二、工具协议（行为空间）

| 工具 | args | 语义 | 对应 v0.5 行为 |
|---|---|---|---|
| `text_retrieve` | `{"query": str, "k": int?}` | 以文本 query 检索文本库，返回编号候选 | REWRITE 的检索动作 |
| `image_retrieve` | `{"k": int?}` | 以查询图像检索图像库（默认查询图像，可传 `image_path` 覆盖） | 图像检索 |
| `keep_evidence` | `{"keep": [编号...]}` | 将本轮候选中选中项加入跨轮证据集（其余视为 drop） | keep/drop |
| `submit_answer` | `{}` | 认可当前证据集，触发生成器作答并结束 | ACCEPT |

**关键变化**：v0.5 的 `ACCEPT/REWRITE` 二选一不再显式存在——「REWRITE」退化为 Agent 自主发起的下一次
`*_retrieve` 调用（换 query 重检索），「ACCEPT」即 `submit_answer`。策略知识在工具选择中自然涌现。

## 三、与旧模块 / 训练链路的关系

- **检索逻辑**：`backends/retrieval_backend.py` 包装 `05_agentic_rag/agentic/code/retrieval_adapter.FirstStageRetriever`
  （动态加载），候选字段与 `schemas.EvidenceItem` 一致。
- **轨迹兼容**：`runtime/runner.py` 输出保留 `qid/question/options/answer/answer_text/query_image_path/obs_candidates`
  等训练链路字段（`train/build_sft_from_trajectory.py` 等可直接消费），另加 `rounds[]`（每步工具调用）。
- **奖励**：`backends/reward_backend.py` 的 `answer_utility` 对齐 `reward_model.score_candidate` 的
  answer-utility 思想（u = 生成器在给定证据下答对为 1）；GRPO 分组优势在训练侧计算。

## 四、运行

```bash
# 冒烟（无 GPU/DB/网络依赖，全 Mock，验证循环与轨迹正确性）
python run_smoke.py

# 正式运行（需 vLLM serve Qwen3.5-4B + Milvus 索引就绪）
vllm serve /mnt/data_1/yds/models/hf_hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --port 8889 --enable-auto-tool-choice  # 或普通 serve，走 chat.completions
python run_harness.py --config harness_config.json --queries <endobench_queries.jsonl> --out runs/xxx.jsonl
```

## 五、模型（统一 Qwen3.5-4B，复现见仓库根 README「模型资产与复现指南」）

- 大脑与生成器：`Qwen/Qwen3.5-4B`（本地 snapshot `851bf6e8...`），chat template 需 `enable_thinking=False`
- 文本检索：`BAAI/bge-m3`；图像检索：Qwen3.5-4B 视觉塔（切换后需重建图像索引）

## 六、Roadmap

- [x] v0.1：工具协议 + 大脑循环 + 会话状态 + Mock 冒烟
- [ ] v0.2：接入真实检索后端 + vLLM，跑通 10 题 smoke
- [ ] v0.3：EndoBench 批量评测 + 与 v0.5 pipeline 对照
- [ ] v0.4：轨迹 → SFT/DPO 数据适配器；GRPO 在线接入
- [ ] v0.5：OPD 在线蒸馏（教师→4B 内化）
