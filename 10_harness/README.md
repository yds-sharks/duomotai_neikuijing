# 10_harness — 中央 Agent Harness（完整 RAG 封装为单一工具）

> Round 2 核心交付：把 RAG 从「固定流水线」重构为「真实 Agent 场景」。
> **完整 RAG 封装为一个工具**：中央 Agent（Qwen3.5-4B）只调用 `rag_search`——输入检索 query，
> 直接返回筛选好的检索文段。检索与证据选择的全部细节（双路检索 + 原 v0.5 证据选择 agent）
> 都在 RAG 工具内部，对大脑透明。大脑只做**查询规划**：何时检索、query 怎么写、何时提交。

## 一、架构

```
┌──────────────────────────────────────────────────────────────┐
│              AgentBrain（中央大脑, Qwen3.5-4B）                  │
│  行为空间（仅两个动作，每轮输出一个 JSON）：                        │
│    {"thought": "...", "tool": "rag_search",   "args": {...}}  │
│    {"thought": "...", "tool": "submit_answer","args": {}}     │
└──────┬───────────────────────────────────┬───────────────────┘
       │ rag_search(query)                  │ 状态渲染（文段+面包屑+预算）
┌──────▼───────────────────────┐  ┌────────▼──────────────────┐
│  RAG 工具（完整流水线封装）        │  │ 会话状态 runtime/session.py │
│  ① 双路检索 FirstStage:         │  │  · collected 跨调用累积文段  │
│     BGE-M3 文本 + 视觉塔图像    │  │  · search_history 面包屑    │
│  ② 内部证据选择 agent            │  │  · 预算: rounds/tool_calls  │
│    （原 v0.5 agent 下沉于此，    │  └───────────────────────────┘
│     失败回落 TopK）              │
│  ③ 直接输出筛选后文段            │
└──────────────────────────────┘
生成器（同一 Qwen3.5-4B，冻结）：submit 后以累积证据集作答，logprobs 定义 answer-utility
```

**关键语义**：
- 大脑**看不到**第一阶段的原始候选——`rag_search` 返回的就是 RAG 内部证据选择后的最终文段。
- v0.5 的 keep/drop 证据选择**下沉**为 RAG 工具内部处理（`AgentEvidenceFilter`）；
  ACCEPT/REWRITE 不再显式存在——「REWRITE」= 大脑换 query 再调一次 `rag_search`，「ACCEPT」= `submit_answer`。
- 多次调用的文段跨调用去重、累积为证据集；`submit_answer` 用累积集触发生成器并计算 `u_set`。

## 二、与旧模块 / 训练链路的关系

- **RAG 内核**：`backends/rag_backend.FullRagPipeline` 动态加载 v0.5 `FirstStageRetriever`
  （双路检索，候选字段与训练链路一致）；内部筛选 `AgentEvidenceFilter` 单次 LLM keep/drop（v0.5 协议），
  异常/解析失败回落 `TopKFilter`，流水线永不中断。
- **轨迹兼容**：`runtime/runner.py` 输出保留 `qid/question/options/answer/answer_text/query_image_path/obs_candidates`
  等训练链路字段，另加 `rounds[]`（每步工具调用，`candidates` = 本轮 RAG 返回的最终文段）。
- **奖励**：`backends/reward_backend.py` 的 answer-utility 对齐 `reward_model.score_candidate` 思想
  （u = 生成器在给定证据下答对为 1）；GRPO 分组优势在训练侧计算。

## 三、运行

```bash
# 冒烟（无 GPU/DB/网络依赖，全 Mock，验证循环与轨迹正确性）
python run_smoke.py

# 正式运行（需 vLLM serve Qwen3.5-4B + Milvus 索引就绪）
vllm serve /mnt/data_1/yds/models/hf_hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --port 8889
python run_harness.py --config harness_config.json --queries <endobench_queries.jsonl> --out runs/xxx.jsonl
```

`harness_config.json` 关键配置：`rag.evidence_filter.backend = agent|topk`（RAG 工具内部筛选方式，
agent 模式的 llm 节可指向同一 4B vLLM 或 GPT 教师）。

## 四、模型（统一 Qwen3.5-4B，复现见仓库根 README「模型资产与复现指南」）

- 大脑 / RAG 内部筛选 / 生成器：`Qwen/Qwen3.5-4B`（本地 snapshot `851bf6e8...`），`enable_thinking=False`
- 文本检索：`BAAI/bge-m3`；图像检索：Qwen3.5-4B 视觉塔（切换后需重建图像索引）

## 五、Roadmap

- [x] v0.1：单一 rag_search 工具 + 大脑循环 + 会话状态 + Mock 冒烟（原四工具细粒度设计已按用户要求反转）
- [ ] v0.2：接入真实检索后端 + vLLM，跑通 10 题 smoke
- [ ] v0.3：EndoBench 批量评测 + 与 v0.5 pipeline 对照
- [ ] v0.4：轨迹 → SFT/DPO 数据适配器；GRPO 在线接入
- [ ] v0.5：OPD 在线蒸馏（教师→4B 内化）
