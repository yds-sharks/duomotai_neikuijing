# Agentic RAG Runtime

本目录实现 v0.4 统一链路：初检候选直接进入 Agent，由 Agent 联合完成证据选择和查询改写。
运行时没有独立 reranker，也不依赖 PPR。

## 运行链路

~~~text
query_text + query_image
  -> FirstStageRetriever：文本/图像双路初检
  -> GPTAgent 或本地 policy：
       keep/drop + ACCEPT/REWRITE(+rewrite_query)
  -> REWRITE：抑制已见证据并重新检索，最多 T 轮
  -> OpenAICompatibleGenerator：基于 kept evidence 作答
  -> answer-utility：P(correct | kept evidence) - P(correct | no evidence)
~~~

Agent 会接收 query image 和候选图像证据的像素。所有证据索引统一为 0-based。
初检 score 只作为观察字段，不作为训练奖励。

## 文件

- agentic_runtime_config.json：本机数据库、模型、服务和运行参数。
- schemas.py：共享数据结构。
- retrieval_adapter.py：冻结的文本/图像初检封装及多轮 suppression。
- gpt_agent_adapter.py：GPT Agent 的结构化 keep/drop + ACCEPT/REWRITE 接口。
- generator_adapter.py：OpenAI-compatible Generator 与 logprobs。
- reward_model.py：answer-utility、约束项和 GRPO 组内优势。
- rag_prompting.py：候选证据与最终回答 prompt。
- agentic_rag_pipeline.py：端到端运行 CLI。
- run_gpt_agent_rollout_smoke.py：GPT Agent + answer-utility smoke。
- run_local_qwen3vl_server.sh：启动冻结 Qwen3-VL Generator。
- run_agentic_rag_smoke.sh：小规模端到端 smoke。
- run_existing_endobench_rag_eval.sh：复用现有 EndoBench 评测器。

## 配置

项目内路径从 agentic 根目录解析，不再依赖旧的绝对 agentic 路径。数据库和模型是本机外部资源，
仍在 agentic_runtime_config.json 中显式配置。

默认本地配置：

- Stage 1/GPT Agent API：../data_construction/api_config.local.json
- QA gold：../outputs/qa_stage1_verified/qa_gold.jsonl
- Runtime 输出：../outputs/runtime/
- Generator API：http://127.0.0.1:8888/v1

密度感知路由尚未接入执行代码，因此 routing.enabled 当前为 false。

## Smoke

在 agentic 根目录运行：

~~~bash
bash code/run_local_qwen3vl_server.sh
bash code/run_agentic_rag_smoke.sh
~~~

只检查检索与 Agent，不调用 Generator：

~~~bash
python3 code/agentic_rag_pipeline.py \
  --input-jsonl outputs/qa_stage1_verified/qa_gold.jsonl \
  --output-jsonl outputs/runtime/retrieval_agent_smoke.jsonl \
  --limit 2 \
  --skip-generator
~~~

运行 GPT rollout 与 answer-utility：

~~~bash
python3 code/run_gpt_agent_rollout_smoke.py --limit 2
~~~

## 设计边界

- Retriever、可选路由和 Generator 冻结，只有 Agent policy 训练。
- Agent 不输出最终答案，只选择证据并决定是否改写。
- evidence-hit 仅用于离线分析，不进入主奖励。
- PPR/外部 reranker 仅作为基线，不进入 runtime。

