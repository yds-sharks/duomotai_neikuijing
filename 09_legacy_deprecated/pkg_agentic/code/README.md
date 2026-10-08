# Agentic RAG Runtime Code

本目录用于承接 Stage 2/3 的真实链路实验：把 Stage 1 构造出的可靠 QA 样本接入本地检索、rerank 和 generator，得到 original query 与 rewrite query 的真实链路结果。

## 目标

统一封装：

```text
query_text + optional query_image
 -> first-stage retrieval (top20)
 -> 按 first-stage 检索分数取 top5（无 reranker 模型）
 -> agent 证据筛选 + rewrite（见 run_gpt_agent_rollout_smoke.py）
 -> local Qwen3-VL generator 或外部 OpenAI-compatible API
 -> prediction / response
```

当前目录只做链路封装，不训练模型，不自动启动大规模实验。

## 文件说明

```text
agentic_runtime_config.json       默认路径和模型服务配置
schemas.py                        统一数据结构
retrieval_adapter.py              first-stage text/image 检索封装
evidence_selection.py             按 first-stage 分数取 top5（无 reranker）+ 证据子集选取
gpt_agent_adapter.py              GPT 证据筛选 + rewrite agent 封装
generator_adapter.py              OpenAI-compatible generator 封装
reward_model.py                   answer-utility reward + GRPO 组内 advantage
rag_prompting.py                  RAG prompt 和答案解析
agentic_rag_pipeline.py           端到端 CLI：QA JSONL -> retrieval/top5/generation JSONL
run_local_qwen3vl_server.sh       启动本地 Qwen3-VL vLLM 服务
run_agentic_rag_smoke.sh          小规模 smoke 示例
source_manifest.md                本封装复用的原始脚本来源
```

## 最小 smoke

只做检索 + rerank，不调用 generator：

```bash
python3 /mnt/data_1/yds/多模态/agentic/code/agentic_rag_pipeline.py \
  --input-jsonl /mnt/data_1/yds/多模态/agentic/outputs/qa_stage1_prepared/pilot_qa_candidates.jsonl \
  --output-jsonl /mnt/data_1/yds/多模态/agentic/outputs/runtime_smoke/pilot_retrieval.jsonl \
  --limit 2 \
  --skip-generator
```

调用本地 Qwen3-VL generator 前，先启动 vLLM：

```bash
bash /mnt/data_1/yds/多模态/agentic/code/run_local_qwen3vl_server.sh
```

然后运行：

```bash
python3 /mnt/data_1/yds/多模态/agentic/code/agentic_rag_pipeline.py \
  --input-jsonl /mnt/data_1/yds/多模态/agentic/outputs/qa_stage1_prepared/pilot_qa_candidates.jsonl \
  --output-jsonl /mnt/data_1/yds/多模态/agentic/outputs/runtime_smoke/pilot_rag.jsonl \
  --limit 2
```

## 输入格式

优先支持 Stage 1 QA gold 数据字段：

```json
{
  "qid": "...",
  "question": "...",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "answer": "A",
  "answer_text": "...",
  "low_information_query_seed": "这张图是什么部位？",
  "query_image_path": "..."
}
```

也兼容当前 `pilot_qa_candidates.jsonl` 的候选字段。若没有 `question/options/answer`，pipeline 会只输出检索和 rerank 结果。

## 设计边界

- first-stage retrieval、generator 全部冻结；无独立 reranker，top5 由 first-stage 分数直接截断。
- agent 负责证据筛选与 query rewrite，不直接改答案。
- original query 默认使用 Stage 1 的规则 `low_information_query_seed`。
- 当前 pipeline 是 Stage 2/3 的运行底座，后续 GPT agent rollout 和 reward 计算会接在这个接口之上。
