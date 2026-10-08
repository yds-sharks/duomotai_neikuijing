# Runtime Source Manifest

本文件记录 agentic runtime 复用的本机组件及边界。

## 冻结双路初检

实现来源：

- /mnt/data_1/yds/多模态/retrieval/多模态/search_multimodal_vector_store.py
- /mnt/data_1/yds/多模态/retrieval/多模态/build_multimodal_milvus.py
- /mnt/data_1/yds/多模态/retrieval/多模态/text_dense_module.py
- /mnt/data_1/yds/多模态/retrieval/多模态/image_dense_module.py

封装入口：retrieval_adapter.py。

## Agent 控制器

gpt_agent_adapter.py 接收问题、选项、query image、候选证据文本和候选证据图像像素，一次输出：

~~~json
{
  "keep": [0, 2],
  "drop": [1, 3],
  "action": "REWRITE",
  "rewrite_query": "..."
}
~~~

Agent 的 keep/drop 是答案效用判断，不生成全序相关性排名。

## 冻结 Generator 与评测

可复用评测入口：

- /mnt/data_1/yds/多模态/retrieval/test/evaluate_qwen3_vl.py
- /mnt/data_1/yds/多模态/retrieval/test/evaluate_unified.py
- /mnt/data_1/yds/多模态/retrieval/test/utils.py
- /mnt/data_1/yds/多模态/retrieval/500_0.1-0.4/run_qwen3_vl_rag_full500.sh

本目录封装：generator_adapter.py、run_local_qwen3vl_server.sh 和
run_existing_endobench_rag_eval.sh。

## 明确排除

旧 PPR 和 Qwen3-VL-Embedding rerank 脚本不再是 agentic runtime 依赖。它们只可用于论文中的
历史/负结果基线，不得重新接入 Agent 前端。

## 当前数据流

~~~text
QA gold
  -> original_query
  -> retrieval_text + retrieval_image
  -> Agent selected_evidence + ACCEPT/REWRITE
  -> generator_response + prediction
  -> answer_utility + group advantage
~~~

