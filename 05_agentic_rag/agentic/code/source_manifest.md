# Source Manifest

`agentic/code` 是后续 agentic RAG 链路的封装层，复用了以下已有工程代码：

## 多模态向量检索

- `/mnt/data_1/yds/多模态/retrieval/多模态/search_multimodal_vector_store.py`
- `/mnt/data_1/yds/多模态/retrieval/多模态/build_multimodal_milvus.py`
- `/mnt/data_1/yds/多模态/retrieval/多模态/text_dense_module.py`
- `/mnt/data_1/yds/多模态/retrieval/多模态/image_dense_module.py`

封装入口：`retrieval_adapter.py`

## 证据取用（无 reranker）

新框架不再使用独立 reranker。粗召回 top20 后直接按 first-stage 检索分数取 top5（文本+图片混排）。

封装入口：`evidence_selection.py`（`select_top_evidence` / `apply_selection`）。

历史上的 PPR/Qwen3-VL-Embedding rerank 脚本（`核心代码梳理/rerank_multimodal_ppr.py` 等）已从本链路移除，仅作历史参考。

## 本地模型评测 / Generator

- `/mnt/data_1/yds/多模态/retrieval/test/evaluate_qwen3_vl.py`
- `/mnt/data_1/yds/多模态/retrieval/test/evaluate_unified.py`
- `/mnt/data_1/yds/多模态/retrieval/test/utils.py`
- `/mnt/data_1/yds/多模态/retrieval/500_0.1-0.4/run_qwen3_vl_rag_full500.sh`

封装入口：`generator_adapter.py`、`rag_prompting.py`、`run_local_qwen3vl_server.sh`

## Stage 2/3 统一入口

- `agentic_rag_pipeline.py`

该入口后续会被 GPT agent rollout 和 reward 计算脚本调用：

```text
original_query/rewrite_query
 -> agentic_rag_pipeline.py
 -> retrieval_text / retrieval_image / top5_evidence / generator_response / prediction
 -> reward annotation
```
