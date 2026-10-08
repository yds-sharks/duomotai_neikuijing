YpathRAG Benchmark: Reranking and QA Evaluation Data for Pathology Evidence Retrieval
======================================================================================

This dataset accompanies the paper:
"YpathRAG: Supportiveness-Aware Evidence Retrieval and Reranking for Pathology Question Answering"

Directory Structure:
--------------------

reranking_benchmark/
  - eval_questions.json          : Reranking evaluation queries with supportive/non-supportive passage annotations
  - RAG_baseline.json            : Baseline RAG retrieval results
  - 基础模型_评估结果.json        : Base model evaluation results

qa_benchmark/
  - all_question.json            : Full pathology QA evaluation question set
  - 评估结果.csv                  : Downstream QA evaluation results (metrics per question)

ablation_data/
  - BGE_only_top30_cls_partitioned.jsonl   : BGE-only retrieval top-30 candidates (partitioned)
  - BGE_only_top30_patched.jsonl           : BGE-only retrieval top-30 candidates (patched)
  - Hybrid_top30_cls_partitioned.jsonl     : Hybrid retrieval top-30 candidates (partitioned)
  - Hybrid_top30_patched.jsonl             : Hybrid retrieval top-30 candidates (patched)

Usage:
------
These datasets can be used to:
1. Evaluate reranking models on pathology evidence retrieval
2. Benchmark downstream QA performance with retrieved evidence
3. Reproduce ablation experiments comparing dense-only vs. hybrid retrieval

Contact:
--------
Corresponding authors: Tian Guan, Yonghong He
Email: heyh@sz.tsinghua.edu.cn
