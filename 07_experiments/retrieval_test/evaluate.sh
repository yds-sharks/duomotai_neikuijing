#!/bin/bash

# --profiles Qwen3-VL-2B-Instruct,Qwen3-VL-8B-Instruct,deepseek-vl2-tiny,gpt-4o-mini,gpt-4o,Hulu-Med-7B,HuatuoGPT-Vision-7B 

cd /mnt/data_10/mwx/workspace/endo_benchmark/project_ver4/

python evaluate_unified.py \
  --mode baseline \
  --profiles Qwen3-VL-8B-Instruct \
  --benchmark Saint-lsy/EndoBench \
  --split test \
  --dataset all \
  --task all \
  --scene all \
  --category all \
  --subtask all \
  --limit 500 \
  --num-runs 1 \
  --temperature 0.2 \
  --top-p 0.9 \
  --max-tokens 512 \
  --save-response \
  --gpu 0,1,2,3 \
  --gpu-memory-utilization 0.9 

python evaluate_unified.py \
  --mode rag \
  --profiles Qwen3-VL-8B-Instruct \
  --benchmark Saint-lsy/EndoBench \
  --split test \
  --dataset all \
  --task all \
  --scene all \
  --category all \
  --subtask all \
  --limit 500 \
  --retrieval-query-rewrite \
  --retrieval-query-rewrite-model gpt-4o-mini \
  --retrieval-query-translate-lang zh \
  --num-runs 1 \
  --temperature 0.2 \
  --top-p 0.9 \
  --max-tokens 512 \
  --retrieval-query-mode question_options \
  --topk 5 \
  --retrieval-batch-size 256 \
  --gpu 0,1,2,3 \
  --save-response \
  --gpu-memory-utilization 0.9 