#!/bin/bash
# run_config.sh <A|B|C|D|E|F|G|SMOKE> <GPU>
# 单卡运行一个消融配置（A100-80G: ctrl+gen+检索 同卡）
#
# ★ 2x2 记忆消融（1002 题严格 matched，与论文 Config-4 的 deterministic M+ 对齐）:
#   B = no memory      | E = M+ only | F = M- only | A = full memory (M+ & M-)
#   A/E 使用 --mplus-det-topk 3：REWRITE 轮无条件注入 top-3 作为 M+（方式二，
#   复刻 42.84% 的 rescue 语义：原始 origin 标签 + REPLACE 更新）
#   四臂共用标准迭代协议（Round-0 ACCEPT 即结束），仅记忆开关不同
set -e
CFG=$1; GPU=${2:-0}
cd "$(dirname "$0")"; PKGROOT=$PWD
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WEIGHT_MODULE_DIR=$PKGROOT/weight_module
export AGENTIC_ROOT=$PKGROOT
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
CKPT=${CTRL_CKPT:-$PKGROOT/models/ckpt_grpo_v4_u50}
QID=stratified_1000.jsonl; OUT=results/${CFG}${OUT_SUFFIX}
EXTRA=""
case $CFG in
  A) EXTRA="--mplus-det-topk 3" ;;                      # Full memory (det. M+ & M-)
  B) EXTRA="--no-mplus --no-mminus" ;;                  # No Memory
  C) EXTRA="--selection-only" ;;                        # KEEP/DROP only, 禁止REWRITE（已废弃）
  D) EXTRA="--rewrite-only" ;;                          # 只改写, 固定Top-k（已废弃）
  E) EXTRA="--no-mminus --mplus-det-topk 3" ;;          # M+ Only (deterministic)
  F) EXTRA="--no-mplus" ;;                              # M- Only
  G) EXTRA="--sequential" ;;                            # 固定 rewrite->retrieval->rerank（已废弃）
  SMOKE) QID=smoke10.jsonl; OUT=smoke/A ;;
  *) echo "unknown CFG $CFG"; exit 1 ;;
esac
mkdir -p logs $OUT
# 并行运行时避免 Milvus Lite 文件锁：第二个进程设 MILVUS_DB 指向 DB 副本
#   cp assets/multimodal_vector_indexes.db assets/multimodal_vector_indexes_copy1.db
#   MILVUS_DB=$PWD/assets/multimodal_vector_indexes_copy1.db bash run_config.sh E 1
MILVUS_EXTRA=""
if [ -n "$MILVUS_DB" ]; then MILVUS_EXTRA="--milvus-db-path $MILVUS_DB"; fi
CUDA_VISIBLE_DEVICES=$GPU python eval_endobench.py --mode agentic \
  --ctrl-model $CKPT --gen-model $PKGROOT/models/Qwen3-VL-8B-Instruct \
  --ctrl-device cuda:0 --gen-device cuda:0 \
  --retr-image-device cuda:0 --retr-text-device cuda:0 --weight-device cuda:0 \
  --out-dir $OUT --cand-cache assets/cand_cache_v2.jsonl \
  --translate-cache assets/endobench_translated_queries.jsonl \
  --use-organ-filter --use-weight-module \
  --max-rounds 3 --qid-file $QID --latency $EXTRA $MILVUS_EXTRA \
  2>&1 | tee logs/${CFG}${OUT_SUFFIX}_$(date +%m%d_%H%M).log
