#!/bin/bash
# run_config.sh <A|B|C|D|E|F|G|SMOKE> <GPU>
# 单卡运行一个消融配置（A100-80G: ctrl+gen+检索 同卡）
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
  A) EXTRA="" ;;                                        # Clean Full (M+ & M-, learned KEEP)
  B) EXTRA="--no-mplus --no-mminus" ;;                  # No Memory
  C) EXTRA="--selection-only" ;;                        # KEEP/DROP only, 禁止REWRITE
  D) EXTRA="--rewrite-only" ;;                          # 只改写, 固定Top-k
  E) EXTRA="--no-mminus" ;;                             # M+ Only
  F) EXTRA="--no-mplus" ;;                              # M- Only
  G) EXTRA="--sequential" ;;                            # 固定 rewrite->retrieval->rerank
  SMOKE) QID=smoke10.jsonl; OUT=smoke/A ;;
  *) echo "unknown CFG $CFG"; exit 1 ;;
esac
mkdir -p logs $OUT
CUDA_VISIBLE_DEVICES=$GPU python eval_endobench.py --mode agentic \
  --ctrl-model $CKPT --gen-model $PKGROOT/models/Qwen3-VL-8B-Instruct \
  --ctrl-device cuda:0 --gen-device cuda:0 \
  --retr-image-device cuda:0 --retr-text-device cuda:0 --weight-device cuda:0 \
  --out-dir $OUT --cand-cache assets/cand_cache_v2.jsonl \
  --translate-cache assets/endobench_translated_queries.jsonl \
  --use-organ-filter --use-weight-module \
  --max-rounds 3 --qid-file $QID --latency $EXTRA \
  2>&1 | tee logs/${CFG}${OUT_SUFFIX}_$(date +%m%d_%H%M).log
