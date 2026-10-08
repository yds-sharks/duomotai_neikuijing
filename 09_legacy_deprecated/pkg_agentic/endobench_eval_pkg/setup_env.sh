#!/bin/bash
# EndoBench 消融评测包 — A100 服务器一键安装
# 用法: bash setup_env.sh   （私有仓库需先 gh auth login）
set -e
cd "$(dirname "$0")"
export PKGROOT=$PWD

echo "== [1/5] python venv =="
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

echo "== [2/5] HF 公开模型 (直连HF请: export HF_ENDPOINT= ) =="
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir models/Qwen3-VL-8B-Instruct
hf download BAAI/bge-m3 --local-dir models/bge-m3
hf download Langboat/mengzi-bert-base --local-dir models/mengzi-bert-base

echo "== [3/5] Release 资产 (u50 ckpt / Milvus DB / 权重模块 / 缓存) =="
bash download_assets.sh

echo "== [4/5] 路径写入配置 =="
sed -i "s|PKGROOT|$PKGROOT|g" code/agentic_runtime_config.json weight_module/weight_module_runtime.yaml
sed -i "s|/mnt/data_1/yds/微调/models/mengzi-bert-base|$PKGROOT/models/mengzi-bert-base|g" \
  weight_module/checkpoints/*/config.resolved.yaml
# 同时修正 config.resolved.yaml 中的 checkpoint 目录和数据路径
sed -i "s|/mnt/data_1/yds/多模态/权重模块|$PKGROOT/weight_module|g" \
  weight_module/checkpoints/*/config.resolved.yaml

export AGENTIC_ROOT=$PKGROOT
echo "== [5/5] 10题冒烟 (GPU0) =="
bash run_config.sh SMOKE 0
echo "SETUP COMPLETE — 运行: bash run_overnight.sh"
