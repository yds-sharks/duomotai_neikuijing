#!/usr/bin/env bash
set -euo pipefail

TRAIN_PYTHON="${TRAIN_PYTHON:-/mnt/data_1/yds/home/miniconda3/bin/python}"
CONFIG="/mnt/data_1/yds/多模态/权重模块/train/configs/routing_cls_api10k.yaml"
SCRIPT="/mnt/data_1/yds/多模态/权重模块/train/train_text_density.py"

exec "$TRAIN_PYTHON" "$SCRIPT" --config "$CONFIG"
