#!/usr/bin/env bash
set -euo pipefail

TRAIN_PYTHON="${TRAIN_PYTHON:-/mnt/data_1/yds/home/miniconda3/bin/python}"

"${TRAIN_PYTHON}" /mnt/data_1/yds/多模态/权重模块/train/train_text_density.py \
  --config /mnt/data_1/yds/多模态/权重模块/train/configs/dependency_cls_v10.yaml
