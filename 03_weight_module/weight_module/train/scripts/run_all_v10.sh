#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="/mnt/data_1/yds/多模态/权重模块/checkpoints/logs"
TRAIN_PYTHON="${TRAIN_PYTHON:-/mnt/data_1/yds/home/miniconda3/bin/python}"
export TRAIN_PYTHON
mkdir -p "${LOG_DIR}"

bash /mnt/data_1/yds/多模态/权重模块/train/scripts/run_train_density_cls.sh \
  2>&1 | tee "${LOG_DIR}/density_cls_v10_$(date +%Y%m%d_%H%M%S).log"

bash /mnt/data_1/yds/多模态/权重模块/train/scripts/run_train_dependency_cls.sh \
  2>&1 | tee "${LOG_DIR}/dependency_cls_v10_$(date +%Y%m%d_%H%M%S).log"

bash /mnt/data_1/yds/多模态/权重模块/train/scripts/run_train_density_rank.sh \
  2>&1 | tee "${LOG_DIR}/density_rank_v10_$(date +%Y%m%d_%H%M%S).log"
