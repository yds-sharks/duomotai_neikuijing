#!/usr/bin/env bash
set -euo pipefail

TRAIN_PYTHON="${TRAIN_PYTHON:-/mnt/data_1/yds/home/miniconda3/bin/python}"
export TRAIN_PYTHON

echo "== Python =="
echo "${TRAIN_PYTHON}"
"${TRAIN_PYTHON}" - <<'PY'
mods=['torch','transformers','numpy','yaml','tqdm']
for m in mods:
    try:
        mod=__import__(m)
        print(m, 'OK', getattr(mod,'__version__',''))
    except Exception as e:
        print(m, 'MISSING', repr(e))
PY

echo
echo "== GPU =="
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

echo
echo "== Disk =="
df -h /mnt/data_1/yds/多模态/权重模块 | tail -n 1

echo
echo "== Dry Run =="
"${TRAIN_PYTHON}" /mnt/data_1/yds/多模态/权重模块/train/train_text_density.py \
  --config /mnt/data_1/yds/多模态/权重模块/train/configs/smoke_density_cls_v10.yaml \
  --dry-run
