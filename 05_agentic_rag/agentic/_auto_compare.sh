#!/bin/bash
cd /mnt/data_1/yds/多模态/rerank_image_and_text/agentic
PY=/mnt/data_1/yds/venvs/qwen35-train/bin/python
while true; do
    $PY _compare_lr.py >/dev/null 2>&1
    echo "[$(date '+%m-%d %H:%M')] lr-compare refreshed" >> train/_auto_compare.log
    if ! pgrep -f "ckpt_grpo_lr3e6_test" >/dev/null; then
        echo "[$(date '+%m-%d %H:%M')] lr-test finished, exit" >> train/_auto_compare.log
        $PY _compare_lr.py >/dev/null 2>&1
        break
    fi
    sleep 1800
done
