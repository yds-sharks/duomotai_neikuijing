#!/bin/bash
# 每2小时自动刷新 GRPO 训练曲线图，训练结束后自动退出
cd /mnt/data_1/yds/多模态/rerank_image_and_text/agentic
PY=/mnt/data_1/yds/venvs/qwen35-train/bin/python
while true; do
    $PY _plot_grpo.py >/dev/null 2>&1
    echo "[$(date '+%m-%d %H:%M')] plot refreshed" >> train/_auto_plot.log
    # 训练进程不在则退出
    if ! pgrep -f train_ctrl_grpo.py >/dev/null; then
        echo "[$(date '+%m-%d %H:%M')] training finished, monitor exit" >> train/_auto_plot.log
        $PY _plot_grpo.py >/dev/null 2>&1
        break
    fi
    sleep 7200
done
