#!/bin/bash
# P2 出题监控：每 15 分钟记录一行进度 + 近 50 行成功率（判断中转站恢复/断供）
# generate 进程退出时记录最终状态并结束。全程只读，不干预任务。
DIR=/mnt/data_1/yds/多模态/内窥镜agentic/11_v03_training/data_construction/p2_full
RAW=$DIR/p2_mcq_raw.jsonl
LOG=$DIR/monitor.log
PY=/mnt/data_1/yds/venvs/qwen35-train/bin/python
while true; do
  ST=$($PY -c "
import json
try:
    rows = [json.loads(l) for l in open('$RAW')]
except FileNotFoundError:
    print('0,0,0,0/50'); raise SystemExit
ok = sum(1 for r in rows if 'error' not in r)
tail = rows[-50:]
tok = sum(1 for r in tail if 'error' not in r)
print(f\"{len(rows)},{ok},{len(rows)-ok},{tok}/50\")
" 2>/dev/null)
  if ! ps aux | grep "p2_generate_mcq.py generate" | grep -v grep > /dev/null; then
    echo "$(date '+%m-%d %H:%M') GENERATE-EXITED rows,ok,err,近50: $ST" >> $LOG
    break
  fi
  echo "$(date '+%m-%d %H:%M') rows,ok,err,近50: $ST" >> $LOG
  sleep 900
done
