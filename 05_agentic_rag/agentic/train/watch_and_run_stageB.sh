#!/bin/bash
# 自动接力守护:等待 Stage A 12 片全部完成 -> 合并 -> 自动启动 Stage B
# 完成信号:每片日志出现 [完成](脚本处理完自己区间后打印)
set -u
cd /mnt/data_1/yds/多模态/rerank_image_and_text/agentic
MERGED=train/policy_actions_v1.jsonl

echo "[守护] $(date '+%F %T') 启动,等待 Stage A 12 片完成..."
while true; do
  done=0
  for i in $(seq 0 11); do
    if grep -qa "\[完成\]" "train/policy_actions_v1_sh${i}.log" 2>/dev/null; then
      done=$((done+1))
    fi
  done
  alive=$(pgrep -fc gen_policy_actions.py 2>/dev/null || echo 0)
  tot=0
  for i in $(seq 0 11); do
    n=$(wc -l < "train/policy_actions_v1_sh${i}.jsonl" 2>/dev/null || echo 0)
    tot=$((tot+n))
  done
  echo "[守护] $(date '+%T') 完成片=${done}/12 累计写出=${tot}/3200 存活进程约=${alive}"
  if [ "$done" -ge 12 ]; then
    break
  fi
  sleep 60
done

echo "[守护] $(date '+%T') Stage A 全部完成,合并 12 片 -> ${MERGED}"
: > "$MERGED"
for i in $(seq 0 11); do
  cat "train/policy_actions_v1_sh${i}.jsonl" >> "$MERGED"
done
n=$(wc -l < "$MERGED")
echo "[守护] 合并完成:${n} 题 -> ${MERGED}"

echo "[守护] $(date '+%T') 启动 Stage B(4 卡单卡双模型 + 库分片)"
bash train/run_stageB_full.sh
echo "[守护] $(date '+%T') Stage B 已启动,守护退出"
