#!/usr/bin/env bash
# 看门狗:等 Stage A(v2,12片)全部完成 -> 严格去重合并到 3200 -> 跑 Stage B(v2)
#          -> 等 Stage B(4片)完成 -> 合并到 rollouts_recalled_v2.jsonl -> 停,报告就绪。
# Stage C/D 待 GPU 空闲后单独 smoke 验证再全量,避免影响正在跑的 Stage A。
set -u
cd /mnt/data_1/yds/多模态/rerank_image_and_text/agentic
PYP=/mnt/data_1/yds/venvs/qwen35-train/bin/python
A_MERGED=train/policy_actions_v2.jsonl
B_MERGED=train/rollouts_recalled_v2.jsonl

echo "[watch] 等待 Stage A 12 片完成..."
while :; do
  done=0
  for i in $(seq 0 11); do
    grep -qa "\[完成\]" "train/policy_actions_v2_sh${i}.log" 2>/dev/null && done=$((done+1))
  done
  echo "[watch $(date +%H:%M:%S)] Stage A 完成 ${done}/12"
  [ "$done" -eq 12 ] && break
  sleep 60
done

echo "[watch] Stage A 完成,严格去重合并 -> $A_MERGED"
"$PYP" - <<'PY'
import json, glob
seen=set(); n=0
with open("train/policy_actions_v2.jsonl","w",encoding="utf-8") as out:
    for fp in sorted(glob.glob("train/policy_actions_v2_sh*.jsonl")):
        for line in open(fp,encoding="utf-8"):
            line=line.strip()
            if not line: continue
            q=json.loads(line).get("qid")
            if q in seen: continue
            seen.add(q); out.write(line+"\n"); n+=1
print(f"[merge] Stage A 唯一题数={n} -> policy_actions_v2.jsonl")
PY

echo "[watch] 启动 Stage B(v2)..."
bash train/run_stageB_full.sh
sleep 30

echo "[watch] 等待 Stage B 4 片完成..."
while :; do
  done=0
  for i in 0 1 2 3; do
    grep -qa "\[完成\]" "train/rollouts_recalled_v2_sh${i}.log" 2>/dev/null && done=$((done+1))
  done
  echo "[watch $(date +%H:%M:%S)] Stage B 完成 ${done}/4"
  [ "$done" -eq 4 ] && break
  sleep 30
done

echo "[watch] Stage B 完成,合并 -> $B_MERGED"
"$PYP" - <<'PY'
import json, glob
seen=set(); n=0
with open("train/rollouts_recalled_v2.jsonl","w",encoding="utf-8") as out:
    for fp in sorted(glob.glob("train/rollouts_recalled_v2_sh*.jsonl")):
        for line in open(fp,encoding="utf-8"):
            line=line.strip()
            if not line: continue
            q=json.loads(line).get("qid")
            if q in seen: continue
            seen.add(q); out.write(line+"\n"); n+=1
print(f"[merge] Stage B 唯一题数={n} -> rollouts_recalled_v2.jsonl")
PY

echo "[watch] A+B 完成。下一步:smoke 验证 Stage C(gen_policy_select_round2.py)与 Stage D(score_rewards.py)后再全量。"
