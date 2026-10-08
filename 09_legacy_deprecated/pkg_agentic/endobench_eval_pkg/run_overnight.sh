#!/bin/bash
# 2x A100-80G 计划：2x2 记忆消融（1002 题严格 matched）
#   wave1: B=no memory (GPU0) + E=M+ only (GPU1)
#   wave2: F=M- only  (GPU0) + A=full memory (GPU1)
# GPU1 的进程使用 Milvus DB 副本避免 Lite 文件锁；
# 每波内 GPU1 延迟启动，等 GPU0 进程检索器上线（日志出现 "[retr] online"）。
cd "$(dirname "$0")"
mkdir -p logs results

# 准备 DB 副本（仅第一次需要，约数分钟拷贝）
if [ ! -f assets/multimodal_vector_indexes_copy1.db ]; then
  echo "[prep] copying milvus db for parallel run $(date '+%m-%d %H:%M')"
  cp assets/multimodal_vector_indexes.db assets/multimodal_vector_indexes_copy1.db
fi

wait_retr_online() {
  # $1 = 配置名; 等待该配置最新日志出现 [retr] online
  local cfg=$1
  for i in $(seq 1 120); do
    if ls -t logs/${cfg}_*.log >/dev/null 2>&1 \
       && grep -q "\[retr\] online" "$(ls -t logs/${cfg}_*.log | head -1)"; then
      return 0
    fi
    sleep 15
  done
  echo "[warn] ${cfg} retr online not detected after 30min, continuing anyway"
}

echo "[wave1] B start $(date '+%m-%d %H:%M')"
bash run_config.sh B 0 & P1=$!
wait_retr_online B
echo "[wave1] E start $(date '+%m-%d %H:%M')"
MILVUS_DB=$PWD/assets/multimodal_vector_indexes_copy1.db bash run_config.sh E 1 & P2=$!
wait $P1 $P2

echo "[wave2] F start $(date '+%m-%d %H:%M')"
bash run_config.sh F 0 & P3=$!
wait_retr_online F
echo "[wave2] A start $(date '+%m-%d %H:%M')"
MILVUS_DB=$PWD/assets/multimodal_vector_indexes_copy1.db bash run_config.sh A 1 & P4=$!
wait $P3 $P4

echo "ALL DONE $(date '+%m-%d %H:%M')"
echo "== summary =="
for c in B E F A; do
  python - <<PYEOF
import json, glob
fs = glob.glob("results/$c/agentic_summary.json")
if fs:
    d = json.load(open(fs[0]))
    o = d.get("overall", {})
    print(f"$c: acc={o.get('acc',0)*100:.2f}% n={o.get('n',0)}")
else:
    print("$c: no summary yet")
PYEOF
done
