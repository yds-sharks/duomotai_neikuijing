#!/bin/bash
# 4配置全并行: B(GPU0)+E(GPU1)+F(GPU0)+A(GPU1)，每卡2进程
# 每进程独立 Milvus DB 副本，无文件锁冲突；DB已预热到页缓存
cd "$(dirname "$0")"
mkdir -p logs results

echo "[all-parallel] start $(date '+%m-%d %H:%M')"
bash run_config.sh B 0 & P1=$!
sleep 20
MILVUS_DB=$PWD/assets/multimodal_vector_indexes_gpu1.db bash run_config.sh E 1 & P2=$!
sleep 20
MILVUS_DB=$PWD/assets/multimodal_vector_indexes_g1.db bash run_config.sh F 0 & P3=$!
sleep 20
MILVUS_DB=$PWD/assets/multimodal_vector_indexes_g2.db bash run_config.sh A 1 & P4=$!
wait $P1 $P2 $P3 $P4

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
