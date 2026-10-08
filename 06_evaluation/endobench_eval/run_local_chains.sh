#!/bin/bash
# 接续链: pair(0,1) A->E->SFT ; pair(2,3) B->F->RFT
# 等待上一配置日志出现 [done] 再启动下一个（避免 Milvus 并发初始化冲突）
cd /mnt/data_1/yds/多模态/endobench_eval
TRAIN=/mnt/data_1/yds/多模态/rerank_image_and_text/agentic/train
wait_done() {
  while ! grep -q "\[done\]" "$1" 2>/dev/null; do sleep 60; done
  sleep 30
}
chain1() {
  wait_done logs/1k_A.log
  echo "[chain1] E start $(date '+%m-%d %H:%M')"
  bash run_one_1k.sh E 0 1
  wait_done logs/1k_E.log
  echo "[chain1] SFT start $(date '+%m-%d %H:%M')"
  bash run_one_1k.sh SFT 0 1 $TRAIN/ckpt_qwen35_ctrl_full_v1
}
chain2() {
  wait_done logs/1k_B.log
  echo "[chain2] F start $(date '+%m-%d %H:%M')"
  bash run_one_1k.sh F 2 3
  wait_done logs/1k_F.log
  echo "[chain2] RFT start $(date '+%m-%d %H:%M')"
  bash run_one_1k.sh RFT 2 3 $TRAIN/ckpt_ctrl_rft_v3
}
chain1 & C1=$!
chain2 & C2=$!
wait $C1 $C2
echo "ALL LOCAL CHAINS DONE $(date '+%m-%d %H:%M')"
