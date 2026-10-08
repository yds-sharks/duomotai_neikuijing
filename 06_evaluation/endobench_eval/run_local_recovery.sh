#!/bin/bash
# 恢复链: E -> F (错开启动) -> SFT -> RFT
# 规则: 同一时刻只允许一个进程初始化 Milvus（等 [retr] online 再启下一个）
cd /mnt/data_1/yds/多模态/endobench_eval
TRAIN=/mnt/data_1/yds/多模态/rerank_image_and_text/agentic/train
wait_log() { while ! grep -q "$2" "$1" 2>/dev/null; do sleep 30; done; }

echo "[recovery] E start $(date '+%m-%d %H:%M')"
bash run_one_1k.sh E 0 1 &
wait_log logs/1k_E.log "\[retr\] online"
sleep 20
echo "[recovery] F start $(date '+%m-%d %H:%M')"
bash run_one_1k.sh F 2 3 &

wait_log logs/1k_E.log "\[done\]"
sleep 90
echo "[recovery] SFT start $(date '+%m-%d %H:%M')"
bash run_one_1k.sh SFT 0 1 $TRAIN/ckpt_qwen35_ctrl_full_v1 &

wait_log logs/1k_F.log "\[done\]"
wait_log logs/1k_SFT_sft.log "\[retr\] online"
sleep 20
echo "[recovery] RFT start $(date '+%m-%d %H:%M')"
bash run_one_1k.sh RFT 2 3 $TRAIN/ckpt_ctrl_rft_v3 &

wait_log logs/1k_SFT_sft.log "\[done\]"
wait_log logs/1k_RFT_rft.log "\[done\]"
echo "RECOVERY ALL DONE $(date '+%m-%d %H:%M')"
