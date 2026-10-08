#!/usr/bin/env bash
set -u

PYTHON_BIN="/mnt/data_1/mwx/anaconda3/envs/endo/bin/python"
BUILD_SCRIPT="/mnt/data_1/yds/多模态/retrieval/多模态/build_multimodal_milvus.py"
MILVUS_DB="/mnt/data_1/yds/多模态/data_house/milvus/multimodal_vector_indexes.db"
TEMP_DIR="/mnt/data_1/yds/多模态/data_house/milvus/image_parallel_sd5aqpxe"
RUN_LOG="/mnt/data_1/yds/多模态/data_house/milvus/build_multimodal_milvus_resume_daemon.log"
DAEMON_LOG="/mnt/data_1/yds/多模态/data_house/milvus/run_resume_image_daemon.log"
PID_FILE="/mnt/data_1/yds/多模态/data_house/milvus/run_resume_image_daemon.pid"
TOTAL_IMAGES=66291

cleanup() {
  rm -f "$PID_FILE"
}

trap cleanup EXIT
echo "$$" > "$PID_FILE"

count_embedded() {
  python3 - <<'PY'
import os
import pickle

base = "/mnt/data_1/yds/多模态/data_house/milvus/image_parallel_sd5aqpxe"
total = 0
for idx in range(4):
    path = os.path.join(base, f"image_embeddings_shard_{idx}.pkl")
    if not os.path.exists(path):
        continue
    with open(path, "rb") as handle:
        while True:
            try:
                batch = pickle.load(handle)
            except EOFError:
                break
            total += len(batch)
print(total)
PY
}

timestamp() {
  date '+%F %T'
}

echo "[$(timestamp)] daemon_start pid=$$ temp_dir=$TEMP_DIR" | tee -a "$DAEMON_LOG"

while true; do
  embedded="$(count_embedded)"
  remaining=$((TOTAL_IMAGES - embedded))
  if [ "$remaining" -lt 0 ]; then
    remaining=0
  fi

  echo "[$(timestamp)] embedded=${embedded}/${TOTAL_IMAGES} remaining=${remaining}" | tee -a "$DAEMON_LOG"

  if [ "$embedded" -ge "$TOTAL_IMAGES" ]; then
    echo "[$(timestamp)] all embeddings are ready, starting final insert pass" | tee -a "$DAEMON_LOG"
  fi

  "$PYTHON_BIN" "$BUILD_SCRIPT" \
    --milvus-db-path "$MILVUS_DB" \
    --skip-text \
    --image-devices cuda:0,cuda:1,cuda:2,cuda:3 \
    --resume-image-temp-dir "$TEMP_DIR" \
    >> "$RUN_LOG" 2>&1
  status=$?

  if [ "$status" -eq 0 ]; then
    embedded="$(count_embedded)"
    echo "[$(timestamp)] resume_job_finished status=0 embedded=${embedded}/${TOTAL_IMAGES}" | tee -a "$DAEMON_LOG"
    exit 0
  fi

  embedded="$(count_embedded)"
  remaining=$((TOTAL_IMAGES - embedded))
  if [ "$remaining" -lt 0 ]; then
    remaining=0
  fi
  echo "[$(timestamp)] resume_job_failed status=${status} embedded=${embedded}/${TOTAL_IMAGES} remaining=${remaining}, retry in 15s" | tee -a "$DAEMON_LOG"
  sleep 15
done
