# ==============================
# Path3 实验：环境变量与目录初始化
# ==============================

# 你当前项目根（你当时就是在这里执行的）
cd /mnt/data_1/yds/RAG/Hybrid_milvus/总版 || exit 1

# 1) 代码根：insert 目录（包含 exp_path3/ 与 bm25_bge_vectorstore_v2/）
export PROJECT_ROOT="/mnt/data_1/yds/RAG/Hybrid_milvus/总版/insert"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# 2) 配置文件
export CONFIG="$PROJECT_ROOT/config.yaml"

# 3) 你的 300 题 + 16 候选数据
export DATA="/mnt/data_1/yds/微调/分类模型微调/数据集构造_new/备份/最终正类_all_负类生成.json"

# 4) 实验输出根目录（SIGIR补充）+ 当前 path3 子目录
export OUT="/mnt/data_1/yds/微调/实验/SIGIR补充/path3_p1_head10"

# 5) 初始化输出目录结构
mkdir -p "$OUT"/{seeds,align,runs,metrics,logs}

# 6) 打印核对（建议你每次开新终端都先跑这一段）
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "CONFIG=$CONFIG"
echo "DATA=$DATA"
echo "OUT=$OUT"

# 7) 基础存在性检查（失败就说明路径写错或未挂载）
ls -lah "$PROJECT_ROOT" | head
ls -lah "$CONFIG"
ls -lah "$DATA"
ls -lah "$OUT"
