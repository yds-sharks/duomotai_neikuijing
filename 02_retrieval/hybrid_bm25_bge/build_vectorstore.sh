#!/usr/bin/env bash
set -euo pipefail

# === 路径按你的实际工程调整（已匹配你给的 YAML）===
PROJECT_ROOT="/mnt/data_1/yds/RAG/Hybrid_milvus/总版/insert"
CONFIG="$PROJECT_ROOT/config.yaml"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

echo "▶️ 使用配置: $CONFIG"

# 1) 构建 BM25 词表与统计（尊重 YAML；如需只中文/只英文，可加 --disable-en / --disable-zh）
echo "▶️ 统计 BM25 参数并生成词表..."
python -m bm25_bge_vectorstore_v2.cli_build_stats \
  --config "$CONFIG"
echo "✅ BM25 统计完成（词表写到 paths.bm25_vocab，统计写到 paths.stats_out_dir/corpus_stats.json）"

# 2) 创建/加载集合与索引（Milvus Lite 会强制 IVF_FLAT / 禁用分区；你 YAML 已设置为 IVF_FLAT、partitions: []）
echo "▶️ 创建/加载 Milvus 集合与索引..."
python -m bm25_bge_vectorstore_v2.cli_create_collection \
  --config "$CONFIG"
echo "✅ 集合与索引就绪"

# 3) 插入数据（支持断点续、去重）
echo "▶️ 开始插入数据..."
python -m bm25_bge_vectorstore_v2.cli_insert \
  --config "$CONFIG"
echo "✅ 插入完成"

# 4) 自测检索（可选，改成你想测的 query）
TEST_Q="宫颈癌的主要病因是什么？"
echo "▶️ 自测检索: $TEST_Q"
python -m bm25_bge_vectorstore_v2.cli_query \
  --config "$CONFIG" \
  --query "$TEST_Q" \
  --topk 10

echo "🎉 全流程完成"
