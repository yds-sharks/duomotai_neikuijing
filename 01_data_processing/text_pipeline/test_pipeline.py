#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试脚本 - 使用小批量数据验证流程
"""

import os
import sys
import json

# 测试数据路径
ASSETPACK = "/mnt/data_1/yds/多模态/data/output/消化系统与内镜/assetpack.jsonl"
TEST_DB = "/mnt/data_1/yds/多模态/data/test_text_database.db"
TEST_MILVUS = "/mnt/data_1/yds/多模态/data/test_vector_store.db"
STATS_DIR = "/mnt/data_1/yds/多模态/data/test_corpus_stats"

def create_test_jsonl():
    """提取前 500 条文本数据作为测试集"""
    test_file = "/mnt/data_1/yds/多模态/data/test_assetpack.jsonl"

    print("创建测试数据（前500条记录）...")
    count = 0
    text_count = 0

    with open(ASSETPACK, "r", encoding="utf-8") as fin, \
         open(test_file, "w", encoding="utf-8") as fout:
        for line in fin:
            try:
                item = json.loads(line.strip())
                if item.get("type") == "text" and item.get("text", "").strip():
                    fout.write(line)
                    text_count += 1
                    if text_count >= 500:
                        break
                count += 1
                if count > 2000:  # 最多扫描2000条
                    break
            except:
                continue

    print(f"测试数据创建完成: {test_file}")
    print(f"  包含 {text_count} 条文本记录")
    return test_file

def test_step1_build_database(test_file):
    """测试代码一：构建中央数据库"""
    print("\n" + "="*60)
    print("步骤1: 构建中央文本数据库")
    print("="*60)

    from build_text_database import TextDatabase, parse_jsonl_file

    # 清理旧数据
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)

    db = TextDatabase(TEST_DB)
    stats = parse_jsonl_file(test_file, db)

    # 验证
    db_stats = db.get_stats()
    print(f"\n数据库验证:")
    print(f"  文档数: {db_stats['documents']}")
    print(f"  文本块: {db_stats['text_blocks']}")
    print(f"  总字符: {db_stats['total_chars']:,}")

    return db_stats['text_blocks'] > 0

def test_step2_build_stats():
    """测试：构建 BM25 统计"""
    print("\n" + "="*60)
    print("步骤2: 构建 BM25 语料统计")
    print("="*60)

    sys.path.insert(0, "/mnt/data_1/yds/多模态/insert")
    from build_vector_store import TextDBReader, BM25StatsBuilder, Tokenizer

    reader = TextDBReader(TEST_DB)
    tokenizer = Tokenizer(
        bm25_vocab_path=None,
        jieba_userdict_path=None,
        stopwords_zh_path=None,
        stopwords_en_path=None,
    )
    builder = BM25StatsBuilder(tokenizer)

    count = 0
    for block in reader.get_text_blocks(min_length=10):
        builder.add_document(block["text"])
        count += 1

    import json
    stats = builder.build()

    os.makedirs(STATS_DIR, exist_ok=True)
    stats_path = os.path.join(STATS_DIR, "corpus_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"统计完成:")
    print(f"  处理文档: {count}")
    print(f"  语料 N: {stats['N']}")
    print(f"  平均长度: {stats['avgdl']:.2f}")
    print(f"  词表大小: {stats['vocab_size']}")
    print(f"  保存路径: {stats_path}")

    return os.path.exists(stats_path)

def test_step3_build_vector_store():
    """测试代码二：构建向量库"""
    print("\n" + "="*60)
    print("步骤3: 构建 Milvus 向量库")
    print("="*60)

    from build_vector_store import build_vectors_from_db

    # 清理旧数据
    if os.path.exists(TEST_MILVUS):
        os.remove(TEST_MILVUS)

    build_vectors_from_db(
        text_db_path=TEST_DB,
        milvus_db_path=TEST_MILVUS,
        collection_name="test_text_blocks",
        stats_dir=STATS_DIR,
        doc_id=None,
        model_name="/mnt/data_1/yds/RAG/Hybrid_milvus/总版/pretrained_models/BAAI/bge-m3",
        batch_size=32,
        min_length=10,
    )

    # 验证
    from pymilvus import connections, Collection
    connections.connect(alias="default", uri=TEST_MILVUS)
    collection = Collection("test_text_blocks")
    count = collection.num_entities
    print(f"\n向量库验证:")
    print(f"  集合: test_text_blocks")
    print(f"  实体数: {count}")

    return count > 0

def test_step4_query():
    """测试查询"""
    print("\n" + "="*60)
    print("步骤4: 查询测试")
    print("="*60)

    from build_vector_store import query_vector_store

    query_vector_store(
        milvus_db=TEST_MILVUS,
        collection_name="test_text_blocks",
        query="ESD 内镜治疗",
        model_name="/mnt/data_1/yds/RAG/Hybrid_milvus/总版/pretrained_models/BAAI/bge-m3",
        topk=3,
    )

def main():
    print("开始分层架构测试...")

    # 准备测试数据
    test_file = create_test_jsonl()

    # 执行测试
    try:
        success = True

        if not test_step1_build_database(test_file):
            print("步骤1失败!")
            success = False

        if not test_step2_build_stats():
            print("步骤2失败!")
            success = False

        if not test_step3_build_vector_store():
            print("步骤3失败!")
            success = False

        test_step4_query()

        print("\n" + "="*60)
        if success:
            print("✅ 所有测试通过!")
        else:
            print("❌ 部分测试失败")
        print("="*60)

        print(f"\n测试文件位置:")
        print(f"  测试数据: {test_file}")
        print(f"  中央数据库: {TEST_DB}")
        print(f"  向量库: {TEST_MILVUS}")
        print(f"  统计文件: {STATS_DIR}/corpus_stats.json")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
