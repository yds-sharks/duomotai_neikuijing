# text_database_stage3_filtered_labeled 向量库构建说明

## 1. 文件位置
- 输入 SQLite：`/mnt/data_10/mwx/workspace/multi_modal_rag/build_database/project_ver2/classify_blocks/data/text_database_stage3_filtered_labeled.db`
- 输出向量库：`/mnt/data_10/mwx/workspace/multi_modal_rag/build_database/project_ver2/build_vector_database/data/text_vector_database_stage3_filtered_labeled.db`
- 构建脚本：`/mnt/data_10/mwx/workspace/multi_modal_rag/build_database/project_ver2/build_vector_database/build_vector_store_multi_gpu.py`
- 查询脚本：`/mnt/data_10/mwx/workspace/multi_modal_rag/build_database/project_ver2/build_vector_database/search_vector_store.py`

## 2. 构建方法
该脚本参考：

- `/mnt/data_1/yds/多模态/code/build_vector_store_multi_gpu.py`

同时针对当前 labeled SQLite 做了字段适配。输入来自 `text_blocks`，向量库中额外保留以下标签字段：

- `organ_tags`
- `primary_knowledge_type`
- `secondary_knowledge_types`

## 3. Milvus 集合结构
默认集合名：`text_blocks`

字段：

- `pk`: INT64, 主键, auto_id
- `block_id`: INT64
- `text_hash`: VARCHAR(64)
- `doc_id`: VARCHAR(16)
- `doc_name`: VARCHAR(256)
- `page_idx`: INT64
- `text`: VARCHAR(8192)
- `summary`: VARCHAR(512)
- `organ_tags`: VARCHAR(512)
- `primary_knowledge_type`: VARCHAR(128)
- `secondary_knowledge_types`: VARCHAR(512)
- `metadata`: JSON
- `lang`: VARCHAR(8)
- `sparse_vector`: SPARSE_FLOAT_VECTOR
- `dense_vector`: FLOAT_VECTOR

索引：

- `sparse_vector`: `SPARSE_INVERTED_INDEX`, `metric_type=IP`
- `dense_vector`: `IVF_FLAT`, `metric_type=IP`

## 4. 运行步骤

### 4.1 构建 BM25 统计
```bash
python3 /mnt/data_10/mwx/workspace/multi_modal_rag/build_database/project_ver2/build_vector_database/build_vector_store_multi_gpu.py \
  --db /mnt/data_10/mwx/workspace/multi_modal_rag/build_database/project_ver2/classify_blocks/data/text_database_stage3_filtered_labeled.db \
  --milvus-db /mnt/data_10/mwx/workspace/multi_modal_rag/build_database/project_ver2/build_vector_database/data/vector_store_stage3_labeled.db \
  --mode stats \
  --stats-dir /mnt/data_10/mwx/workspace/multi_modal_rag/build_database/project_ver2/build_vector_database/corpus_stats
```

### 4.2 多 GPU 构建向量库
```bash
python3 /mnt/data_10/mwx/workspace/multi_modal_rag/build_database/project_ver2/build_vector_database/build_vector_store_multi_gpu.py \
  --db /mnt/data_10/mwx/workspace/multi_modal_rag/build_database/project_ver2/classify_blocks/data/text_database_stage3_filtered_labeled.db \
  --milvus-db /mnt/data_10/mwx/workspace/multi_modal_rag/build_database/project_ver2/build_vector_database/data/vector_store_stage3_labeled.db \
  --mode build \
  --stats-dir /mnt/data_10/mwx/workspace/multi_modal_rag/build_database/project_ver2/build_vector_database/corpus_stats \
  --collection text_blocks \
  --gpus 0,1,2,3 \
  --batch-size 16 \
  --chunk-size 10000
```

### 4.3 查询测试
```bash
python3 /mnt/data_10/mwx/workspace/multi_modal_rag/build_database/project_ver2/build_vector_database/build_vector_store_multi_gpu.py \
  --milvus-db /mnt/data_10/mwx/workspace/multi_modal_rag/build_database/project_ver2/build_vector_database/data/vector_store_stage3_labeled.db \
  --mode query \
  --collection text_blocks \
  --query "胃癌的诊断标准" \
  --topk 5
```

### 4.4 独立查询接口
```python
from search_vector_store import search, batch_search

results = search("胃癌的诊断标准", topk=5)
batch = batch_search(["胃癌的诊断标准", "结肠镜检查注意事项"], topk=3)
```

## 5. 输入表字段来源
从 SQLite `text_blocks` 读取：

- `block_id`
- `doc_id`
- `page_idx`
- `content_hash`
- `text`
- `bbox`
- `char_length`
- `organ_tags`
- `primary_knowledge_type`
- `secondary_knowledge_types`

从 SQLite `documents` 读取：

- `doc_name`

