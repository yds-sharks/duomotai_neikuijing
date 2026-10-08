# Rerank 说明

## 1. 当前任务定位

这部分不是原始检索接口，而是在“已有检索结果”基础上做二阶段重排序。

当前目标是：

- 直接输入已导出的检索结果 `retrieval_export.jsonl`
- 从每行的 `retrieval.text_top20` 和 `retrieval.image_top20` 取候选
- 做统一多模态 embedding 编码
- 构图后执行 PPR 重排序
- 最终导出给大模型 RAG 测试使用的 `retrieval.jsonl`

## 2. 当前核心脚本

重排序主脚本：

- [`rerank_multimodal_ppr.py`](/mnt/data_1/yds/多模态/核心代码梳理/rerank_multimodal_ppr.py)

它负责：

- 读取 `rag_retrieval_export` 导出的单文件 JSONL
- 调用 `Qwen3-VL-Embedding-2B`
- 构建候选图
- 执行 PPR
- 做去冗余 topk 选择
- 导出评测输入

## 3. 当前运行环境

独立新环境：

- Python: `/mnt/data_1/yds/venvs/qwen3vl-rerank/bin/python`
- Pip: `/mnt/data_1/yds/venvs/qwen3vl-rerank/bin/pip`

这个环境是单独新建的，不会改动别人的 `endo` 环境。

当前已验证：

- 可以加载 [`Qwen3-VL-Embedding-2B`](/mnt/data_1/yds/models/Qwen/Qwen3-VL-Embedding-2B)
- 可以执行最小 `encode`

## 4. 当前模型与路径

统一编码模型：

- 模型名：`Qwen3-VL-Embedding-2B`
- 本地路径：`/mnt/data_1/yds/models/Qwen/Qwen3-VL-Embedding-2B`

当前检索结果输入文件：

- 检索导出：
  [`retrieval_export.jsonl`](/mnt/data_10/mwx/workspace/multi_modal_rag/evaluate_benchmark/rag_retrieval_export/output/20260511_151007/retrieval_export.jsonl)

说明：格式文档里的旧路径写作 `evaluate_benchmark/output/retrieval_export/...`，当前机器上的实际路径是 `evaluate_benchmark/rag_retrieval_export/output/...`。

兼容旧输入：

- 如果还需要使用原来的两份 recall JSONL，可传 `--retrieval-export-jsonl ""`，再配合 `--text-jsonl` 与 `--image-jsonl`。

下游评测脚本：

- [`evaluate_from_retrieval_jsonl.py`](/mnt/data_10/mwx/workspace/multi_modal_rag/evaluate_benchmark/rag_generation_eval/evaluate_from_retrieval_jsonl.py)

## 5. 当前固定方案

文本候选：

- 使用 `retrieval.text_top20`

图片候选：

- 使用 `retrieval.image_top20`

每题候选池：

- `20 text + 20 image`

候选原始分数：

- 优先读取 `weighted_score`
- 没有时回退到 `raw_score`
- 再没有时兼容旧字段 `score`

query 初始化：

- 只使用 `query-candidate` 相似度
- 不融合第一阶段检索分数

最终导出：

- 固定 `top3`

## 6. 当前 PPR 配置

当前脚本里使用的关键参数：

- `top_m_neighbors = 5`
- `pair_boost = 0.15`
- `pair_min_weight = 0.45`
- `alpha = 0.7`
- `ppr_iters = 10`
- `select_k = 3`
- `redundancy_weight = 0.3`
- `pair_complete_bonus = 0.1`
- `min_gain = 0.05`

## 7. 当前图文配对规则

这部分非常重要。

最开始希望使用历史 `group_id` 把图片候选和文本候选直接桥接，但当前确认：

- 图片候选来自图文对侧
- 文本候选 `retrieval.text_top20` 来自文本检索链路
- 不能可靠依赖历史 `group_id` 直接给图片和 `retrieval.text_top20` 建配对边

因此当前脚本采用的是严格配对规则：

- 若文本候选 `text`
- 与图片候选 `content`
- 在规范化后完全一致
- 才认为图文对齐并添加 pair edge

规范化规则当前只做：

- 去首尾空白
- 去所有空白字符

这意味着：

- pair edge 会比较稀疏
- 但规则最明确、最安全

## 8. 输出文件

冒烟输出：

- [`endobench_multimodal_ppr_top3_smoke.jsonl`](/mnt/data_1/yds/多模态/核心代码梳理/output/endobench_multimodal_ppr_top3_smoke.jsonl)
- [`endobench_multimodal_ppr_top3_smoke_debug.jsonl`](/mnt/data_1/yds/多模态/核心代码梳理/output/endobench_multimodal_ppr_top3_smoke_debug.jsonl)

全量输出：

- [`endobench_multimodal_ppr_top3.jsonl`](/mnt/data_1/yds/多模态/核心代码梳理/output/endobench_multimodal_ppr_top3.jsonl)
- [`endobench_multimodal_ppr_top3_debug.jsonl`](/mnt/data_1/yds/多模态/核心代码梳理/output/endobench_multimodal_ppr_top3_debug.jsonl)

说明：

- `top3.jsonl` 是给评测脚本直接使用的
- `top3_debug.jsonl` 保留每题 40 个候选的分数、来源和是否被选中，便于人工排查

## 9. 当前调用命令

冒烟命令：

```bash
CUDA_VISIBLE_DEVICES=0 \
/mnt/data_1/yds/venvs/qwen3vl-rerank/bin/python \
/mnt/data_1/yds/多模态/核心代码梳理/rerank_multimodal_ppr.py \
  --retrieval-export-jsonl /mnt/data_10/mwx/workspace/multi_modal_rag/evaluate_benchmark/rag_retrieval_export/output/20260511_151007/retrieval_export.jsonl \
  --device cuda:0 \
  --limit 1 \
  --batch-size 4 \
  --output-jsonl /mnt/data_1/yds/多模态/核心代码梳理/output/endobench_multimodal_ppr_top3_smoke.jsonl \
  --debug-jsonl /mnt/data_1/yds/多模态/核心代码梳理/output/endobench_multimodal_ppr_top3_smoke_debug.jsonl
```

全量命令：

```bash
CUDA_VISIBLE_DEVICES=0 \
/mnt/data_1/yds/venvs/qwen3vl-rerank/bin/python \
/mnt/data_1/yds/多模态/核心代码梳理/rerank_multimodal_ppr.py \
  --retrieval-export-jsonl /mnt/data_10/mwx/workspace/multi_modal_rag/evaluate_benchmark/rag_retrieval_export/output/20260511_151007/retrieval_export.jsonl \
  --device cuda:0 \
  --batch-size 4 \
  --output-jsonl /mnt/data_1/yds/多模态/核心代码梳理/output/endobench_multimodal_ppr_top3.jsonl \
  --debug-jsonl /mnt/data_1/yds/多模态/核心代码梳理/output/endobench_multimodal_ppr_top3_debug.jsonl
```

## 10. 当前进度说明

截至当前：

- 冒烟测试已完成
- 输出格式已对齐评测脚本
- 全量 rerank 正在运行

当前全量规模：

- 文本题数：`6832`
- 图片题数：`6832`
- 实际处理题目数：`6832`

## 11. 后续测试怎么接

如果全量 rerank 完成，后续你要跑大模型增强测试时，应优先使用：

- [`endobench_multimodal_ppr_top3.jsonl`](/mnt/data_1/yds/多模态/核心代码梳理/output/endobench_multimodal_ppr_top3.jsonl)

接评测命令时，`--retrieval-jsonl` 就填这份文件。
