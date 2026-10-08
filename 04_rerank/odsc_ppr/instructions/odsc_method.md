# ODSC: Option-Discriminative Submodular Composition

## 1. 方法定位

在已有检索结果（retrieval_export.jsonl）基础上做二阶段重排序。
与 PPR 方法对应，输入输出格式完全相同，可直接替换。

## 2. 核心思想

不问"候选跟问题有多相关"，而问"这组候选能不能帮模型把正确答案从干扰项中挑出来"。

三个关键创新：
1. **Option-Discriminative**: 把 question 拆成多个 sub-query（question + 每个选项），计算候选对每个选项的支持度，选择"态度鲜明"的候选
2. **Submodular Composition**: 用 submodular 函数建模证据的组合价值，而非独立打分取 topk
3. **Cross-Modal Coherence**: 同一知识点的图文互证给额外加分

## 3. 处理流程

```
输入: retrieval_export.jsonl（每题 text_top20 + image_top20）

Step 1: 构造选项感知 sub-query
  question + option_A, question + option_B, ...

Step 2: 统一编码
  - sub-queries: encode_queries（带 prompt）
  - 文本候选: encode_texts
  - 图片候选: encode_images_with_text（图片+描述联合编码）

Step 3: 计算支持度向量
  每个候选 → [sim(cand, sub_A), sim(cand, sub_B), ...]

Step 4: 计算区分力
  disc = max_support - second_max_support

Step 5: Submodular 贪心选择
  f(S) = λ_disc * 区分力之和
       + λ_cover * 选项覆盖度
       + λ_cross * 跨模态互证
  每步选 marginal gain 最大的候选

输出: retrieval.jsonl（每题 top-k 候选）
```

## 4. 对比 PPR 方法的改进

| 维度 | PPR | ODSC |
|------|-----|------|
| 排序信号 | query-candidate 二元相似度 | option-candidate 多维支持度 |
| 图片编码 | 只传图片像素 | 图片+描述联合编码 |
| 选择策略 | PPR传播 + MMR去冗余 | Submodular 贪心（区分力+覆盖+互证） |
| 图文桥接 | 硬匹配 pair_boost | 基于 embedding 相似度 |
| 理论保证 | 无 | (1-1/e) 近似最优 |

## 5. 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| lambda_disc | 1.0 | 区分力权重 |
| lambda_cover | 0.5 | 选项覆盖权重 |
| lambda_cross | 0.3 | 跨模态互证权重 |
| cross_sim_threshold | 0.5 | 跨模态配对的相似度阈值 |
| select_k | 3 | 每题选出的候选数 |

## 6. 运行命令

冒烟测试：

```bash
CUDA_VISIBLE_DEVICES=0 \
/mnt/data_1/yds/venvs/qwen3vl-rerank/bin/python \
/mnt/workspace/yds/图文重排序/重排序/code/rerank_odsc.py \
  --retrieval-export-jsonl /mnt/data_10/mwx/workspace/multi_modal_rag/evaluate_benchmark/rag_retrieval_export/output/20260511_151007/retrieval_export.jsonl \
  --model-path /mnt/data_1/yds/models/Qwen/Qwen3-VL-Embedding-2B \
  --device cuda:0 \
  --limit 5 \
  --batch-size 4 \
  --output-jsonl /mnt/workspace/yds/图文重排序/重排序/data/tmp/endobench_odsc_top3_smoke.jsonl \
  --debug-jsonl /mnt/workspace/yds/图文重排序/重排序/data/tmp/endobench_odsc_top3_smoke_debug.jsonl
```

## 7. 论文 story

**Option-Discriminative Submodular Evidence Composition for Multimodal Medical RAG**

三个贡献点：
1. 重新定义 reranking 目标：从 relevance 转为 option-discriminative power
2. Submodular 组合优化框架：同时建模区分力、选项覆盖、跨模态互证
3. EndoBench 实验验证
