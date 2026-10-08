# 多模态 RAG 系统完整技术管线总结

> 用于论文撰写和架构图绘制参考

## 总体架构

```
用户 Query
    │
    ├─── [文本路径] BGE-M3 编码 ──→ text_vector_index ──→ text_top20
    │
    └─── [图像路径] Qwen3-VL-8B 视觉编码 ──→ image_vector_index ──→ image_top20
                                                                        │
                    ┌────────────────────────────────────────────────────┘
                    │
            ┌───────▼────────┐
            │  门控路由模块    │
            │  (Mengzi-BERT)  │
            ├─────────────────┤
            │ • density L0-L4 │
            │ • image_dep R1-R3│
            └───────┬─────────┘
                    │
            ┌───────▼─────────────────┐
            │   多模态重排序 (PPR v2)    │
            │  Qwen3-VL-Embedding-2B   │
            ├──────────────────────────┤
            │ • restart = β·FS+(1-β)·ES│
            │ • 候选图 + 跨模态边       │
            │ • PPR 传播 + 贪心去冗余   │
            └───────┬─────────────────┘
                    │
                    ▼
            top-3 evidence ──→ VLM (Qwen3-VL) ──→ 答案
```

---

## 第一阶段：双路向量召回（Dual-Path Retrieval）

### 文本检索路径

| 项目 | 内容 |
|------|------|
| Embedding 模型 | BAAI/BGE-M3 |
| 向量维度 | 1024 |
| 向量库 | Milvus Lite `text_vector_index` |
| 库规模 | 517,717 条 |
| 检索方式 | query text → BGE-M3 编码 → IP 相似度搜索 |
| 返回数量 | top-20 |
| 模型路径 | `/mnt/data_1/yds/RAG/Hybrid_milvus/总版/pretrained_models/BAAI/bge-m3` |

### 图像检索路径

| 项目 | 内容 |
|------|------|
| Embedding 模型 | Qwen3-VL-8B-Instruct（仅视觉编码器，不走 decoder/lm_head） |
| 向量维度 | 4096 |
| 向量库 | Milvus Lite `image_vector_index` |
| 库规模 | 66,291 条 |
| 检索方式 | query image → Qwen3-VL 视觉编码 → IP 相似度搜索 |
| 返回数量 | top-20 |
| 模型路径 | `/mnt/data_10/mwx/huggingface_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/` |

### 共同配置

- 向量库位置：`/mnt/data_1/yds/多模态/data_house/milvus/multimodal_vector_indexes.db`
- 主数据库：`/mnt/data_1/yds/多模态/data_house/multimodal_samples.db`
- 分类过滤字段：`body_site_main`（level1）+ `knowledge_type_main`（level2）
- nprobe：64

### 备选：混合 BM25+向量检索

- 位置：`insert/bm25_bge_vectorstore_v2/`
- 配置：BM25 k1=1.2, b=0.75; dense 权重 0.6, sparse 权重 0.4
- 当前主管线未使用此方案

---

## 第二阶段：双门控路由（Dual-Gate Routing）

### 门控 1：语义密度评估（Semantic Density Gate）

| 项目 | 内容 |
|------|------|
| 底座模型 | Mengzi-BERT-base |
| 任务类型 | 5 分类 |
| 输入 | query_text（max_length=96） |
| 输出 | density_level: L0-L4 |
| Dropout | 0.1 |
| Checkpoint | `权重模块/checkpoints/density_cls_v10_mengzi` |

**密度等级语义**：
- L0：极低密度（泛问/短句，几乎不提供可定位语义）
- L1：低密度（部分约束，很弱的方向）
- L2：中密度（包含部位或病变中的一部分关键信息）
- L3：高密度（较完整的判别线索）
- L4：极高密度（问题非常具体，包含多项明确约束）

### 门控 2：图片依赖评估（Image Dependency Gate）

| 项目 | 内容 |
|------|------|
| 底座模型 | Mengzi-BERT-base |
| 任务类型 | 3 分类 |
| 输入 | query_text（max_length=96） |
| 输出 | image_dependency_level: R1/R2/R3 |
| Checkpoint | `权重模块/checkpoints/dependency_cls_v10_mengzi` |

**依赖等级语义**：
- R1：低图片依赖（仅看文本通常也可完成判断）
- R2：中图片依赖（文本提供部分信息，结合图片更稳妥）
- R3：高图片依赖（不看图片通常难以可靠判断）

### 辅助模型：密度排序器

| 项目 | 内容 |
|------|------|
| 底座模型 | Mengzi-BERT-base |
| 任务类型 | Pairwise ranking |
| 输出 | 连续密度分数（同核心查询内相对排序） |
| Checkpoint | `权重模块/checkpoints/density_rank_v10_mengzi` |

### 门控决策逻辑

- density_level → 调控检索策略激进程度
- image_dependency_level → 决定是否启用/加权图像检索路径
- 两者联合 → 动态路由权重分配

### 训练数据构造方案

- 策略：锚点驱动混合构造
- 流程：final_description → 槽位抽取(anatomy/lesion/attribute/distribution/severity) → semantic_core → L0-L4 query链生成 → 独立审查
- 锚点评分公式：slot_score + specificity_bonus - deictic_penalty - generic_penalty - redundancy_penalty

---

## 第三阶段：多模态重排序（Cross-Modal Reranking）

### 方案 A：PPR v2（Retrieval-Anchored PPR）⭐ 当前主方案

| 组件 | 技术细节 |
|------|---------|
| 编码模型 | Qwen3-VL-Embedding-2B |
| 图结构 | 候选间 cosine similarity 边（top-5 neighbors）+ 跨模态配对边 |
| Restart 向量 | `restart[i] = β × first_stage_norm[i] + (1-β) × query_cand_sim_norm[i]` |
| β（推荐值） | 0.7（偏信一阶段检索，0.3 给新 embedding 信号） |
| PPR α | 0.7（teleport 概率） |
| 迭代次数 | 10 |
| 选择策略 | 贪心去冗余 top-3（redundancy_weight=0.3, pair_complete_bonus=0.1） |
| 跨模态配对边 | group_id 匹配 或 规范化文本完全一致 → boost 0.15, min_weight 0.45 |

**核心创新**：
- 一阶段检索分数不丢弃，作为 PPR restart 向量的主导信号
- embedding 相似度作为补充信号（30%权重）
- 避免了纯 embedding 信号导致的排序退化

### 方案 B：SEV 交叉编码器重排序

| 组件 | 技术细节 |
|------|---------|
| 底座模型 | Qwen2.5-7B-Instruct |
| 微调方式 | LoRA 适配器 |
| 头部 | cls_head (logits) + score_head (sigmoid) |
| 输入模板 | `"问题：{q}\n内容：{p}"` |
| 输出 | P(support) ∈ [0,1] |
| 融合 | `final = α × first_stage_norm + (1-α) × sev_score`（α=0.5） |
| 适配器路径 | `/mnt/data_1/yds/微调/分类模型微调/数据集构造_new/微调BGE/训练/listwise微调/save_weights_ddp` |

### 方案 C：ODSC（Option-Discriminative Submodular Composition）

| 组件 | 技术细节 |
|------|---------|
| 编码模型 | Qwen3-VL-Embedding-2B |
| 子查询构建 | question + 每个 option 独立编码 |
| 优化目标 | 判别性 + 选项覆盖度 + 跨模态一致性 |
| 选择方式 | 贪心子模优化 |
| 特点 | 针对 MCQ 场景设计 |

---

## 第四阶段：端到端 RAG 生成与评测

| 项目 | 内容 |
|------|------|
| 生成模型 | Qwen3-VL-2B-Instruct（通过 vLLM 部署） |
| 输入方式 | 图文混合（text + image 多模态输入） |
| 评测数据 | EndoBench 6832 题 MCQ |
| 评测指标 | MCQ 准确率（端到端） |
| 数据组成 | 84.5% 纯图像 query / 15.3% text+image 混合 / 0.1% 纯文本 |

---

## 已验证的实验结果

| 方案 | 准确率 | 相对 Baseline |
|------|--------|---------------|
| Baseline（直接 top-3） | 53.0% (n=500) | — |
| SEV α=0.5 | 52.2% | -0.8% |
| PPR v2 β=0.7（纯图像query, n=50） | 78.0% | -2.0% vs 80.0% |
| PPR v2 β=0.7（混合query, n=50） | 50.0% | ±0.0% vs 50.0% |

**关键发现**：
- 之前评测仅传入文本 context，VLM 无法看到原图
- 对 84% 的纯图像 query，文本重排序方法天然受限
- 需要改为图文混合输入评测，才能真正体现视觉重排序的价值

---

## 核心代码位置汇总

| 模块 | 路径 |
|------|------|
| 向量库构建 | `retrieval/多模态/build_multimodal_milvus.py` |
| 统一检索接口 | `retrieval/多模态/search_multimodal_vector_store.py` |
| 门控权重服务 | `权重模块/multimodal_weight_service.py` |
| 语义密度推理 | `权重模块/semantic_density_service.py` |
| PPR v1 重排序 | `rerank/重排序/code/rerank_multimodal_ppr.py` |
| PPR v2 重排序 | `rerank/重排序/code/rerank_ppr_v2.py` |
| SEV 重排序 | `rerank/重排序/code/rerank_sev_multimodal.py` |
| ODSC 重排序 | `rerank/重排序/code/rerank_odsc.py` |
| 评测脚本 | `rerank/重排序/code/eval_ppr_v2_mixed.py` |
| 官方评测 | `/mnt/data_10/mwx/.../evaluate_from_retrieval_jsonl.py` |

---

## 模型路径汇总

| 模型 | 用途 | 路径 |
|------|------|------|
| BGE-M3 | 文本 embedding | `/mnt/data_1/yds/RAG/Hybrid_milvus/总版/pretrained_models/BAAI/bge-m3` |
| Qwen3-VL-8B | 图像 embedding（检索） | `/mnt/data_10/mwx/huggingface_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/` |
| Qwen3-VL-Embedding-2B | 统一编码（重排序图） | `/mnt/data_1/yds/models/Qwen/Qwen3-VL-Embedding-2B` |
| Mengzi-BERT-base | 门控分类 | `/mnt/data_1/yds/微调/models/mengzi-bert-base` |
| Qwen2.5-7B-Instruct | SEV 重排序 | `/mnt/data_1/yds/微调/models/Qwen2.5-7B-Instruct` + LoRA |
| Qwen3-VL-2B-Instruct | RAG 生成评测 | `/mnt/data_10/mwx/.../Qwen3-VL-2B-Instruct` |
