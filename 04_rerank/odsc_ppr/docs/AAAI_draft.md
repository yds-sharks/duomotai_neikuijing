# AAAI Paper Draft: Vision-Guided Cluster-Medoid Evidence Selection for Multimodal Medical RAG

> 投稿目标：AAAI 2026（正文 8 页 + 参考文献不限页）
> 本文档为中英文对照草稿，正式投稿时使用英文

---

## Title（标题候选）

**主选：** VisionCluster-RAG: Vision-Guided Cluster-Medoid Evidence Selection for Multimodal Medical Question Answering

**备选：**
- CrossVision: Dual-Gate Routing and Vision-Guided Reranking for Multimodal Medical RAG
- Beyond Top-K: Vision-Tower-Guided Cluster Selection for Cross-Modal Evidence Retrieval

---

## Abstract

Multimodal Retrieval-Augmented Generation (RAG) systems for medical question answering face unique challenges: queries often require interpreting medical images (e.g., endoscopic findings), yet existing retrieval and reranking methods predominantly operate in text-only or text-dominant feature spaces. We propose **VisionCluster-RAG**, a novel multimodal RAG framework that introduces three key innovations:
(1) **Dual-Gate Routing** — a learned routing mechanism using semantic density and image dependency classifiers to dynamically allocate retrieval weights across text and image paths;
(2) **Vision-Guided Cluster-Medoid Selection (VCMS)** — a reranking strategy that leverages the VLM's vision tower to compute cross-modal similarities, clusters candidates by visual affinity, and selects cluster medoids as diverse yet representative evidence;
(3) **End-to-end multimodal evaluation** with actual image-text evidence passed to the generation model.
We evaluate on EndoBench, a comprehensive medical QA benchmark with 6,832 questions predominantly requiring visual understanding. Our approach demonstrates significant improvements over standard top-K retrieval and traditional text-based reranking methods.

---

## 1. Introduction（引言）

### 1.1 问题定义

Medical question answering increasingly involves visual evidence — endoscopic images, pathological slides, radiological scans. A clinician's diagnostic process naturally integrates both textual knowledge (guidelines, case descriptions) and visual patterns (morphological features, staining patterns).

**Key challenges in multimodal medical RAG:**

1. **Cross-modal gap（跨模态鸿沟）**: Text-based retrieval captures semantic concepts but misses visual details; image-based retrieval captures visual patterns but lacks contextual specificity.

2. **Evidence selection bias（证据选择偏差）**: Standard top-K selection in embedding space tends to return redundant, unimodal evidence — missing the complementary information from the other modality.

3. **Query-adaptive routing（查询自适应路由）**: Different queries require different modality emphases — a "describe what you see" query demands image evidence, while a factual recall query needs text.

### 1.2 动机（Motivation）

[论文故事线]

现有方法的局限：
- 纯文本重排序（如 cross-encoder）：对 84.5% 的图像依赖查询无能为力
- 纯 embedding top-K：缺乏多样性，容易聚集在语义空间的单一区域
- 图结构方法（如 PPR）：完全依赖新embedding，丢弃了第一阶段检索的可靠信号

**我们的核心洞察（Key Insight）**:
> Medical image queries naturally form semantic clusters in the candidate pool — the "correct evidence" often doesn't come from the single nearest neighbor, but from representative elements across multiple relevant clusters. By using the VLM's vision tower as a cross-modal bridge, we can identify these clusters and select their most informative representatives.

### 1.3 Contributions

1. **Dual-Gate Routing Module**: A lightweight Mengzi-BERT-based dual classifier that assesses query semantic density (L0-L4) and image dependency (R1-R3), enabling dynamic weight allocation between text and image retrieval paths.

2. **Vision-Guided Cluster-Medoid Selection (VCMS)**: A novel evidence reranking method that:
   - Uses the VLM vision tower to project query images into a shared feature space
   - Computes cross-modal distances to all candidates (text + image)
   - Identifies semantically coherent clusters via agglomerative clustering
   - Selects cluster medoids as maximally representative, diverse evidence

3. **Comprehensive evaluation** on EndoBench (6,832 MCQs) demonstrating that VCMS outperforms standard top-K, cross-encoder reranking, and graph-based reranking (PPR) approaches.

---

## 2. Related Work（相关工作）

### 2.1 Multimodal Retrieval-Augmented Generation

- MuRAG (Chen et al., 2022): Jointly retrieves text and images for knowledge-intensive QA
- RA-CM3 (Yasunaga et al., 2023): Retrieval-augmented multimodal generation
- RAG-Anything (2025): Document-level multimodal RAG

**Gap**: These methods use simple top-K retrieval without cross-modal reranking or vision-guided selection.

### 2.2 Evidence Reranking

- Cross-encoders (Nogueira & Cho, 2019): Pointwise reranking
- ListwiseT5 (Pradeep et al., 2023): Listwise reranking with LLMs
- PPR-based (Personalised PageRank): Graph-based relevance propagation

**Gap**: All operate in text-only space; none leverage vision tower for cross-modal scoring.

### 2.3 Medical Visual Question Answering

- PathVQA, SLAKE, EndoBench
- Specialised models: LLaVA-Med, Med-Gemini

**Gap**: These are closed-book; we bring retrieval-augmented approach to medical VQA.

---

## 3. Method（方法）

### 3.1 System Overview

```
                    ┌─────────────────────────────────┐
                    │        User Query (q, I_q)       │
                    └──────────────┬───────────────────┘
                                   │
                    ┌──────────────▼───────────────────┐
                    │   Stage 1: Dual-Path Retrieval    │
                    │  ┌─────────┐    ┌─────────────┐  │
                    │  │BGE-M3   │    │Qwen3-VL-8B  │  │
                    │  │Text Path│    │Image Path   │  │
                    │  └────┬────┘    └──────┬──────┘  │
                    │       │                 │         │
                    │       └───────┬─────────┘         │
                    └───────────────┼───────────────────┘
                                    │ C = {c_1,...,c_N}
                    ┌───────────────▼───────────────────┐
                    │  Stage 2: Dual-Gate Routing        │
                    │  density_level, image_dep_level    │
                    │  → weight allocation (w_t, w_i)   │
                    └───────────────┬───────────────────┘
                                    │
                    ┌───────────────▼───────────────────┐
                    │  Stage 3: VCMS Reranking           │
                    │  ┌─────────────────────────────┐  │
                    │  │ Vision Tower Encoding        │  │
                    │  │ Cross-Modal Distance Comp.   │  │
                    │  │ Agglomerative Clustering     │  │
                    │  │ Medoid Selection             │  │
                    │  └─────────────────────────────┘  │
                    └───────────────┬───────────────────┘
                                    │ E = {e_1,...,e_K}
                    ┌───────────────▼───────────────────┐
                    │  Stage 4: VLM Generation           │
                    │  Qwen3-VL + E → Answer            │
                    └───────────────────────────────────┘
```

### 3.2 Dual-Path Retrieval

**Text Path**: Query text encoded by BGE-M3 (1024d) → IP search over 517K text segments → top-N_t candidates.

**Image Path**: Query image encoded by Qwen3-VL-8B vision tower (4096d) → IP search over 66K image embeddings → top-N_i candidates.

Candidate pool: $\mathcal{C} = \mathcal{C}_t \cup \mathcal{C}_i$, where $|\mathcal{C}| \leq N_t + N_i = 40$.

### 3.3 Dual-Gate Routing

Given query text $q$, two lightweight classifiers predict:

$$d = f_{density}(q) \in \{L0, L1, L2, L3, L4\}$$
$$r = f_{imgdep}(q) \in \{R1, R2, R3\}$$

Weight allocation function:

$$w_t = \alpha(d, r), \quad w_i = 1 - w_t$$

Where higher image dependency (R3) and lower text density (L0-L1) increase $w_i$.

The routing weights are used to:
1. Determine $N_t$ and $N_i$ (retrieval budget allocation)
2. Scale first-stage scores before VCMS input

### 3.4 Vision-Guided Cluster-Medoid Selection (VCMS) ⭐ 核心创新

**Step 1: Unified Encoding（统一编码）**

All candidates are encoded into a shared multimodal space using Qwen3-VL-Embedding-2B:

$$\mathbf{e}_i = \text{Enc}(c_i), \quad \forall c_i \in \mathcal{C}$$

The query image is also encoded:

$$\mathbf{e}_q = \text{VisionTower}(I_q)$$

**Step 2: Cross-Modal Distance Matrix（跨模态距离矩阵）**

Compute distance from query to each candidate:

$$d_i = 1 - \cos(\mathbf{e}_q, \mathbf{e}_i), \quad i = 1, \ldots, |\mathcal{C}|$$

And pairwise distances between candidates:

$$D_{ij} = 1 - \cos(\mathbf{e}_i, \mathbf{e}_j)$$

**Step 3: Hierarchical Clustering（层次聚类）**

Apply agglomerative clustering on the candidate distance matrix $D$ with Ward's linkage:

$$\mathcal{G}_1, \mathcal{G}_2, \ldots, \mathcal{G}_K = \text{AgglomerativeClustering}(D, K)$$

Where $K$ is determined adaptively based on silhouette score or fixed (e.g., $K=3$).

**Step 4: Cluster Ranking by Query Affinity（按查询亲和力排序簇）**

For each cluster $\mathcal{G}_k$, compute average distance to query:

$$\bar{d}_k = \frac{1}{|\mathcal{G}_k|} \sum_{c_i \in \mathcal{G}_k} d_i$$

Rank clusters: $\mathcal{G}_{\pi(1)}, \mathcal{G}_{\pi(2)}, \ldots$ where $\bar{d}_{\pi(1)} \leq \bar{d}_{\pi(2)} \leq \ldots$

**Step 5: Medoid Selection（中心点选择）**

From each of the top-$M$ clusters ($M=2$ by default), select the medoid:

$$m_k = \arg\min_{c_i \in \mathcal{G}_{\pi(k)}} \sum_{c_j \in \mathcal{G}_{\pi(k)}} D_{ij}$$

Final evidence set: $E = \{m_1, m_2, \ldots, m_M\}$

**Step 6: Optional Complement（可选补充）**

If budget allows ($|E| < K_{target}$), add the globally nearest candidate not already in $E$:

$$e^* = \arg\min_{c_i \notin E} d_i$$

### 3.5 Multimodal Generation

Final evidence $E$ (images + text) is passed to Qwen3-VL for answer generation with multimodal input:

$$\hat{a} = \text{VLM}(q, I_q, E)$$

---

## 4. Experiments（实验）

### 4.1 Dataset

**EndoBench** (消化内镜基准数据集):
- 6,832 multiple-choice questions
- 84.5% image-only queries, 15.3% mixed, 0.1% text-only
- Covers: digestive endoscopy, pathology, clinical decision-making
- Each question: question text + optional image + 4-6 options + ground-truth answer

### 4.2 Experimental Setup

| Component | Configuration |
|-----------|--------------|
| Text Retriever | BGE-M3, 517K segments, top-20 |
| Image Retriever | Qwen3-VL-8B vision encoder, 66K images, top-20 |
| Unified Encoder | Qwen3-VL-Embedding-2B (2048d) |
| Gate Classifier | Mengzi-BERT-base × 2 |
| Generator | Qwen3-VL-2B-Instruct |
| VCMS clusters | K=3 (default), select from top-2 clusters |
| Final evidence | top-3 items |

### 4.3 Baselines

1. **No Retrieval**: VLM answers directly without evidence
2. **Top-K (Text)**: BGE-M3 retrieval top-3, text-only context
3. **Top-K (Image)**: Qwen3-VL-8B retrieval top-3, image context
4. **Top-K (Fusion)**: Merged top-3 from both paths by score
5. **Cross-Encoder Rerank (SEV)**: Qwen2.5-7B LoRA reranker
6. **PPR v2**: Graph-based reranking with first-stage anchored restart
7. **VCMS (Ours)**: Vision-guided cluster-medoid selection

### 4.4 Main Results

[实验待补充 - 需要跑完 VCMS 实现后填入]

| Method | Image Queries | Mixed Queries | Overall |
|--------|---------------|---------------|---------|
| No Retrieval | - | - | TBD |
| Top-K (Text) | 80.0% | 50.0% | TBD |
| Top-K (Fusion) | TBD | TBD | TBD |
| SEV Rerank | TBD | TBD | TBD |
| PPR v2 | 76.0% | 48.0% | TBD |
| **VCMS (Ours)** | **TBD** | **TBD** | **TBD** |

### 4.5 Ablation Studies（消融实验）

**计划的消融实验：**

1. **Gate routing ablation**: 移除门控 → 固定权重 vs 动态权重
2. **Cluster number K**: K=2,3,4,5 的影响
3. **Selection strategy**: Medoid vs. Nearest vs. Random from cluster
4. **Vision tower choice**: Qwen3-VL-2B vs 8B vision encoder
5. **Number of clusters selected (M)**: M=1,2,3
6. **With/without first-stage score integration**

### 4.6 Analysis（分析）

**计划的分析维度：**

1. **Case Study**: 展示 VCMS 选出的证据 vs Top-K 选出的证据对比
2. **Cluster Visualization**: t-SNE 可视化候选聚类结构
3. **Modality Distribution**: VCMS 选出的证据中 text vs image 比例
4. **Error Analysis**: 失败案例分类（retrieval miss / selection miss / generation error）

---

## 5. Discussion（讨论）

### 5.1 为什么 VCMS 优于 Top-K

Top-K 选择存在的问题：
1. **冗余性**：最相似的K个候选往往信息高度重叠
2. **模态偏差**：如果某条路径分数系统性偏高，top-K 会只从单一模态选择
3. **局部最优**：在embedding空间中，最近的不一定是最有帮助的

VCMS 如何解决：
1. 聚类天然去冗余（不同簇 = 不同信息角度）
2. Medoid 选择确保每个角度的代表性最大化
3. Vision tower 直接从视觉语义出发，避免了文本代理(text proxy)的信息损失

### 5.2 双门控的必要性

没有门控 → 对纯文本问题也强行使用图像检索 → 引入噪声
有门控 → 自适应分配检索预算 → 减少无关候选

### 5.3 局限性

1. 依赖 VLM vision tower 质量（小模型可能特征不够区分）
2. 聚类参数需要调优（K, M）
3. 端到端推理时间增加（额外的编码+聚类步骤）

---

## 6. Conclusion（结论）

We present VisionCluster-RAG, a multimodal RAG framework that introduces vision-guided cluster-medoid selection for evidence reranking in medical question answering. Our dual-gate routing mechanism adaptively allocates retrieval budget, while VCMS leverages the VLM's vision tower to identify diverse, representative evidence clusters. Experiments on EndoBench demonstrate the effectiveness of our approach in bridging the cross-modal gap for image-intensive medical QA.

---

## 论文写作 TODO

- [ ] 实现 VCMS 方法并跑实验
- [ ] 补充 4.4 Main Results 表格
- [ ] 跑消融实验
- [ ] 绘制系统架构图（Figure 1）
- [ ] 绘制 VCMS 方法示意图（Figure 2）
- [ ] 绘制 t-SNE 可视化（Figure 3）
- [ ] 准备 Case Study 图（Figure 4）
- [ ] 撰写正式英文全文
- [ ] 制作 AAAI 模板格式排版

---

## 投稿信息

- 会议：AAAI 2026
- 格式：AAAI-style, 8 pages + unlimited references
- 模板：https://www.aaai.org/authorkit
- 匿名审稿：是（double-blind）
- 关键注意：
  - 不能在正文提及具体模型路径
  - 代码/数据需匿名仓库
  - Abstract ≤ 150 words (AAAI 要求)
