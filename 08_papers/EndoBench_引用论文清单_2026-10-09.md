# EndoBench 论文原文与引用论文清单

> 整理日期：2026-10-09。来源：Semantic Scholar citations API（arXiv:2505.23601，共 18 篇收录；Google Scholar 显示约 19+，可能略全）。

## 一、论文原文

**EndoBench: A Comprehensive Evaluation of Multi-Modal Large Language Models for Endoscopy Analysis**

- 作者：Shengyuan Liu 等（CUHK-AIM-Group）
- 发表：**NeurIPS 2025 Datasets & Benchmarks Track**（Advances in Neural Information Processing Systems 38）
- arXiv：<https://arxiv.org/abs/2505.23601>（2025-05-29）
- 项目主页：<https://cuhk-aim-group.github.io/EndoBench.github.io/>
- 代码：<https://github.com/CUHK-AIM-Group/EndoBench>
- 数据集：HF `Saint-lsy/EndoBench`（本项目所用 test split，6832 题）
- 内容：4 内镜场景 × 12 临床任务 × 6 视觉提示粒度，6832 题多选验证题，语料来自 21 个数据源

## 二、引用论文（按用法分类）

### A 类：把 EndoBench 当评测基准/对比表（最值得细看）

| 论文 | 出处 | EndoBench 用法 |
|---|---|---|
| **Evaluating Domain-Agnostic Retrieval-Augmented Generation for Gastrointestinal Medical VQA** | ImageCLEF 2026（CLEF 工作坊） | **与本项目最同赛道**：GI 内镜 MedVQA + 领域无关 RAG。PDF 含 3 处指向 EndoBench arXiv 的引用链接。PDF: <https://clef-staging.pages.dev/paper257.pdf> |
| Benchmarking Egocentric Clinical Intent Understanding Capability for Medical MLLMs | arXiv 2601.06750 | 将 EndoBench 列入医学基准对比表（6832 题、non-ego、仅图像输入），作为能力维度对照 |
| Parameter-Efficient VLMs for Gastrointestinal Endoscopy | IEEE EMBS 2025（arXiv 2605.24792） | 综述式介绍：EndoBench 覆盖 4 场景/12 任务/6832 验证题，作为 GI 领域标准评测集引用 |
| Beyond the Leaderboard: Rethinking Medical Benchmarks for LLMs | Annual Meeting（AAAI 系） | 医学基准方法论反思，以 EndoBench 为讨论对象之一 |

### B 类：同类医学多模态基准互引（基准设计参考）

| 论文 | 出处 | 说明 |
|---|---|---|
| X-PCR: Cross-modality Progressive Clinical Reasoning (Ophthalmic) | arXiv 2604.20350 | 眼科同类基准，引用 EndoBench 作领域对照 |
| Colon-X: Advancing Intelligent Colonoscopy toward Clinical Reasoning | arXiv 2512.03667 | 结肠镜推理数据集+基准，方法论互引 |
| OmniBrainBench: Brain Imaging Multi-stage Benchmark | arXiv 2511.00846 | 脑影像同类基准互引 |
| DermoGPT: Dermatological Reasoning MLLMs | arXiv 2601.01868 | 皮肤科领域，引用 EndoBench 作跨领域范式参考 |

### C 类：内镜/手术领域应用论文（背景引用）

| 论文 | 出处 |
|---|---|
| SurgGraph: Geometry-Grounded Scene Graphs for Laparoscopic Video | arXiv 2609.25651 |
| Colon-Bench: Agentic Workflow for Dense Lesion Annotation in Colonoscopy Videos | arXiv 2603.25645 |
| SurgΣ: Large-Scale Multimodal Data and Foundation Models for Surgical Intelligence | arXiv 2603.16822 |
| Surg-R1: Hierarchical Reasoning Foundation Model for Surgical Decision Support | arXiv 2603.12430 |
| LLM-Driven Analysis and Report Generation of Endoscopy Videos (Pilot) | Digestive Endoscopy（期刊） |
| Measuring and Improving Complex-Atomic Answer Consistency in Endoscopic VQA | arXiv 2607.17834（related work 引用：EndoBench 评测多场景/多任务/多粒度） |
| Metascope: Optics-Driven Neural Network for Ultra-Micro Metalens Endoscopy | IEEE ICCV 系（arXiv 2508.03596） |

### D 类：医学 AI 方法论文（参考文献引用）

| 论文 | 出处 |
|---|---|
| FD-MSP: Domain-Adaptive Polyp Segmentation | BMC Medical Imaging |
| MKG-CARE: Case-Aware Reasoning with Multimodal Knowledge Graphs | arXiv 2605.22547 |
| MedSAM-Agent: Interactive Segmentation with Multi-turn Agentic RL | arXiv 2602.03320（agentic RL 方法论相关，任务为分割非 VQA） |

## 三、对本项目的启示

1. **#18 ImageCLEF 2026 是唯一直接同行**（GI MedVQA + RAG）：重点看其检索/重排设计与 ImageCLEFmed 任务口径，可对照本 Harness 的 rag_search 管道。
2. 主流用法是**整集评测 MLLM**（A 类口径）：我们的 agent 化用法（检索规划 + 证据选择下沉）在引用者中尚无先例——这正是论文叙事的新颖点。
3. #10 MedSAM-Agent 证明"MLLM + 多轮 agentic RL"路线在医学影像活跃，可作为方法学引用背景。
