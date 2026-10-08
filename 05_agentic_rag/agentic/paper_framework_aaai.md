# MedAlign-RAG: Answer-Utility Guided Evidence Selection and Query Rewriting for Multimodal Medical RAG

## Title Candidates
1. MedAlign-RAG: Learning to Select and Rewrite via Answer-Utility Reward for Multimodal Medical QA
2. Close the Loop: Online Retrieval-Augmented RL for Evidence-Grounded Medical Question Answering
3. When to Rewrite? Answer-Utility Guided Agentic RAG for Multimodal Medical QA

---

## Abstract (待训练数据填充后定稿，以下为结构模板)

Multimodal medical question answering (QA) systems increasingly rely on
retrieval-augmented generation (RAG) to ground answers in clinical evidence.
However, when initial retrieval fails—due to sparse queries, strong image
dependency, or visually similar distractors—conventional RAG pipelines lack a
self-recovery mechanism. We propose **MedAlign-RAG**, a lightweight controller
that operates on a *frozen* multimodal RAG front-end (retriever + generator)
and learns two coupled skills: (1) **evidence selection**—identifying the subset
of retrieved passages that maximally improves answer correctness, and (2)
**adaptive query rewriting**—deciding *when* evidence is insufficient and
*how* to reformulate the query for a second retrieval round.

The controller is trained with **GRPO** (Group Relative Policy Optimization)
under a novel **answer-utility reward**: the log-probability lift of the gold
answer induced by the selected evidence, measured by a frozen vision-language
generator. Crucially, REWRITE actions are evaluated through an
**online retrieval-in-the-loop** mechanism—the rewritten query is executed
against the real vector index during training, so the reward reflects genuine
retrieval improvement rather than a proxy signal. To stabilize training, we
introduce (i) a **tier-aware curriculum** that classifies questions by
difficulty and allocates exploration budget accordingly, (ii) **forced-prefix
exploration** to prevent mode collapse toward trivial ACCEPT actions, and
(iii) **noise-robust advantage estimation** with spread gating to suppress
gradient noise from uninformative groups.

Experiments on EndoBench (6,832 clinically validated VQA pairs across 4
endoscopic scenarios and 12 clinical tasks) show that MedAlign-RAG improves
accuracy by [X.X]% over the frozen RAG baseline and [X.X]% over a GPT-4o
agent, while the controller adds only [Y]M trainable parameters. Ablations
confirm that both the answer-utility reward and the online retrieval loop are
essential: removing either degrades performance to below the no-rewrite
baseline.

---

## 1. Introduction

### 1.1 Motivation
- Multimodal medical QA: image + text → answer (endoscopy, pathology, radiology)
- RAG is essential: medical knowledge is vast, LLMs hallucinate
- Problem: one-shot retrieve → generate is brittle
  - Short/image-dependent queries → weak text retrieval signal
  - Visually similar distractors (same organ, different pathology)
  - Reranking can only reorder existing candidates, cannot change recall distribution

### 1.2 Our Approach
- Train a SMALL controller (Qwen3.5-4B) on a FROZEN RAG stack
- Controller decides: which evidence to keep + whether to rewrite query
- Key insight: reward = "does this evidence/rewrite help the generator answer correctly?"
  - Not relevance, not format compliance, not LLM-as-judge
- Training: GRPO with online retrieval for REWRITE actions (closed loop)

### 1.3 Contributions
1. We formalize **answer-utility reward** (ΔlogP_gold) for training evidence controllers, providing dense, continuous training signal directly tied to downstream QA performance.
2. We introduce **online retrieval-in-the-loop GRPO**: REWRITE rollouts execute real vector search during training, closing the reward loop for query reformulation.
3. We propose a **tier-aware curriculum** with forced-prefix exploration and noise-robust advantage estimation, enabling stable RL training on heterogeneous medical QA.
4. On EndoBench (NeurIPS 2025), MedAlign-RAG achieves [X]% accuracy with a 4B controller, outperforming [baselines] while keeping retriever and generator frozen.

---

## 2. Related Work

### 2.1 Multimodal Medical QA & RAG
- MedVQA benchmarks: PathVQA, VQA-RAD, SLAKE, EndoBench
- Medical RAG: MedRAG, MA-RAG (multi-round), retrieval-augmented MLLMs
- Limitation: fixed pipeline, no self-recovery

### 2.2 Agentic RAG & Query Rewriting
- Self-RAG, CRAG, Adaptive-RAG: when to retrieve / when to rewrite
- MA-RAG: multi-round agentic RAG for medical (2026)
- ReasonRAG: process vs outcome reward (2025)
- Limitation: reward is sparse (correct/incorrect) or proxy (relevance)

### 2.3 RL for LLM Alignment
- GRPO, Dr. GRPO, REINFORCE
- RLHF for retrieval: RLRF, Search-R1
- Limitation: not applied to multimodal medical RAG with real retrieval

---

## 3. Method

### 3.1 Problem Formulation
- Frozen front-end: Retriever R (BGE-M3 + Milvus), Generator G (Qwen3-VL-8B)
- Controller π_θ: given (question, options, query_image, top-K evidence) → JSON action
- Action space: {ACCEPT(keep⊆[K]), REWRITE(keep⊆[K], new_query)}
- Multi-round: REWRITE → re-retrieve → new top-K → controller again (max 2 rounds)

### 3.2 Answer-Utility Reward
```
u(action) = log P_G(gold | question, options, evidence(action)) 
           − log P_G(gold | question, options, ∅)
```
- Continuous, dense signal (vs binary correct/incorrect)
- Same scale for ACCEPT and REWRITE actions
- Leakage penalty: hard penalty if rewrite_query contains gold answer letter

### 3.3 Online Retrieval-in-the-Loop
- During GRPO rollout: REWRITE action → execute `search_text(new_query, k=5)` on live Milvus index
- Retrieved evidence merged with kept evidence → scored by frozen generator
- Deduplication cache: identical rewritten queries within a group share retrieval results
- Cost: ~200ms per unique rewrite query (BGE-M3 encode + Milvus ANN)

### 3.4 Tier-Aware Curriculum
- Offline judge pass: for each training question, test if any single evidence flips the answer
  - **trivial**: generator correct without evidence → subsample 15%
  - **selection**: some evidence flips answer → train evidence selection (K_rewrite=1)
  - **rewrite**: no single evidence helps → train query rewriting (K_rewrite=8)
- Effect: focuses compute on states where the controller can actually improve outcomes

### 3.5 Forced-Prefix Exploration
- In each GRPO group of G=8: 2 rollouts are forced behind a REWRITE prefix
- Prefix tokens excluded from policy gradient (no gradient through forced tokens)
- Prevents mode collapse: without this, policy converges to always-ACCEPT (safe but suboptimal)

### 3.6 Noise-Robust Advantage Estimation
- **Spread gate**: skip PG update if max(reward) − min(reward) < τ_spread (group carries no ranking signal)
- **Std floor**: advantage denominator = max(std, σ_floor), preventing noise-only groups from being inflated to ±1
- Addresses Dr. GRPO's critique of z-normalization in low-variance groups

### 3.7 Training Details
- Policy: Qwen3.5-4B (bfloat16, gradient checkpointing)
- GRPO: G=8, T=1.4, top-p=0.95, β_KL=0.02, lr=3e-6, Adafactor
- 4× A6000 48GB: policy(c0) + ref(c1) + scorer×2(c2,c3)
- Training data: 3200 source-grounded MCQ (NOT from benchmark), 4-choice, with query images
- ~1672 effective training states after tier filtering

---

## 4. Experimental Setup

### 4.1 Evaluation Benchmark
- **EndoBench** (NeurIPS 2025): 6,832 VQA pairs, 4 endoscopic scenarios, 12 clinical tasks
  - Tasks: anatomical recognition, lesion detection, procedure identification, etc.
  - Multi-choice format (A-F), clinically validated by experts
- [Optional: PathVQA for cross-domain generalization]

### 4.2 Baselines
| Method | Description |
|---|---|
| Closed-book | Generator answers without any retrieval |
| RAG-top5 | Retrieve top-5, feed all to generator (no selection) |
| RAG-top5 + BM25 rerank | Traditional reranking baseline |
| GPT-4o agent | GPT-4o as the controller (same prompt, same tools) |
| SFT controller | Supervised fine-tuning on GPT-4o trajectories (no RL) |
| **MedAlign-RAG (ours)** | GRPO-trained controller with answer-utility reward |

### 4.3 Metrics
- **Accuracy**: overall + per-task + per-scenario
- **Rewrite trigger rate**: % questions where controller issues REWRITE
- **Rewrite success rate**: % REWRITEs that improve answer utility
- **Evidence precision**: % selected evidence that is answer-relevant

### 4.4 Ablations
1. w/o answer-utility reward (replace with binary correct/incorrect)
2. w/o online retrieval (REWRITE reward = constant, no real search)
3. w/o tier-aware curriculum (uniform sampling)
4. w/o forced-prefix exploration (natural rollouts only)
5. w/o noise-robust advantage (vanilla z-norm)
6. w/o evidence selection (keep all top-5)
7. w/o query rewriting (ACCEPT only)

---

## 5. Results (待填充)

### 5.1 Main Results (Table 1)
[EndoBench overall + per-scenario accuracy]

### 5.2 Ablation Study (Table 2)
[Each component's contribution]

### 5.3 Analysis
- Rewrite behavior: trigger rate, success rate, examples
- Training dynamics: probe mean_u curve (greedy held-out evaluation)
- Case studies: retrieval failure → successful rewrite → correct answer

---

## 6. Discussion & Limitations
- Controller adds latency (1 extra retrieval round for ~X% of questions)
- Domain-specific: trained on GI endoscopy, generalization to radiology/pathology needs validation
- Frozen generator assumption: if generator is weak, answer-utility signal is noisy
- Single retrieval index: cross-institution knowledge not covered

---

## 7. Conclusion
[Summary + future work: multi-index routing, end-to-end retriever fine-tuning, clinical deployment]

---

## Figures Plan
| Figure | Content | Source |
|---|---|---|
| Fig. 1 | System overview: frozen RAG + trainable controller loop | 手绘/draw.io |
| Fig. 2 | Training pipeline: tier classification → GRPO with online retrieval | 代码流程 |
| Fig. 3 | Training curves: probe mean_u over updates (greedy held-out) | grpo_v4_curves.png |
| Fig. 4 | Case study: bad retrieval → rewrite → good retrieval → correct answer | 从评测结果挑 |
| Fig. 5 | Ablation bar chart | 实验数据 |

---

## Key Selling Points for Reviewers
1. **Clean attribution**: only the controller is trained → all gains are from the controller
2. **Principled reward**: answer-utility is directly tied to downstream task, not a proxy
3. **Closed-loop training**: REWRITE is evaluated by actually executing retrieval (not simulated)
4. **Practical**: 4B model, adds <1s latency, works on top of any existing RAG system
5. **Reproducible**: training data is source-grounded (not from benchmark), code will be released
