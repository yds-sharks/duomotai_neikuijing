# AgenticRL: Training RAG Controllers via Internal Probability Probing and Online Retrieval

## Abstract

Retrieval-augmented generation (RAG) controllers must learn whether to retain a subset of retrieved evidence or reformulate the query, but their training commonly relies on either coarse terminal correctness rewards or teacher-action supervision. The former provides weak credit assignment, whereas the latter may constrain exploration to teacher-generated trajectories. We present AgenticRL, a reinforcement learning framework for training a lightweight controller for a frozen multimodal RAG system. For each controller rollout, AgenticRL defines evidence utility as the difference between the frozen generator's log-probability of the gold answer with the resulting evidence and its log-probability under a no-evidence baseline. This generator-derived reward is continuous, is computed directly from model logits without sampling an answer, and evaluates the consequences of controller actions rather than their agreement with teacher actions. For query rewriting, each rewritten query is executed against the retrieval index used by the system during training, and the resulting evidence is scored by the frozen generator, closing the loop between rewriting, retrieval, and reward computation. We further introduce a tier-aware curriculum based on retrieval need, together with noise-robust group-relative advantage estimation that suppresses updates from rollout groups with near-identical rewards. We train AgenticRL on an endoscopy corpus constructed without using EndoBench question-answer pairs for training and evaluate it on EndoBench, which contains 6,832 clinically validated visual question answering pairs across four endoscopic scenarios and 12 clinical tasks. Experiments show that AgenticRL improves over both the frozen RAG pipeline and a proprietary multimodal controller baseline, while ablations demonstrate that generator-derived utility rewards and online retrieval feedback each contribute to the gains.

---

## 1. Introduction

Retrieval-augmented generation (RAG) has become the standard approach for grounding large language model outputs in external knowledge [citations]. In a typical RAG pipeline, a query is issued to a retrieval system, the top-ranked passages are passed to a generator, and an answer is produced. This one-shot design works well when initial retrieval succeeds—but offers no recourse when it fails.

Retrieval failures are common in knowledge-intensive domains. Short or image-dependent queries provide weak text retrieval signals. Domain-specific collections contain visually similar but semantically distinct entries (e.g., endoscopic images of different organs). And the passages most relevant by embedding similarity are not always those that help the generator answer correctly. Conventional RAG systems, including those with rerankers, can only reorder existing candidates—they cannot change the retrieval distribution itself.

A natural response is to train an agent that actively manages the retrieval process: deciding which evidence to retain, and when evidence is insufficient, reformulating the query for a second retrieval round. Reinforcement learning (RL) provides a principled framework for this, but existing RL approaches to RAG controller training face a fundamental dilemma:

- **Terminal correctness rewards** (e.g., "did the generator answer correctly?") provide weak credit assignment. They collapse the generator's graded assessment of evidence into a single bit, making it impossible to distinguish "this evidence raised confidence from 30% to 85%" from "this evidence barely changed anything."

- **Teacher-action supervision** sidesteps reward sparsity by having the student imitate a strong teacher's decisions. But imitation constrains exploration to teacher-generated trajectories—the student learns behavioral patterns without developing its own model of how evidence quality interacts with the retrieval system.

We propose **AgenticRL**, an RL framework that resolves this dilemma through two key mechanisms. First, instead of terminal correctness, we derive the reward from the frozen generator's **internal log-probability** of the gold answer, measured with and without the selected evidence. This **generator-relative evidence utility** is continuous, deterministic, and directly reflects how much the evidence influenced the generator's prediction. Second, for query rewriting actions, the rewritten query is executed against the **real vector index** during training, and the newly retrieved evidence is rescored by the generator—closing the loop between rewriting, retrieval, and reward computation.

The agent explores its own action space—selecting evidence subsets and formulating rewrite queries—guided by the consequences of its actions as measured through the generator's internal state, not by agreement with any teacher's choices. Training is stabilized by a **retrieval-need curriculum** that allocates exploration budget according to the evidence requirements of each training state, and by **variance-gated advantage estimation** that suppresses gradient updates from groups whose rewards are too similar to carry reliable ranking signal.

We validate AgenticRL by training a lightweight controller on a frozen multimodal medical RAG stack and evaluating on EndoBench [citation], a clinically validated endoscopy VQA benchmark. Experiments show consistent improvements over both the frozen RAG baseline and a GPT-4o controller, with ablations confirming that each core component contributes materially.

**Contributions.**

1. **Generator-relative evidence utility reward.** We define a continuous, deterministic reward that measures how much the selected evidence shifts the frozen generator's log-probability of the gold answer, providing dense credit assignment without requiring a separate reward model or binary correctness labels.

2. **Online retrieval-in-the-loop for query rewriting.** Rewritten queries are executed against the deployed vector index during training, and the newly retrieved evidence is rescored by the frozen generator. This closes the reward loop for query reformulation—ensuring that rewrite quality is assessed by real retrieval outcomes, not proxies.

3. **Retrieval-need curriculum and variance-gated advantage estimation.** We classify training states by their evidence requirements and allocate exploration accordingly, and we suppress policy updates from groups whose rewards lack sufficient spread to provide reliable ranking signal.

---

## 2. Related Work

### 2.1 Retrieval-Augmented Generation

RAG [Lewis et al., 2020] grounds LLM outputs in retrieved evidence. Subsequent work improves retrieval through better encoders [citations], hybrid sparse-dense methods [citations], and reranking [citations]. In medicine, MedRAG [citation] applies RAG to clinical QA. These systems follow a fixed retrieve-then-generate pipeline without self-recovery when retrieval fails.

### 2.2 Adaptive and Agentic RAG

Self-RAG [citation] trains the generator to decide when to retrieve. CRAG [citation] adds a retrieval evaluator for corrective actions. Adaptive-RAG [citation] routes queries by complexity. MA-RAG [citation] uses a multi-round agent for medical RAG but relies on proprietary models without RL training. ReasonRAG [citation] explores process-level rewards but does not execute real retrieval during training. Our work differs in using generator-derived utility as a dense reward and in closing the loop between rewriting and real retrieval.

### 2.3 RL for LLM Agents

RLHF [citation] uses learned reward models from human preferences. GRPO [citation] simplifies PPO via group-relative advantages. Dr. GRPO [citation] identifies z-normalization issues in low-variance groups. Search-R1 [citation] applies RL to search decisions with binary outcome rewards. Our approach differs in reward granularity (continuous log-probability vs. binary correctness) and in evaluating actions through real retrieval execution rather than cached or simulated results.

### 2.4 Knowledge Distillation and Imitation

Distillation [citation] transfers knowledge by matching teacher outputs. In agentic settings, imitation learning constrains the student to teacher trajectories. Our method uses a strong model's internal probability as a *reward signal*—not as a behavior template. The student explores actions the teacher never took, and is rewarded based on consequences, not conformity.

---

## 3. System Overview

The architecture consists of a **frozen front-end** (retriever + generator) and a **trainable controller**.

### 3.1 Frozen Front-End

**Multimodal Retriever.** Given a query (text + optional image), the retriever searches text and image vector indexes built on a domain-specific medical corpus. The two channels are fused with adaptive weights (Section 4.1), and the top-K passages (K=5) form the candidate evidence set.

**Hierarchical Index.** The vector store is partitioned by anatomical site and content type (Section 4.2), ensuring retrieval operates within clinically coherent subspaces.

**Frozen Generator.** A vision-language model answers multiple-choice questions given question, options, optional query image, and selected evidence. During training, it also provides the log-probability signals used for reward computation (Section 5.2).

### 3.2 Trainable Controller

A lightweight LM observes: question, options, query image (if present), and the current top-K evidence with retrieval scores. It outputs a structured JSON action:

```json
{"selected_evidence": [0, 2], "action": "ACCEPT"}
{"selected_evidence": [1], "action": "REWRITE", "rewrite_query": "..."}
```

- **ACCEPT**: Selected evidence is passed to the generator for the final answer.
- **REWRITE**: The rewritten query triggers a new retrieval round. New top-K candidates are presented to the controller (up to 2 rounds).

Only the controller is trained. Retriever, index, and generator remain frozen.

---

## 4. Retrieval Environment Design

We describe the retrieval front-end that provides the training and inference environment for the controller. These components address domain-specific retrieval challenges in endoscopy, but are **engineering designs** rather than algorithmic contributions of this work; their purpose is to ensure the controller operates on sufficiently high-quality candidates.

### 4.1 Adaptive Channel Fusion

Text and image retrieval channels carry different information densities. A single endoscopic image can immediately identify the anatomical site, while a text passage may contain extensive background with varying relevance. Fixed fusion weights cannot accommodate this asymmetry across query types. We use a lightweight gating module that predicts per-query fusion weights based on query features (text embedding statistics, image presence/type), adapting the relative contribution of each channel. [TODO: quantitative analysis of density asymmetry]

### 4.2 Hierarchical Index Partitioning

Flat vector indexes ignore anatomical structure. Endoscopic images from different organs can be visually similar (pink mucosal surfaces) but clinically unrelated, causing cross-organ confusion in retrieval. We partition the index by anatomical site (Level 1: stomach, esophagus, colorectum, etc.) and content type (Level 2: lesion features, procedures, landmarks, etc.), restricting retrieval to clinically coherent subspaces. [TODO: before/after recall precision comparison]

---

## 5. AgenticRL: Training Framework

### 5.1 Problem Formulation

At each step, the controller observes state s = (question, options, query_image, evidence_set) and selects action a ∈ {ACCEPT(keep), REWRITE(keep, query')}. The episode terminates on ACCEPT or after the maximum number of retrieval rounds.

### 5.2 Generator-Relative Evidence Utility Reward

#### 5.2.1 Distinguishing Evidence Quality Concepts

We distinguish three notions of evidence quality that are often conflated:

1. **Retrieval relevance** Rel(e, q): Does the evidence match the query in text, visual, or entity space? This is what embedding-based retrievers optimize.
2. **Answer supportiveness** Supp(e, y*): Does the evidence logically or factually support the correct answer? This is a property of the evidence itself, independent of any particular generator.
3. **Generator-relative utility** U_G(E): Does the evidence increase *this specific generator's* probability of the correct answer? This depends on the generator's knowledge, reasoning ability, and how it integrates evidence.

Our reward is based on the third notion. It does not assume that relevant evidence is supportive, nor that supportive evidence will be correctly utilized by the generator. Instead, it measures the actual causal effect of evidence on the generator's prediction.

#### 5.2.2 Reward Definition

```
u(E) = log P_G(y* | q, opts, img, E) − log P_G(y* | q, opts, img, ∅)
```

where P_G is the frozen generator's probability over answer options, y* is the gold answer, and ∅ denotes the empty evidence set.

**Probability computation.** We compute P_G by evaluating the generator's log-probability for each option letter token, then renormalizing over the valid option set:

```
p_G(y_i) = exp(s_i) / Σ_j exp(s_j)
```

where s_i is the logit (or summed log-probability) for the i-th option letter, and the sum runs over all valid options. This restricted normalization avoids length bias across options of different token lengths.

**Consistency conditions.** For each reward computation:
- The query image is identical across evidence and no-evidence conditions.
- Evidence order is fixed (by retrieval score rank).
- The generator's scoring template is identical across all evaluations.
- No sampling is involved; logits are read directly from a single forward pass.

**Key properties:**
- **Continuous**: u ∈ ℝ, capturing gradations of evidence influence.
- **Deterministic**: computed from logits, not sampled text.
- **Uniform scale**: same metric for ACCEPT and REWRITE, enabling direct comparison.
- **Consequence-based**: evaluates what the evidence *does* to the generator, not whether it matches a teacher's choice.

#### 5.2.3 Leakage Penalty

To prevent the controller from embedding the gold answer in the rewrite query, we apply a hard penalty when the query contains the answer letter or exact answer text:

```
r = u(E) − λ_leak · 1[leak detected]
```

We further mitigate reward hacking through: (1) option permutation tests during evaluation (Section 6.4), (2) monitoring rewrite-option lexical overlap, and (3) human audits of high-reward evidence supportiveness.

### 5.3 Online Retrieval-in-the-Loop

For REWRITE actions, the reward must reflect whether the rewritten query actually retrieves better evidence. Scoring the rewrite without executing it produces a constant or proxy reward that carries no signal about rewrite quality.

**Mechanism.** During each GRPO step:
1. Controller generates G rollouts; some contain REWRITE actions with rewritten queries.
2. Each unique rewritten query is encoded and executed against the live vector index (top-k retrieval).
3. Retrieved evidence is merged with kept evidence from the current observation.
4. The frozen generator scores the merged evidence set.
5. Answer utility is computed as the reward.

**Overhead.** ~200ms per unique rewritten query (encoding + ANN search). Identical queries within a group are deduplicated. With G=8 and 2 forced-REWRITE rollouts per group, this adds at most 2 retrieval calls per training state—negligible compared to generation and scoring.

### 5.4 Multi-Stage Training Pipeline

Training proceeds in three stages, progressively shifting from supervised initialization to autonomous RL exploration.

**Stage 1: SFT Cold Start.** A strong proprietary model (GPT-4o) generates demonstration trajectories for a subset of training questions. The controller is supervised fine-tuned on these trajectories to learn the output format (valid JSON, correct field names) and basic decision patterns.

**Stage 2: RFT Self-Distillation.** The SFT model samples its own trajectories. High-reward samples (as measured by the generator-relative utility reward) are selected for another round of fine-tuning. This bridges the gap between the teacher's behavior distribution and the student's own policy distribution.

**Stage 3: GRPO Online RL.** The RFT model is trained with Group Relative Policy Optimization. For each training state, G rollouts are sampled, rewards are computed via generator-relative utility (with online retrieval for REWRITE), and group-relative advantages update the policy with KL regularization.

**Relationship to "non-imitative."** Stages 1–2 use imitation-based initialization. The "non-imitative" property of AgenticRL applies specifically to Stage 3: the RL reward evaluates the *consequences* of the controller's actions (their effect on generator probability), not their *agreement* with teacher actions. The ablation in Section 6.3 isolates the contribution of each stage.

### 5.5 Retrieval-Need Curriculum

Not all training states are equally informative. We perform an offline judge pass on each training question, evaluating the generator under multiple evidence conditions:

- **No evidence**: generator answers with only the question and options.
- **Per-candidate evidence**: each candidate passage is individually provided.
- **Full candidate set**: all top-K candidates are provided.

Based on these evaluations, we classify training states into four categories:

| Category | Criterion | Training role | Sampling |
|----------|-----------|---------------|----------|
| **No-retrieval** | Correct without evidence, high confidence | Minimal training value | Subsample 15% |
| **Selection-solvable** | Initial candidates contain evidence that flips the answer | Train evidence selection | Full |
| **Rewrite-solvable** | Initial candidates insufficient; oracle rewrite retrieves effective evidence | Train query rewriting | Full, upweight |
| **Unsupported** | No evidence (initial or oracle-rewrite) enables correct answer | Exclude from training | Drop |

The **Rewrite-solvable** classification requires a verification step: for questions where initial candidates fail, we test whether an oracle rewrite (constructed from gold metadata) can retrieve evidence that flips the answer. Questions where even oracle rewrites fail are classified as **Unsupported** and excluded, preventing the controller from wasting exploration on unsolvable states.

### 5.6 Forced-Prefix Exploration

Without intervention, RL training may converge to an always-ACCEPT policy—ACCEPT is the "safe" action that avoids rewrite risk. In each GRPO group of G rollouts, n_forced rollouts are generated with a REWRITE prefix. Forced prefix tokens are excluded from the policy gradient; continuation tokens (the actual rewrite query) receive normal gradients. This ensures every group contains genuine REWRITE exploration.

### 5.7 Variance-Gated Advantage Estimation

Standard GRPO normalizes rewards within a group: advantage = (r − μ) / σ. When all rollouts produce nearly identical rewards (σ ≈ 0), z-normalization amplifies noise to full ±1 advantages, causing the policy to update in random directions.

**Spread gate.** If max(reward) − min(reward) < τ_spread, the group is skipped—no policy update. A group where all rollouts achieve similar rewards carries no usable ranking signal.

**Std floor.** For groups that pass the spread gate: advantage = (r − μ) / max(σ, σ_floor). This prevents moderate-spread groups from being over-amplified.

---

## 6. Experiments

### 6.1 Setup

**Benchmark.** EndoBench [citation]: 6,832 clinically validated VQA pairs, 4 endoscopic scenarios, 12 clinical tasks. Multi-choice format (A–F).

**Training data.** 3,200 source-grounded MCQ from our proprietary endoscopy corpus. No benchmark samples used for training (benchmark-disjoint).

**Frozen components.** Retriever: BGE-M3 [citation]; Generator: Qwen3-VL-8B-Instruct [citation].

**Trainable controller.** Qwen3.5-4B [citation] (bfloat16).

### 6.2 Evaluation Metrics

We report four categories of metrics:

**Effectiveness.**
- Overall accuracy
- Macro accuracy across tasks
- Per-scenario accuracy
- Per-task accuracy

**Evidence quality.**
- Evidence Precision@k: fraction of selected evidence that is answer-relevant
- Rewrite evidence improvement: change in evidence utility after rewriting
- Human supportiveness audit: fraction of high-reward evidence judged truly supportive [TODO: N samples]

**Controller behavior.**
- ACCEPT rate / REWRITE trigger rate
- Rewrite success rate: fraction of REWRITEs that improve answer utility
- Unnecessary rewrite rate: REWRITEs on states where initial evidence was sufficient
- Average retrieval rounds per question

**Efficiency.**
- End-to-end latency (controller + retrieval + generator)
- Controller latency / retrieval latency / generator latency
- Training GPU hours

### 6.3 Baselines

| Baseline | Description |
|----------|-------------|
| Closed-book | Generator answers without retrieval |
| RAG-top5 | Top-5 evidence fed directly to generator |
| GPT-4o agent | GPT-4o as controller (same prompt, same tools) |
| SFT controller | After Stage 1 (GPT-4o trajectory SFT) |
| RFT controller | After Stage 2 (self-distillation) |
| Binary-reward GRPO | Same as AgenticRL but reward = 1[correct] (isolates reward granularity) |
| Oracle evidence selector | Selects the evidence subset maximizing U_G(E) from initial candidates (upper bound for selection) |
| **AgenticRL (full)** | After Stage 3 (complete pipeline) |

**Key comparison: Binary-reward GRPO.** This baseline uses identical architecture, training data, rollout count, online retrieval, optimizer, and training steps—the only difference is replacing the continuous utility reward with binary correctness. It directly isolates the value of generator-derived reward granularity.

**Key comparison: Oracle evidence selector.** This baseline computes U_G for every subset of the initial top-K candidates and selects the optimal subset. It measures the headroom available from evidence selection alone, providing context for how much of the theoretically available improvement the controller captures.

### 6.4 Main Results

[TODO: Table 1 — Overall accuracy on EndoBench]

| Method | Overall | Upper GI | Lower GI | Hepatobiliary | Small Bowel |
|--------|---------|----------|----------|---------------|-------------|
| Closed-book | — | — | — | — | — |
| RAG-top5 | — | — | — | — | — |
| GPT-4o agent | — | — | — | — | — |
| SFT controller | — | — | — | — | — |
| RFT controller | — | — | — | — | — |
| Binary-reward GRPO | — | — | — | — | — |
| Oracle selector | — | — | — | — | — |
| **AgenticRL** | — | — | — | — | — |

### 6.5 Ablation Studies

We separate ablations into **core algorithm** and **retrieval environment** to clearly attribute improvements.

**Table 2: Core Algorithm Ablations**

| Variant | Overall Acc | Δ vs Full |
|---------|-------------|-----------|
| Full AgenticRL | — | — |
| Binary outcome reward (no utility probing) | — | — |
| Utility reward, no online retrieval (REWRITE gets proxy reward) | — | — |
| No retrieval-need curriculum (uniform sampling) | — | — |
| No forced-prefix exploration | — | — |
| Vanilla GRPO advantages (no spread gate / std floor) | — | — |
| SFT only → GRPO (skip RFT stage) | — | — |
| Format-only SFT → GRPO (no GPT-4o trajectories) | — | — |

**Table 3: Retrieval Environment Ablations**

| Variant | Overall Acc | Δ vs Full |
|---------|-------------|-----------|
| Dynamic gating + hierarchical index (full) | — | — |
| Fixed fusion + hierarchical index | — | — |
| Dynamic gating + flat index | — | — |
| Fixed fusion + flat index | — | — |

### 6.6 Robustness Analysis

**Option permutation test.** [TODO: shuffle option order; report accuracy variance. Tests whether the controller exploits letter position bias.]

**Answer masking test.** [TODO: mask lexical overlap between rewrite queries and option text; report performance change. Tests whether rewrites leak answer content beyond the leakage penalty.]

**Human evidence audit.** [TODO: sample N high-reward evidence passages; human judges rate whether each truly supports the gold answer. Report supportiveness rate.]

### 6.7 Analysis

**Training dynamics.** [TODO: Figure — probe mean_u over training updates on held-out states, showing greedy evaluation improves]

**Controller behavior.** [TODO: Figure — ACCEPT/REWRITE distribution, rewrite trigger rate by question type, rewrite success rate over training]

**Case study.** [TODO: Figure — retrieval failure → rewrite → successful retrieval → correct answer. Show original query, initial top-5 (wrong evidence), rewrite query, new top-5, final answer.]

---

## 7. Discussion

### 7.1 Limitations

**Single domain and benchmark.** We validate on endoscopy QA only. Generalization to other medical domains (pathology, radiology) or non-medical RAG requires further study.

**Multiple-choice format.** Internal probability probing assumes a finite set of answer options. Extension to open-ended generation requires alternative utility formulations.

**Frozen generator dependence.** The utility reward is generator-relative: it measures evidence's effect on *this specific* generator. If the generator is too weak, the signal is noisy; if too strong, evidence rarely changes predictions.

**Latency.** REWRITE adds one retrieval round and one controller forward pass.

### 7.2 Broader Applicability

AgenticRL may be applicable to other RAG systems in which a frozen generator can assign comparable scores to candidate answers, although its cross-domain generalization remains to be validated. The paradigm does not require medical data—the key requirements are a vector retrieval index, a frozen generator capable of option-level probability estimation, and a training corpus with verifiable answers.

---

## 8. Conclusion

We presented AgenticRL, a reinforcement learning framework for training RAG controllers that bridges the gap between sparse terminal rewards and rigid imitation. By deriving reward from the frozen generator's internal probability distribution and executing real retrieval for query rewriting during training, AgenticRL provides dense, consequence-based training signal while preserving exploration autonomy. A retrieval-need curriculum and variance-gated advantage estimation further stabilize training. Experiments on a clinically validated endoscopy benchmark demonstrate that a lightweight controller trained with AgenticRL consistently outperforms both the frozen RAG pipeline and strong proprietary agents.

---

## References

[TODO: full bibliography]
