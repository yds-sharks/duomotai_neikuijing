# v0.3 升级完成总结

> 时间：2026-07-12  
> 范围：`data_construction/` (Stage 1 数据构造) + `code/` (推理与训练代码)

---

## 一、核心架构变化（v0.2 → v0.3）

### 1.1 删除独立 reranker
- **v0.2**：query → retrieve → **Qwen3-VL-Embedding + PPR rerank** → agent → generate
- **v0.3**：query → retrieve → **agent (keep/drop + ACCEPT/REWRITE)** → generate

**原因**：reranker 是"相关性排序"，而 agent 的 keep/drop 是"价值判断"，两者目标不同。直接用 agent 做价值判断更简洁、更符合 answer-utility 优化目标。

### 1.2 Agent 一次输出
- **v0.2**：agent 只输出 `ACCEPT/REWRITE` + `rewrite_candidates`
- **v0.3**：agent 一次输出 `keep/drop`（逐条价值判断）+ `ACCEPT/REWRITE`（是否改写）

**输出 schema**：
```json
{
  "keep": [0, 2, 4],
  "drop": [1, 3],
  "action": "ACCEPT" | "REWRITE",
  "rewrite_query": "...",
  "reason": "..."
}
```

### 1.3 Answer-utility 奖励
- **v0.2**：`evidence_hit`（gold doc/page 是否进 top-k）
- **v0.3**：`answer_utility`（generator logprob 概率提升）

**奖励公式**：
```
r(a) = P_G(a* | q, I_q, E_a) − P_G(a* | q, I_q, ∅)
```

**原因**：
- `evidence_hit` 是相关性代理信号，与"答案是否正确"弱相关
- `answer_utility` 直接衡量"证据是否提升答案概率"，与最终目标一致
- **护城河**：避免退化为"学一个检索 query 改写器"，保持医学特异性

### 1.4 训练流程
- **v0.2**：SFT（GPT rollout 模仿）
- **v0.3**：SFT（keep/drop 软标签 + GPT rollout 模仿）→ GRPO（answer-utility 优化）

**数据来源**：
- **keep/drop 软标签**：用冻结 generator 测每条证据的边际贡献（自动标注，无需人工）
- **GPT rollout**：GPT-4o 模拟 agent 行为，生成 ACCEPT/REWRITE + rewrite_query
- **answer-utility**：GRPO 组内相对优势 A=(r-mean)/std

---

## 二、data_construction/ 改动（Stage 1 数据构造）

### 2.1 核心改动
1. **Verifier 盲答**：Verifier 不看到 `candidate_answer`，独立判断正确答案，然后比对
2. **答案质量门**：新增多个硬性检查（answer_text 泄露、文件存在性、pdf_context_pages 长度）
3. **代码三件套**：所有脚本添加原子写、计时/ETA、断点续传
4. **删除死字段**：移除不存在的 `evidence_text` 字段检查

### 2.2 文件清单
| 文件 | 状态 | 说明 |
|------|------|------|
| `qa_stage1_common.py` | 新增 | 原子写、安全字段访问 |
| `prepare_qa_source_pool.py` | 重写 | 动态 SELECT、原子写、计时/ETA |
| `route_qa_source_pool.py` | 重写 | 原子写、计时 |
| `qa_api_prompts.py` | 重写 | Verifier 盲答 prompt、新增质量门字段 |
| `qa_api_common.py` | 新建 | 共享 API 工具（config、image messages、JSON 提取、usage、resume） |
| `generate_qa_candidates_with_api.py` | 新建 | 生成脚本（多线程、硬约束校验、状态追踪、断点续传） |
| `verify_qa_candidates_with_api.py` | 新建 | 验证+组装脚本（盲答比对、质量门、去重、组装） |
| `validate_qa_gold_dataset.py` | 重写+改名 | 删除死字段、新增 5 项检查、改名 |
| `README.md` | 更新 | 同步文档与代码状态 |
| `fixes_zh.md` | 新建 | 所有修改的详细原因说明 |

### 2.3 关键设计
- **必须传图像像素**：`build_image_messages()` 强制传图像像素，否则"图像依赖"判定失效
- **盲答比对**：Verifier 独立判断正确答案，避免橡胶图章
- **答案质量门**：多个硬性检查确保数据质量
- **去重**：`_pick_best()` 按 image_dependency + question_naturalness 排序，保留最佳
- **断点续传**：`load_done_ids()` + `StreamWriter` 支持多线程并发写入

---

## 三、code/ 改动（推理与训练代码）

### 3.1 核心改动
1. **删除独立 reranker**：检索结果直接给 agent
2. **Agent 一次输出**：keep/drop + ACCEPT/REWRITE
3. **Answer-utility 奖励**：基于 generator logprob，而非 evidence_hit
4. **多轮循环**：REWRITE 则重检索，召回层抑制 dropped/已见证据
5. **训练流程**：SFT（keep/drop 软标签 + GPT rollout 模仿）→ GRPO（answer-utility 优化）

### 3.2 文件清单
| 文件 | 状态 | 说明 |
|------|------|------|
| `rerank_adapter.py` | 删除 | v0.3 架构无独立 reranker |
| `gpt_agent_adapter.py` | 重写 | v0.3 输出 keep/drop + ACCEPT/REWRITE |
| `generator_adapter.py` | 重写 | 新增 `generate_with_logprobs()`、`extract_option_probability()` |
| `reward_model.py` | 重写 | `score_answer_utility()` + `measure_p_correct()` + `attach_group_advantages()` |
| `agentic_rag_pipeline.py` | 重写 | 无 reranker、多轮循环、召回层抑制 |
| `run_gpt_agent_rollout_smoke.py` | 重写 | 无 reranker、answer-utility 奖励、GRPO 组内优势 |
| `agentic_runtime_config.json` | 重写 | 删除 `rerank` 段、新增 `routing` + `agent` 段 |
| `run_agentic_rag_smoke.sh` | 重写 | 删除 `--no-ppr-rerank`、新增 `--max-rounds` |
| `retrieval_adapter.py` | 修改 | 新增 `exclude_ids` 参数（召回层抑制） |

### 3.3 关键设计
- **召回层抑制**：`exclude_ids` 在召回层过滤 dropped/已见证据，避免多轮打转
- **Generator logprobs**：`generate_with_logprobs()` 调用 vLLM API 的 logprobs 接口
- **Answer-utility 奖励**：`score_answer_utility()` 计算 P_G(a*|E) - P_G(a*|∅)
- **GRPO 组内优势**：`attach_group_advantages()` 计算 A=(r-mean)/std
- **轻量约束项**：`leakage_penalty()`、`unnecessary_rewrite_penalty`、`length_penalty`

---

## 四、下一步工作

### 4.1 验证 data_construction/
1. 在真实环境跑 Stage 1 pipeline，生成 `qa_gold_4000.jsonl`
2. 检查 `construction_report.json` 中的 status_counts + api_usage_total
3. 人工抽检 10-20 条数据，验证答案质量

### 4.2 验证 code/
1. 在 4 张 A6000 上跑 smoke test：
   ```bash
   cd code/
   bash run_agentic_rag_smoke.sh
   ```
2. 检查输出 JSON，验证 v0.3 流程（无 reranker、多轮循环、answer-utility 奖励）
3. 人工抽检 5-10 条数据，验证 agent 行为（keep/drop + ACCEPT/REWRITE）

### 4.3 实现密度感知路由
- `retrieval_adapter.py` 需要集成 routing 模块
- 根据 query 的密度级别（L0-L4）和图像依赖级别（R1-R3）调整检索策略

### 4.4 实现 SFT 训练
- 用 `qa_gold_4000.jsonl` + GPT rollout 数据训练 agent policy
- 训练目标：keep/drop 软标签 + ACCEPT/REWRITE + rewrite_query

### 4.5 实现 GRPO 训练
- 用 answer-utility 奖励优化 policy
- GRPO 组内相对优势 A=(r-mean)/std
- 对比 SFT-only vs SFT+GRPO 的效果

---

## 五、关键文档

| 文档 | 说明 |
|------|------|
| `CHANGES_LOG.md` | 所有代码修改的详细记录（含原因、影响、设计） |
| `data_construction/fixes_zh.md` | data_construction/ 修改的详细原因说明 |
| `data_construction/README.md` | Stage 1 数据构造流水线说明 |
| `code/README.md` | 推理与训练代码说明 |
| `medalign_paper_framework_zh.md` | v0.3 架构设计（权威来源） |
| `data_construction/data_construction_design_zh.md` | Stage 1 数据构造设计 |

---

## 六、4 张 A6000 部署建议

### 6.1 硬件分配
- **GPU 0**：Generator（Qwen3-VL-8B，vLLM 部署，开启 logprobs）
- **GPU 1**：Agent policy（Qwen3-4B，训练 + 推理）
- **GPU 2-3**：Retrieval（BGE-M3 text + Jina-CLIP-v2 image）

### 6.2 部署步骤
1. 启动 vLLM server（Generator）：
   ```bash
   bash code/run_local_qwen3vl_server.sh
   ```
2. 加载 Retrieval 模型（BGE-M3 + Jina-CLIP-v2）
3. 运行 smoke test：
   ```bash
   bash code/run_agentic_rag_smoke.sh
   ```
4. 运行 GPT agent rollout：
   ```bash
   python3 code/run_gpt_agent_rollout_smoke.py
   ```

### 6.3 关键配置
- `agentic_runtime_config.json`：
  - `generator.logprobs = true`：开启 logprobs 输出
  - `generator.top_logprobs = 5`：返回 top-5 logprobs
  - `agent.max_rounds = 2`：最多 2 轮 REWRITE 循环
  - `routing.enabled = true`：启用密度感知路由（待实现）

---

## 七、常见问题

### Q1: 为什么删除独立 reranker？
**A**: reranker 是"相关性排序"，而 agent 的 keep/drop 是"价值判断"，两者目标不同。直接用 agent 做价值判断更简洁、更符合 answer-utility 优化目标。

### Q2: 为什么用 answer-utility 而非 evidence_hit？
**A**: `evidence_hit` 是相关性代理信号，与"答案是否正确"弱相关。`answer_utility` 直接衡量"证据是否提升答案概率"，与最终目标一致。这是护城河，避免退化为"学一个检索 query 改写器"。

### Q3: 为什么 Verifier 要盲答？
**A**: 避免橡胶图章。如果 Verifier 看到 `candidate_answer`，可能会直接抄答案，无法真正验证答案质量。盲答比对更可靠。

### Q4: 为什么要在召回层抑制 dropped/已见证据？
**A**: 避免多轮打转。如果 agent REWRITE 后重检索，但检索结果还是同一批垃圾，就没有意义了。召回层抑制确保多轮检索单调探索新证据。

### Q5: 为什么 SFT → GRPO 而非直接 RL？
**A**: 
- **SFT**：用 keep/drop 软标签 + GPT rollout 数据做模仿学习，快速收敛到合理行为
- **GRPO**：用 answer-utility 奖励优化 policy，突破 SFT 的天花板（SFT 只能模仿 GPT，GRPO 可以发现更优策略）

---

## 八、联系方式

如有问题，请查阅：
- `CHANGES_LOG.md`：所有代码修改的详细记录
- `data_construction/fixes_zh.md`：data_construction/ 修改的详细原因说明
- `medalign_paper_framework_zh.md`：v0.3 架构设计（权威来源）
