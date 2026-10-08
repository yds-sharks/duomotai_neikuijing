# 修改记录日志

> 本文档记录所有代码修改，便于快速理解修改方向和内容。

---

## 第一部分：data_construction/ 数据构造流水线（Stage 1）

### 修改背景
数据构造流水线从 v0.2 升级到 v0.3，核心变化：
- **Verifier 盲答**：Verifier 不再看到 candidate_answer，独立判断正确答案，然后比对
- **答案质量门**：新增多个硬性检查（answer_text 泄露、文件存在性、pdf_context_pages 长度）
- **代码三件套**：所有脚本添加原子写、计时/ETA、断点续传
- **删除死字段**：移除不存在的 `evidence_text` 字段检查

---

### 1. qa_stage1_common.py
**新增函数**：
- `write_jsonl_atomic(path, records)` - 原子写入 JSONL，防止进程中断导致文件损坏
- `write_json_atomic(path, data)` - 原子写入 JSON
- `row_get(row, key, default)` - 安全获取 dict/Row 字段，兼容 DB schema 漂移

**原因**：代码三件套要求，确保数据写入的原子性和容错性。

---

### 2. prepare_qa_source_pool.py
**修改内容**：
- 动态 SELECT：先 `PRAGMA table_info` 查 schema，只 SELECT 存在的列
- 原子写：`json.dump(open("w"))` → `write_jsonl_atomic()`
- 计时/ETA：每 2000 条打印 rate/eta，report 加 elapsed_s

**原因**：
- 兼容性：不同版本的 DB schema 可能不同，动态 SELECT 避免硬编码报错
- 容错：原子写防止进程中断导致文件损坏
- 可观测性：计时/ETA 便于监控长时间任务

---

### 3. route_qa_source_pool.py
**修改内容**：
- 原子写：`json.dump(open("w"))` → `write_json_atomic()`
- 计时：report 加 elapsed_s

**原因**：代码三件套要求，与 prepare_qa_source_pool.py 保持一致。

---

### 4. qa_api_prompts.py
**修改内容**：
- QA_VERIFICATION_SYSTEM_PROMPT：明确要求 verifier **独立作答**，不看到 candidate_answer
- QA_VERIFICATION_USER_TEMPLATE：新增字段（multi_organ_ambiguity, option_quality, text_leakage, all_options_same_granularity, multiple_correct_options）
- render_verification_user_prompt：删除 candidate_answer / candidate_answer_text / evidence_basis 的展示

**原因**：v0.3 设计要求 verifier 盲答比对，避免 verifier 直接抄生成器的答案。新增字段覆盖答案质量门的多个维度。

---

### 5. qa_api_common.py（新建）
**内容**：共享 API 工具模块
- `load_api_config()` - 加载 API 配置
- `build_image_messages()` - 构建带图片的 messages，**必须传图像像素**
- `parse_json_content()` - 健壮的 JSON 提取（容忍代码围栏、正则 fallback）
- `extract_usage()` / `accumulate_usage()` - API usage 统计
- `load_done_ids()` / `StreamWriter` - 断点续传 + 流式写入

**原因**：抽取公共逻辑，避免 generate_qa_candidates_with_api.py 和 verify_qa_candidates_with_api.py 重复代码。

**关键设计**：
- `build_image_messages()` 强制传图像像素，否则"图像依赖"判定失效
- `StreamWriter` 带锁 append + fsync，支持多线程并发写入

---

### 6. generate_qa_candidates_with_api.py（新建）
**内容**：生成脚本
- 多线程 API 调用（ThreadPoolExecutor）
- 硬约束校验：选项完整性、候选答案合法性、image_dependency != "low"
- 状态追踪：generation_status ∈ {no_image, api_error, invalid_json, source_rejected, invalid:*, accepted}
- 断点续传：load_done_ids + StreamWriter
- 进度监控：每 25 条打印 rate/eta/usage

**原因**：实现 Stage 1 的 API 生成环节，复用 qa_api_prompts.py 的 prompt。

**关键设计**：
- **必须传图像像素**：否则"图像依赖"判定失效
- 硬约束校验：过滤掉不符合要求的候选，减少下游验证负担
- 状态机：generation_status 统一表示各种失败/成功状态，便于统计和调试

---

### 7. verify_qa_candidates_with_api.py（新建）
**内容**：验证 + 组装脚本
- 盲答 verifier：verifier 不看到 candidate_answer，独立判断正确答案
- 质量门：verifier_answer == candidate_answer + 多个硬性检查
- 去重：_pick_best() 按 image_dependency + question_naturalness 排序，保留最佳
- 组装：按设计 §6 schema 组装 qa_gold.jsonl + train/dev/internal_test 拆分
- 报告：construction_report.json 包含 status_counts + api_usage_total

**原因**：实现 Stage 1 的验证和最终数据组装。

**关键设计**：
- **盲答比对**：verifier 独立判断，避免橡胶图章
- 质量门：多个硬性检查确保数据质量
- 去重：避免同一 source 产生多道题导致数据偏斜

---

### 8. validate_qa_gold_dataset.py（重写 + 改名）
**修改内容**：
- 删除死字段：`evidence_text` → 检查 caption_or_pair_text / image_local_context_text / pdf_context_text 至少一个非空
- 新增 query_type 允许集合检查
- 新增文件存在性检查：query_image_path + origin_pdf
- 新增 pdf_context_pages 长度检查（1-2 页）
- 新增 answer_text_in_question 泄露检查
- 改名：`validate_agentic_mcq_dataset.py` → `validate_qa_gold_dataset.py`

**原因**：
- `evidence_text` 是死字段，新 schema 已拆分为三个具体字段
- 新增检查覆盖更多数据质量问题
- 改名：旧名 "agentic_mcq_dataset" 是 v0.2 命名，新流水线产出是 "qa_gold"

---

### 9. README.md
**修改内容**：
- step 4/5：去掉 (TODO)
- step 6：`validate_agentic_mcq_dataset.py` → `validate_qa_gold_dataset.py`
- Scripts 列表：加 qa_api_common.py
- Example 命令：更新路径

**原因**：同步文档与代码状态。

---

### 10. fixes_zh.md（新建）
**内容**：所有修改的详细原因说明

**原因**：便于作者快速理解每个修改的动机和背景。

---

## 第二部分：code/ 推理与训练代码（v0.2 → v0.3）

### 修改背景
v0.3 架构核心变化：
- **删除独立 reranker**：不再用 Qwen3-VL-Embedding + PPR，检索结果直接给 agent
- **Agent 一次输出**：keep/drop（逐条价值判断）+ ACCEPT/REWRITE（是否改写）
- **Answer-utility 奖励**：基于 generator logprob 计算答案概率提升，而非 evidence_hit
- **训练流程**：SFT（keep/drop 软标签 + GPT rollout 模仿）→ GRPO（answer-utility 优化）

---

### 1. rerank_adapter.py（删除）
**原因**：v0.3 架构无独立 reranker，检索结果直接给 agent 做价值判断。

**影响**：
- agentic_rag_pipeline.py 不再调用 PPRReranker
- run_gpt_agent_rollout_smoke.py 不再调用 NoOpReranker / PPRReranker

---

### 2. gpt_agent_adapter.py（重写）
**修改内容**：
- v0.2：只输出 ACCEPT/REWRITE + rewrite_candidates
- v0.3：输出 keep/drop + ACCEPT/REWRITE(+rewrite_query)

**新输出 schema**：
```json
{
  "keep": [0, 2, 4],
  "drop": [1, 3],
  "action": "ACCEPT" | "REWRITE",
  "rewrite_query": "...",
  "reason": "..."
}
```

**原因**：v0.3 要求 agent 对检索结果做逐条价值判断（keep/drop），然后决定是否改写。

**关键设计**：
- `_parse_keep_drop()`：校验并规整 keep/drop 字段，处理缺失/重叠/越界索引
- `GPTRewriteAgent = GPTAgent`：兼容旧名，避免调用方立即崩溃

---

### 3. generator_adapter.py（重写）
**修改内容**：
- 新增 `generate_with_logprobs()`：返回完整 logprobs，用于 answer-utility 奖励计算
- 新增 `extract_option_probability()`：从 logprobs 中提取某个选项字母的概率

**原因**：v0.3 奖励基于 generator logprob 计算答案概率提升，需要 generator 输出 logprobs。

**关键设计**：
- `generate_with_logprobs()` 调用 vLLM API 的 logprobs 接口（OpenAI-compatible）
- `extract_option_probability()` 找到生成内容中**第一个**与选项字母匹配的 token，取其 logprob 转概率

---

### 4. reward_model.py（重写）
**修改内容**：
- v0.2：`score_candidate()` 基于 evidence_hit（gold doc/page 是否进 top-k）
- v0.3：`score_answer_utility()` 基于 answer-utility（generator logprob 概率提升）

**新奖励公式**：
```
r(a) = P_G(a* | q, I_q, E_a) − P_G(a* | q, I_q, ∅)
```

**原因**：
- evidence_hit 是相关性代理信号，与"答案是否正确"弱相关
- answer-utility 直接衡量"证据是否提升答案概率"，与最终目标一致
- 护城河：避免退化为"学一个检索 query 改写器"，保持医学特异性

**关键设计**：
- `score_answer_utility()`：主奖励函数
- `measure_p_correct()`：用冻结 generator 测量给定证据集下正确选项概率
- `attach_group_advantages()`：GRPO 组内相对优势 A=(r-mean)/std
- `evidence_hit()`：保留为离线分析指标，**不进 reward**
- `score_candidate()`：保留签名兼容旧调用，内部转调 `score_answer_utility()`

**轻量约束项**：
- `leakage_penalty()`：rewrite query 含答案字母或直接泄露 gold answer 扣分
- `unnecessary_rewrite_penalty`：原 evidence 已把 utility 抬到较高水平（>=0.15），仍 REWRITE 扣分
- `length_penalty`：rewrite query 过长扣分

---

### 5. agentic_rag_pipeline.py（重写）
**修改内容**：
- v0.2：query → retrieve → **PPR rerank** → generate
- v0.3：query → retrieve → **agent(keep/drop+ACCEPT/REWRITE)** → generate（REWRITE 则重检索，最多 T 轮）

**关键流程**：
```
for round in range(max_rounds):
    evidence = retrieve(query, exclude_ids=suppressed_ids)
    agent_out = agent.decide(evidence)
    kept = apply_keep_drop(evidence, agent_out["keep"])
    suppressed_ids |= collect_ids(evidence, agent_out["drop"])  # 召回层抑制
    if agent_out["action"] == "ACCEPT":
        break
    query = agent_out["rewrite_query"]
response = generator.generate(kept)
```

**原因**：v0.3 架构无独立 reranker，agent 直接对检索结果做价值判断，并根据 ACCEPT/REWRITE 决定是否重检索。

**关键设计**：
- `suppressed_ids`：在**召回层**抑制 dropped/已见证据，避免多轮打转
- `apply_keep_drop()`：按 agent 的 keep 索引取 kept 证据子集
- `collect_evidence_ids()`：收集指定索引的证据 ID，用于 suppression
- 多轮循环：最多 max_rounds 轮，REWRITE 则用 rewrite_query 重检索

---

### 6. run_gpt_agent_rollout_smoke.py（重写）
**修改内容**：
- v0.2：调用 NoOpReranker / PPRReranker，reward 基于 evidence_hit
- v0.3：无 reranker，reward 基于 answer-utility

**关键流程**：
```
p_baseline = measure_p_correct(generator, evidence=None)  # P_G(a*|∅)
for round in range(max_rounds):
    evidence = retrieve(query, exclude_ids=suppressed_ids)
    agent_out = agent.decide(evidence)
    kept = apply_keep_drop(evidence, agent_out["keep"])
    p_with_kept = measure_p_correct(generator, evidence=kept)  # P_G(a*|E)
    score = score_answer_utility(p_with_kept, p_baseline)
    suppressed_ids |= collect_ids(evidence, agent_out["drop"])
    if agent_out["action"] == "ACCEPT":
        break
    query = agent_out["rewrite_query"]
reward_group = attach_group_advantages(candidate_records)  # GRPO
```

**原因**：v0.3 奖励基于 answer-utility，需要测量 baseline P_G(a*|∅) 和每轮的 P_G(a*|E)。

**关键设计**：
- `measure_p_correct()`：用冻结 generator 测量给定证据集下正确选项概率
- `p_baseline`：整组共享的 baseline，所有轮次的 reward 都减去它
- `attach_group_advantages()`：GRPO 组内相对优势，用于 policy gradient 训练

---

### 7. agentic_runtime_config.json（重写）
**修改内容**：
- v0.2：包含 `rerank` 配置段（PPR 参数）
- v0.3：删除 `rerank` 段，新增 `routing` 段（密度感知路由）和 `agent` 段

**新配置**：
```json
{
  "routing": {
    "enabled": true,
    "router_checkpoint": "...",
    "density_levels": ["L0", "L1", "L2", "L3", "L4"],
    "image_dependency_levels": ["R1", "R2", "R3"]
  },
  "generator": {
    "logprobs": true,
    "top_logprobs": 5
  },
  "agent": {
    "policy_model": "Qwen3-4B",
    "max_rounds": 2,
    "keep_drop_threshold": 0.0,
    "accept_threshold": 0.6
  }
}
```

**原因**：v0.3 架构无独立 reranker，但需要密度感知路由和 agent 配置。

**关键配置**：
- `generator.logprobs = true`：开启 logprobs 输出，用于 answer-utility 奖励
- `agent.max_rounds = 2`：最多 2 轮 REWRITE 循环

---

### 8. run_agentic_rag_smoke.sh（重写）
**修改内容**：
- v0.2：包含 `--no-ppr-rerank` 参数
- v0.3：删除 `--no-ppr-rerank`，新增 `--max-rounds 2`

**原因**：v0.3 架构无独立 reranker，不需要 `--no-ppr-rerank` 开关。

---

### 9. retrieval_adapter.py（修改）
**修改内容**：
- 新增 `exclude_ids` 参数：在**召回层**抑制已见/dropped 的证据 ID

**关键设计**：
- `_filter_excluded()`：过滤掉 exclude_ids 中的证据
- `search_text()` / `search_image()`：多取一些（fetch_k += len(exclude_ids)），过滤后仍可能有足够数量

**原因**：v0.3 要求多轮检索单调探索新证据，避免在同一批垃圾上打转。

---

## 总结

### data_construction/ 核心改动
1. **Verifier 盲答**：避免橡胶图章，确保答案质量
2. **答案质量门**：多个硬性检查（泄露、文件存在性、pdf_context_pages 长度）
3. **代码三件套**：原子写、计时/ETA、断点续传
4. **删除死字段**：`evidence_text` → 三个具体字段

### code/ 核心改动
1. **删除独立 reranker**：检索结果直接给 agent
2. **Agent 一次输出**：keep/drop + ACCEPT/REWRITE
3. **Answer-utility 奖励**：基于 generator logprob，而非 evidence_hit
4. **多轮循环**：REWRITE 则重检索，召回层抑制 dropped/已见证据
5. **训练流程**：SFT（keep/drop 软标签 + GPT rollout 模仿）→ GRPO（answer-utility 优化）

---

## 下一步

1. **验证 data_construction/**：在真实环境跑 Stage 1 pipeline，生成 qa_gold_4000.jsonl
2. **验证 code/**：在 4 张 A6000 上跑 smoke test，验证 v0.3 流程
3. **实现密度感知路由**：retrieval_adapter.py 需要集成 routing 模块
4. **实现 SFT 训练**：用 qa_gold_4000.jsonl + GPT rollout 数据训练 agent policy
5. **实现 GRPO 训练**：用 answer-utility 奖励优化 policy
