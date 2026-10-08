# 数据构造脚本修改说明

本文档记录 agentic data-construction 数据构造流水线中每个修改点及其原因。

> 背景：v0.3 设计（decoupled agent + answer-utility reward）定稿后，数据构造
> 代码与设计文档之间出现了多处不一致。本文档逐一标注修复内容和原因，方便
> 在真实环境中验证。

---

## 1. `qa_stage1_common.py`：新增原子写 + safe row getter

### 1.1 `write_jsonl_atomic` / `write_json_atomic`
- **修改**：新增两个函数，使用 `tempfile.mkstemp` → write → flush → os.fsync →
  `os.replace` 原子落盘。
- **原因**：原脚本直接用 `json.dump(open(..., "w"))`，进程被 kill 时文件损坏。
  对应 CLAUDE.md "代码三件套" 中的流式落盘要求。原子写也天然支持
  `--resume`（下次启动从已写入的最后一行继续）。

### 1.2 `row_get(row, key, default)`
- **修改**：安全访问 `sqlite3.Row` 或 `dict` 的字段，缺失时返回 default。
- **原因**：`multimodal_samples.db` 在不同版本间 schema 有漂移（部分列可能
  缺失）。直接 `row[key]` 会在缺少列时抛 `IndexError`。现在 `prepare` 脚本
  先做 `PRAGMA table_info` 动态 SELECT 可用列，再配合 `row_get` 容忍缺失。

### 1.3 `non_generic_organs` / `clean_text` / `stable_hash` 等已有函数
- **修改**：无，保留原逻辑。
- **原因**：这些工具函数在 v0.3 下仍然正确，不需要改。

---

## 2. `prepare_qa_source_pool.py`：动态 SELECT + 原子写 + 计时

### 2.1 动态 SELECT
- **修改**：`fetch_rows()` 先 `PRAGMA table_info(multimodal_samples)` 查可用列，
  检查 REQUIRED_COLUMNS，然后只 SELECT 存在的列。缺失可选列时打印 warn。
- **原因**：DB schema 不确定（版本间可能增减列）。硬编码 SELECT 所有期望列会
  在缺列时报 SQL error。动态 SELECT 让脚本对 schema 漂移鲁棒。

### 2.2 `build_source` 中 `row["x"]` → `row_get(row, "x", ...)`
- **修改**：所有 DB 字段访问改用 `row_get`。
- **原因**：配合 2.1 的动态 SELECT，缺失列返回 None 而不是崩溃。

### 2.3 原子写 + 计时
- **修改**：`write_jsonl_atomic` / `write_json_atomic` 替换原来的 `json.dump`；
  main() 加 `t0` / `elapsed_s` / 每 2000 条进度（rate + ETA）。
- **原因**：三件套（时间统计 + 原子落盘 + 可续跑）。原脚本没有计时也没有原子
  落盘。

---

## 3. `route_qa_source_pool.py`：原子写 + 计时

- **修改**：`write_jsonl_atomic` / `write_json_atomic` 替换原来的 `json.dump`；
  加 `elapsed_s` 到 routing_report.json。
- **原因**：与 prepare 脚本一致，确定性脚本都应原子写，避免中断导致输出文件
  损坏。路由是纯规则计算，不需要计时太细，但 report 加 elapsed_s 方便排查。

---

## 4. `qa_api_prompts.py`：verifier 改盲答比对

### 4.1 QA_VERIFICATION_SYSTEM_PROMPT 重写
- **修改**：明确要求 verifier **独立作答**，不看到 candidate_answer /
  candidate_answer_text / evidence_basis，自己从 A/B/C/D 中选唯一正确答案。
- **原因**：原 verifier 直接看到生成器给的候选答案，容易"橡皮图章"式通过（rubber-
  stamping），丧失独立质检价值。盲答后比对 verifier_answer vs
  candidate_answer 才是真正的交叉验证。

### 4.2 QA_VERIFICATION_USER_TEMPLATE 重写
- **修改**：删除 `candidate_answer` / `candidate_answer_text` / `evidence_basis`
  的展示。新增 verifier_answer / verifier_confidence / answer_supported /
  multi_organ_ambiguity / option_quality / text_leakage / all_options_same_granularity
  / multiple_correct_options 等审核字段。
- **原因**：verifier 必须独立判断。新增字段覆盖了原审核中缺失的维度（多器官歧义、
  选项质量、题面泄露、多解），这些都是设计文档中明确要求的 gold quality gate。

### 4.3 `render_verification_user_prompt` 更新
- **修改**：不再传递/展示 candidate_answer / answer_text / evidence_basis。
- **原因**：配合 prompt 的盲答改造，确保实现层也不泄露这些信息。

---

## 5. `qa_api_common.py`：新建共享 API 工具模块

### 5.1 `load_api_config`
- **修改**：新建。校验 api_key 非 placeholder、base_url / model 存在。
- **原因**：防止用示例配置直接跑 API（之前没有校验，跑起来才发现 key 是假的）。

### 5.2 `build_image_messages`
- **修改**：新建。用 `encode_image_path_to_data_url` 将图片作为 data URL 嵌入
  OpenAI 格式消息。
- **原因**：设计文档 §17 明确要求 API 调用**必须传图片像素**，不能只传文本。
  否则"图像依赖"判定和 verifier 独立性都塌缩成纯文本判断。

### 5.3 `parse_json_content`
- **修改**：新建。容忍代码围栏（```json ... ```）和正文中混 JSON，最后用正则
  `\{.*\}` fallback。
- **原因**：模型回复经常带 markdown code fence 或解释性正文，直接 `json.loads`
  会频繁失败。健壮解析减少 API 浪费。

### 5.4 `extract_usage` / `accumulate_usage`
- **修改**：新建。从 API 回复中提取 prompt/completion/total tokens。
- **原因**：CLAUDE.md 记忆制度要求"API 消耗必须内嵌记录"。pipeline 调 API 时
  必须记录 usage 到结果 JSON，方便统计总成本和定位高消耗样本。

### 5.5 `load_done_ids` / `StreamWriter`
- **修改**：新建。`load_done_ids` 从已有输出 JSONL 收集 candidate_id；`StreamWriter`
  带锁的 append-mode JSONL 写入，每条 flush + os.fsync。
- **原因**：对应三件套的断点续传和流式落盘。API 脚本运行时间长（数千次调用），
  中断后必须能从断点继续。append + fsync 保证每条记录即时落盘不丢失。

---

## 6. `generate_qa_candidates_with_api.py`：新建生成脚本

### 6.1 整体新建
- **原因**：之前是 TODO 占位。现在需要实际的 API 调用脚本。

### 6.2 `validate_generation` 硬约束
- **修改**：校验 options == {A,B,C,D}、选项值互不相同、candidate_answer ∈ ABCD、
  answer_text == options[answer]、question/evidence_basis 非空、
  image_dependency ∈ {high,medium,low} 且不是 low。
- **原因**：模型回复可能格式错误/约束违反。生成期就做硬过滤，不合法的不流入
  verifier 节省 API 成本。reject low image_dependency 是因为设计文档要求
  "低图像依赖不能进入训练集"。

### 6.3 `process_one` 返回统一 record
- **修改**：统一返回带 generation_status 的 record，status ∈ {no_image, api_error,
  invalid_json, source_rejected, invalid:*, accepted}。
- **原因**：多线程并发下需要统一的结果格式，方便流式写入和统计。

### 6.4 ThreadPoolExecutor + StreamWriter + resume
- **修改**：`load_done_ids` 跳过已处理的 candidate_id，`StreamWriter` 实时
  append+fsync，每 25 条打印进度（rate + ETA + usage）。
- **原因**：三件套 + 大规模 API 调用需要并行 + 断点续跑 + 实时监控。

### 6.5 `generation_report.json`
- **修改**：记录 status_counts + api_usage_total_this_run。
- **原因**：方便后续排查哪些 status 失败最多、API 总消耗多少。

---

## 7. `verify_qa_candidates_with_api.py`：新建验证 + 组装

### 7.1 盲答 verifier
- **修改**：用更新后的 prompt 调 API，verifier 不看到 candidate_answer。通过判定：
  verifier_answer == candidate_answer AND answer_supported AND image_dependency ∈
  {high,medium} AND question_naturalness ∈ {high,medium} AND benchmark_style_match
  AND NOT multi_organ_ambiguity AND option_quality == good AND NOT text_leakage
  AND all_options_same_granularity AND NOT multiple_correct_options。
- **原因**：设计文档要求 verifier 独立作答，只有与生成器候选一致且满足所有质量门
  时才接受为 gold。这是 gold 数据质量的核心保障。

### 7.2 逐源去重 `_pick_best`
- **修改**：同一 source_id 最多保留 max_per_source（默认 1）条，优先 image_dependency
  high 再 question_naturalness high。
- **原因**：同一个 PDF 页面可能被路由到多个 query_type（如同时命中 anatomical +
  lesion），不去重会导致同一页面出现多道题，训练分布偏向高密度页面。

### 7.3 gold 组装 `_gold_record`
- **修改**：按设计 §6 schema 组装：顶层 qid/split/query_type/question/options/answer
  /answer_text/low_information_query_seed/query_image_path + source/generation
  /verification/provenance 子对象。
- **原因**：下游 SFT 格式（`data_request_mcq_sft_format.md`）要求这个 schema。
  generation/verification/provenance 是元信息，SFT 时只取顶层字段。

### 7.4 拆分 train/dev/internal_test
- **修改**：按 `source.split`（doc_id hash 预先分配）拆分到 qa_train.jsonl /
  qa_dev.jsonl / qa_internal_test.jsonl。
- **原因**：训练需要独立的开发集和测试集。doc_id 级别拆分保证同一文档的样本
  不会跨集合泄露。

---

## 8. `validate_qa_gold_dataset.py`：修 validator + 改名

### 8.1 删除 `evidence_text` 死字段检查
- **原代码**：`if not source.get("evidence_text"): add("missing_evidence_text")`
- **修改**：改为检查三个实际存在的证据字段中至少一个非空：
  `caption_or_pair_text` / `image_local_context_text` / `pdf_context_text`。
- **原因**：新 schema 不再使用 `evidence_text` 字段（已拆分为三个具体字段）。
  原检查始终报错，是 bug。

### 8.2 新增 `query_type` 允许集合检查
- **修改**：检查 query_type 是否在四个合法值中：anatomical_site_recognition /
  lesion_or_finding_identification / procedure_or_operation_recognition /
  spatial_region_understanding。
- **原因**：防止路由或生成阶段产生非法 query_type 流入 gold。

### 8.3 新增文件存在性检查
- **修改**：检查 `query_image_path` 和 `origin_pdf` 对应的文件是否真实存在。
- **原因**：图片/PDF 可能被移动或删除。如果 gold 引用了不存在的文件，下游 SFT
  会失败。

### 8.4 新增 `pdf_context_pages` 长度检查
- **修改**：要求 1 ≤ len(pdf_context_pages) ≤ 2。
- **原因**：设计文档规定 PDF 上下文最多取两页（当前页+相邻页）。超出说明上下文
  策略有误。

### 8.5 新增 `answer_text_in_question` 泄露检查
- **修改**：如果 answer_text 原文（≥4 字）完整出现在 question 中，标记泄露。
- **原因**：原 LEAKAGE_PATTERNS 只检查"答案/正确"等模板词，没有检查答案文本
  直接出现在题干中。这会让学生模型靠匹配而不是理解来答题。

### 8.6 改名 `validate_agentic_mcq_dataset.py` → `validate_qa_gold_dataset.py`
- **修改**：`git rm` 旧名，新名写代码。README 同步更新。
- **原因**：旧名 "agentic_mcq_dataset" 是 v0.2 命名，对应的是旧的 MCQ 构造方式。
  新流水线产出是 "qa_gold"（Stage 1 generate → verify → gold assembly），
  名字应与设计文档一致。

---

## 9. `README.md`：同步脚本名和流程

### 9.1 更新 step 4/5 去掉 (TODO)
- **修改**：`generate_qa_candidates_with_api.py` 和 `verify_qa_candidates_with_api.py`
  不再是 TODO，标记为已实现。
- **原因**：脚本已写完。

### 9.2 更新 step 6 validator 名字
- **修改**：`validate_agentic_mcq_dataset.py` → `validate_qa_gold_dataset.py`。
- **原因**：配合 8.6 改名。

### 9.3 更新 Scripts 列表
- **修改**：列出 qa_api_common.py，去掉旧 validator 名。
- **原因**：新增的共享模块需要在 README 中说明。

### 9.4 更新 Example 命令
- **修改**：validator 命令路径更新为 `validate_qa_gold_dataset.py`。
- **原因**：配合改名，示例需要能直接跑通。

---

## 在真实环境中测试的注意事项

1. **DB schema 验证**：`prepare_qa_source_pool.py` 已做动态 SELECT，但启动时打印
   的 `[warn] optional columns absent` 需要确认是否缺失了关键字段（organ_tags /
   primary_knowledge_type 等）。
2. **图片文件**：所有样本的 `query_image_path` 必须真实存在且可访问。
3. **API 配置**：复制 `api_config.example.json` → `api_config.json`，填入真实 key。
4. **API 成本**：`generate` + `verify` 每个候选调用两次 API，总计 token 消耗会
   比较大。建议先用 `--limit 10` 做 smoke test。
5. **Smoke test**：先跑 4 类 pilot（每类 20 条）验证全流程通顺，再全量跑。
