# MedAlign-RAG 数据构造与 Agent 训练工程设计

> 版本 v0.4（2026-07-12 统一）。本文是数据侧的工程设计，配合权威设计思路
> `../medalign_paper_framework_zh.md`（Agent 统一控制器版）。核心对齐：agent = 单个 VLM policy，一次输出
> `keep/drop + ACCEPT/REWRITE`；奖励是 **answer-utility**（冻结 generator 的 logprob 概率提升），
> 不是 evidence-hit/相关性；主链路不含独立 reranker/PPR；训练走轻量 SFT 暖启 → GRPO。
> 所有 API 调用（生成/校验/rollout）**必须传入图像像素**。

本文档记录当前确认后的完整工程设计。设计分为两个层次：

1. 当前立即落地的任务：构造一批可靠医学多模态 QA/MCQ 数据（pilot 目标约 4,000 条）。
2. 后续训练链路：基于这批可靠任务数据——
   - 用冻结 generator 当探针，自动生成 **keep/drop 软标签**（无需人工，见 §keep/drop 软标签）；
   - 让 GPT 作为真实 agent 接入冻结 RAG 链路，产生 `keep/drop + ACCEPT/REWRITE` rollout；
   - 计算 **answer-utility 奖励**（`P_G(a*|E) − P_G(a*|∅)`）用于 SFT 暖启与在线 GRPO。

核心边界：本轮数据构造只保证问题、选项、答案和来源证据正确；不在本轮臆造最终 agent 监督。
keep/drop 标签由 generator 探针自动产生；ACCEPT/REWRITE 与 rewrite 文本监督必须来自真实检索、Agent 选择、
生成链路中不同动作的 answer-utility 差异，不凭空构造。

## 1. 总体目标

构建一套可追溯、可验证、可训练的医学多模态 agentic RAG 数据工程链路：

```text
Stage 1: Reliable QA Construction
自有书籍/期刊图文对 + PDF 两页上下文
 -> API（带图）生成候选题面和选项；正确答案由源数据、图文上下文和独立 verifier（带图）共同约束

Stage 2a: keep/drop 软标签（自动探针，无需人工）
可靠 QA/MCQ -> 冻结双路检索产生 candidate evidence
 -> 冻结 generator 逐条测边际概率提升（+该条证据是否抬高正确选项概率）
 -> 每条候选证据打上 keep/drop 软标签（SFT 阶段一目标）

Stage 2b: GPT Agent Rollout
可靠 QA/MCQ -> 构造低信息 original query -> 冻结双路检索
 -> GPT agent（带图）一次输出 keep/drop + ACCEPT/REWRITE(+rewrite_query)
 -> 每个动作真实跑完整闭环（REWRITE 则重检索/再选择）
 -> 冻结 generator 计算 answer-utility

Stage 3: Answer-utility Reward & Training Data
对每个动作 a，其证据集 E_a -> 冻结 generator 取正确选项 logprob
 -> r(a) = P_G(a*|q,I_q,E_a) − P_G(a*|q,I_q,∅)
 -> GPT rollout 中 utility 真升的决策做 SFT 暖启标签
 -> group-level answer-utility 用于 GRPO
```

这套设计的关键优势：

- 可靠 QA 数据先保证任务本身正确，避免训练信号建立在错误答案上。
- keep/drop 软标签由 generator 探针自动产生，无需逐条人工标注。
- GPT rollout 不是凭空生成标签，而是在真实冻结 RAG 链路中行动。
- 奖励是 answer-utility（对正确答案概率的真实提升），与最终问答目标一致，不退回相关性代理信号。
- 小模型训练时不需要从完全随机探索开始，先 SFT 暖启（合法 JSON + 粗略先验），再进入 GRPO。

## 2. 阶段边界

### 2.1 Stage 1 只构造可靠任务数据

Stage 1 只回答：

```text
这条医学图像问答数据本身是否正确、自然、可追溯？
```

Stage 1 不回答：

```text
这条样本的最佳 rewrite query 是什么？
rewrite 是否提升了检索效果？
```

Stage 1 产物是 `qa_gold_4000.jsonl`，其中每条样本包含：

- 问题 `question`
- 四个选项 `options`
- 正确答案 `answer` / `answer_text`
- query image
- 来源 PDF、页码、两页上下文
- API 候选生成记录与独立验证记录
- 用于后续 agent rollout 的低信息 query seed

Stage 1 的答案治理原则：API generator 可以提出候选答案，但不能被视为 gold answer 的唯一来源。最终 `answer/answer_text` 必须由源数据、图像、图片邻近文本、PDF 两页上下文、`evidence_basis` 和独立 verifier 共同约束；特别是 lesion/finding 类题，若 verifier 不能确认候选答案真实受证据支持，必须 reject。

`low_information_query_seed` 默认由规则生成：部位题为“这张图是什么部位？”，病变题为“图中是什么异常？”，操作题为“图中在做什么操作？”，空间题为“图中异常在哪里？”。API 只可提出备选，不应把 original query 写得过于具体，否则会削弱 Stage 2 rewrite 的提升空间。

### 2.2 Stage 2 才产生 agent 轨迹

Stage 2 让 GPT 作为真实 agent，接入冻结 RAG 链路。GPT 读入初检候选证据（含图像像素），
一次输出 `keep/drop + ACCEPT/REWRITE(+rewrite_query)`；REWRITE 时重检索并再次选择。

Stage 2 产物是 `gpt_agent_rollouts.jsonl`，其中每条记录保存：

- original query
- original candidate evidence（含 modality/text/image_path/retrieval score）
- GPT 的 keep/drop 决策
- GPT action: `ACCEPT` / `REWRITE`（REWRITE 带 rewrite_query）
- 每个动作产出的证据集与冻结 generator 的正确选项 logprob

### 2.3 Stage 3 才产生训练 reward（answer-utility）

Stage 3 对每个动作 a 计算 answer-utility 奖励，得到 group-level reward：

```text
r(a) = P_G(a* | q, I_q, E_a) − P_G(a* | q, I_q, ∅)
```

- `a*` 为正确选项 token；`P_G` 为冻结 generator 给正确选项的概率（vLLM logprobs）。
- ACCEPT 的 `E_a` = kept 子集；REWRITE 的 `E_a` = 重检索+再选择后的证据集；baseline `E=∅`。

Stage 3 产物是 `agent_group_rewards.jsonl`，用于：

- SFT 暖启：从 rollout 里挑 utility 真升的 `keep/drop + action` 决策作模仿目标。
- GRPO：同一 query 一个 group，组内多动作用相对优势 `A=(r−mean)/std` 训练。

注意：如果训练时只使用离线保存的 GPT rollout reward，更准确地说是 offline RL / replay-based warm-up。
如果训练时小模型生成的新动作仍然实时调用冻结 RAG 链路打分，则是更标准的 online GRPO。
工程上建议先离线 warm-up，再在线小规模 GRPO。

> ⚠️ 护城河：奖励必须是 answer-utility。一旦退回 evidence-hit/相关性命中，方法就塌成"学一个检索
> query 改写器"，与 Search-R1 / IterRetGen / DeepRAG 撞车且丢失医学特异性。**generator 必须能出
> logprob**，否则奖励和 keep/drop 软标签都算不出来。

## 3. 数据来源

当前主库：

```text
/mnt/data_1/yds/多模态/data_house/multimodal_samples.db
```

表：

```text
multimodal_samples
```

已确认规模：

```text
text_only: 163,713
image_text_pair: 66,291
```

Stage 1 主数据只使用：

```text
source_type = 'image_text_pair'
have_image = 1
```

原因：

- 目标场景是低信息图像 query，如“这张图是什么部位”“图中是什么病变”。
- 样本必须具有真实图像依赖，否则后续 rewrite agent 学到的是文本证据题。
- 图文对样本天然有 query image、caption/text、文档来源和 PDF 解析目录。

v1 阶段严格只使用 `image_text_pair`。`text_only` 样本暂不进入主训练池，后续只作为对照组、补充组或 ablation 数据，避免第一版任务偏离多模态图像依赖目标。

Benchmark 数据只用于：

- 统计题型分布。
- 学习问题风格。
- 最终外部测试。

Benchmark 样本内容不得进入训练数据，也不得被 GPT 用来改写成本项目训练题。

## 4. 主库字段使用

从 `multimodal_samples` 读取字段：

```text
sample_id
group_id
source_type
have_image
doc_id
doc_name
source_path
text
text_role
organ_tags
primary_knowledge_type
secondary_knowledge_types
labels
retrieval_meta
images
extra
```

字段职责：

- `sample_id`: 样本唯一来源 ID。
- `group_id`: 可用于同组去重或采样控制。
- `doc_id`: split 分组键，防止同一文档跨 split。
- `doc_name`: 追溯书籍/期刊来源。
- `source_path`: 定位 MinerU 输出目录。
- `images`: 定位 query image。
- `text`: 图文对描述或 caption，作为私有证据，不直接进入题面。
- `organ_tags`: 器官/部位候选提示，不能机械取第一个作为答案。
- `primary_knowledge_type` / `secondary_knowledge_types`: 题型路由弱提示，不能直接作为 gold answer。
- `labels`: 辅助候选信息，需 API 结合图像和上下文验证。
- `retrieval_meta` / `extra`: 保留以便后续调试和追溯。

## 5. PDF 两页上下文定位

每条图文样本通过 `source_path` 定位：

```text
{source_path}/auto/*_origin.pdf
{source_path}/auto/*_content_list.json
```

定位流程：

1. 从 `images[0].image_path` 获取图片文件名。
2. 在 `_content_list.json` 中查找同名图片条目。
3. 获取该图片对应 `page_idx`。
4. 默认抽取 `[page_idx, page_idx + 1]` 两页文本。
5. 如果 content_list 能定位图片条目，优先抽取图片前后邻近文本块，保存为 `source.image_local_context_text`。
6. 同时保留页级上下文：默认抽取 `[page_idx, page_idx + 1]`；若下一页不存在，则抽取 `[page_idx - 1, page_idx]`。
7. 将页级文本保存为 `source.pdf_context_text`，作为 fallback/补充证据。
8. 保留 `origin_pdf`、`content_list_path`、`image_item_index` 和 `pdf_context_pages`，便于后续人工核查。

这里“传入 PDF”指的是：

```text
query image + 图文对 text + 原始 PDF 中与图片相关的两页上下文
```

不是整本书或整篇 PDF。

## 6. Stage 1 输出格式

目标文件：

```text
outputs/qa_stage1_verified/qa_gold.jsonl
outputs/qa_stage1_verified/train.jsonl
outputs/qa_stage1_verified/dev.jsonl
outputs/qa_stage1_verified/internal_test.jsonl
outputs/qa_stage1_verified/construction_report.json
outputs/qa_stage1_verified/validation_report.json
```

单条样本结构：

```json
{
  "qid": "qa_xxx",
  "split": "train",
  "query_type": "anatomical_site_recognition",
  "question": "这张图显示的主要解剖部位是哪里？",
  "options": {
    "A": "胃窦",
    "B": "十二指肠球部",
    "C": "食管下段",
    "D": "结肠"
  },
  "answer": "B",
  "answer_text": "十二指肠球部",
  "low_information_query_seed": "这张图是什么部位？",
  "query_image_path": "...",
  "source": {
    "sample_id": "...",
    "group_id": "...",
    "doc_id": "...",
    "doc_name": "...",
    "origin_pdf": "...",
    "page_idx": 12,
    "pdf_context_pages": [12, 13],
    "image_item_index": 123,
    "image_local_context_text": "...",
    "caption_or_pair_text": "...",
    "pdf_context_text": "...",
    "organ_tags": ["十二指肠", "通用"],
    "primary_knowledge_type": "解剖特征",
    "secondary_knowledge_types": [],
    "labels": {}
  },
  "generation": {
    "generator_model": "...",
    "accept_source": true,
    "candidate_answer": "B",
    "candidate_answer_text": "十二指肠球部",
    "evidence_basis": "...",
    "image_dependency": "high",
    "ambiguity": false
  },
  "verification": {
    "verifier_model": "...",
    "accept": true,
    "answer_supported": true,
    "image_dependency": "high",
    "question_naturalness": "high",
    "benchmark_style_match": true,
    "multi_organ_ambiguity": false,
    "option_quality": "good",
    "text_leakage": false,
    "reason": "..."
  },
  "provenance": {
    "generated_by": "api_with_source_context",
    "verified_by": "api_with_image_and_pdf_context",
    "benchmark_used_for_training": false
  }
}
```

## 7. 题型覆盖与配比

目标总量：

```text
total: 4000
train: 3200
dev: 400
internal_test: 400
```

建议启动题型配比：

```text
anatomical_site_recognition: 35% 约 1400 条
lesion_or_finding_identification: 30% 约 1200 条
procedure_or_operation_recognition: 20% 约 800 条
spatial_region_understanding: 15% 约 600 条
```

该配比只是 pilot 启动配置，不是硬性死配比。最终比例应由 source pool 统计、API generator reject rate、独立 verifier accepted rate 共同决定：

- 空间定位题歧义率高，若 reject 率过高可降到 8%-10%，不要为了凑满 600 条硬造。
- 病变/发现识别和解剖部位识别通常更稳定，可作为补足来源。
- 操作/流程题必须有可见器械/操作场景，或 PDF 上下文明说该图对应某个操作步骤；不满足则不构造。

### 7.1 anatomical_site_recognition

目标：识别图像主要解剖部位、器官或局部区域。

可接受问题：

```text
这张图显示的主要解剖部位是哪里？
图中所示区域最可能属于哪个消化道部位？
这张内镜图像主要对应哪个部位？
```

题目难度应覆盖：

- 粗部位：食管/胃/小肠/结肠等。
- 亚部位：胃窦/胃体/十二指肠球部/回盲部等。
- 结构识别：皱襞、乳头、瓣口、开口等。

答案依据：

- 图像视觉结构为主。
- `organ_tags` 只作为候选提示。
- caption/text 和 PDF 两页上下文作为辅助确认。
- 如果多器官标签无法确定唯一主部位，必须 reject。

### 7.2 lesion_or_finding_identification

目标：识别病变、异常表现、内镜下发现或诊断相关视觉特征。

可接受问题：

```text
图中异常更符合哪一类病变？
这张图中最主要的内镜下发现是什么？
图中表现更符合哪种内镜下改变？
```

答案依据：

- 图像中的异常表现。
- caption/text 中的病变描述。
- 优先构造同部位不同病理或相似视觉表现的 hard negatives，避免过于容易的跨器官干扰项。
- PDF 两页中对应图注或正文说明。
- 不能仅依据 `labels` 直接定答案。

### 7.3 procedure_or_operation_recognition

目标：识别检查步骤、治疗操作、器械行为或正在处理的结构。

可接受问题：

```text
这一步内镜操作主要是在进行什么处理？
图中器械当前最可能用于哪类操作？
该图像最可能对应哪一类治疗步骤？
```

答案依据：

- 图像中是否有器械、切除、夹闭、注射、止血、标记等场景。
- caption/text 和 PDF 两页上下文中的操作说明。
- 若图像没有明确操作场景，且 PDF 上下文也没有明确说明该图对应某个操作步骤，不应硬构造成该题型。

### 7.4 spatial_region_understanding

目标：识别病灶、器械或操作区域的空间位置。

可接受问题：

```text
图中病变主要位于哪个区域？
图中器械所在位置最可能对应哪个部位？
图中异常主要集中在哪一侧或哪一区域？
```

答案依据：

- 图像空间布局。
- 图文描述和 PDF 两页上下文。
- 选项必须同粒度，例如都为区域/方位/解剖亚部位。

## 8. 不允许的题型和题面

以下题型不进入主数据：

```text
这张医学图像主要对应下列哪类内容？
图中信息最主要属于哪一类医学内容？
该图像属于病变特征/检查操作/治疗操作中的哪一类？
```

原因：

- 这类问题是内部 taxonomy 分类，不像真实用户查询。
- 它过度依赖 `primary_knowledge_type`，不利于训练低信息图像 query 的 rewrite agent。
- 对后续检索失败恢复的帮助有限。

题面禁止：

- 直接复述 caption 或 PDF 证据。
- 在问题中出现答案文本。
- 写成“根据上述证据/根据图注/根据文本”。
- 选项粒度混乱，例如把器官、疾病、操作混在一起。
- 多个选项都可能正确。

## 9. Stage 1 问题-答案构造链路

Stage 1 由四个子步骤组成：

```text
1. Source Pool Extraction
2. Source Triage & Type Routing
3. API Candidate QA Generation
4. API Independent Verification + Local Validation
```

### 9.1 Source Pool Extraction

目标：从主库中抽取可用于构造 QA 的源样本池。

输入：

```text
multimodal_samples.db
```

过滤条件：

```text
source_type = 'image_text_pair'
have_image = 1
images 非空
image_path 文件存在
source_path/auto/*_origin.pdf 存在
source_path/auto/*_content_list.json 存在
能从 content_list 匹配 image filename
能定位 page_idx
两页上下文非空
text 非空或 PDF 上下文非空
```

输出：

```text
source_pool.jsonl
```

每条 `source_pool` 记录包含：

```json
{
  "source_id": "...",
  "sample_id": "...",
  "group_id": "...",
  "doc_id": "...",
  "doc_name": "...",
  "query_image_path": "...",
  "origin_pdf": "...",
  "page_idx": 12,
  "pdf_context_pages": [12, 13],
  "image_item_index": 123,
  "image_local_context_text": "...",
  "caption_or_pair_text": "...",
  "pdf_context_text": "...",
  "organ_tags": [],
  "primary_knowledge_type": "...",
  "secondary_knowledge_types": [],
  "labels": {},
  "split": "train"
}
```

本步骤只做确定性抽取，不调用 API。

### 9.2 Source Triage & Type Routing

目标：判断每条 source 适合生成哪类问题，或者是否不适合生成问题。

建议先用规则进行粗路由，再由 API 在生成阶段二次确认。

规则路由示例：

```text
organ_tags 有明确非通用标签 -> anatomical_site_recognition 候选
text/PDF 中出现息肉、癌、溃疡、狭窄、出血、炎症、肿物等 -> lesion_or_finding_identification 候选
text/PDF 中出现切除、夹闭、注射、止血、ESD、EMR、活检、染色等 -> procedure_or_operation_recognition 候选
text/PDF 中出现近端、远端、前壁、后壁、左侧、右侧、边缘、中心、开口等 -> spatial_region_understanding 候选
```

如果一条 source 可对应多个题型，允许生成多个候选，但最终 4,000 条中应控制同一 `sample_id` 不重复过多。建议默认每个 source 最多保留 1 条 accepted QA，除非图像和文本明显支持多个独立问题。

输出：

```text
routed_source_pool.jsonl
```

关键字段：

```json
{
  "source_id": "...",
  "candidate_query_types": ["anatomical_site_recognition", "lesion_or_finding_identification"],
  "routing_reasons": ["organ_tags contains 胃", "caption mentions lesion"]
}
```

### 9.3 API Candidate QA Generation

目标：让 API 基于图像、caption/text、PDF 两页上下文生成候选 MCQ。

输入给 API：

```text
query image
caption_or_pair_text
PDF 两页上下文
organ_tags 全量列表
primary_knowledge_type
secondary_knowledge_types
labels
候选 query_type
禁止事项和输出 JSON schema
```

API 任务不是简单改写标签，也不是单独决定最终正确答案。API 只生成候选题面、候选选项和候选答案，并必须给出 `evidence_basis`。如果来源不能支持一个自然、低信息、图像依赖的医学问题，必须返回 `accept_source=false`。

Generator 输出 JSON：

```json
{
  "accept_source": true,
  "reject_reason": "",
  "query_type": "lesion_or_finding_identification",
  "question": "图中异常更符合哪一类病变？",
  "options": {
    "A": "炎性糜烂",
    "B": "息肉样病变",
    "C": "静脉曲张",
    "D": "正常皱襞"
  },
  "candidate_answer": "B",
  "candidate_answer_text": "息肉样病变",
  "evidence_basis": "图像显示隆起性病变，caption/PDF 上下文描述为息肉样改变。",
  "image_dependency": "high",
  "ambiguity": false,
  "low_information_query_seed": "图中是什么病变？"
}
```

硬约束：

- `accept_source=false` 直接丢弃。
- `candidate_answer` 必须为 `A/B/C/D`。
- `candidate_answer_text` 必须等于 `options[candidate_answer]`。
- 候选答案必须有 `evidence_basis`，后续 verifier 才能决定是否转成最终 gold answer。
- 四个选项必须同类型、同粒度。
- `question` 不能泄露 caption/PDF 证据。
- 不能把 `organ_tags` 第一个标签机械作为答案。
- `image_dependency` 只能为 `high/medium/low`。
- `low_information_query_seed` 由规则优先生成，API 只可给备选建议；该字段只是后续 GPT agent 的 original query，不作为 rewrite 标签。

### 9.4 API Independent Verification

目标：独立验证候选题是否可保留。Verifier 与 Generator 使用同样的源输入，但任务是严格审题，不负责润色。

Verifier 输入：

```text
query image
question
options
candidate_answer
candidate_answer_text
caption_or_pair_text
PDF 两页上下文
organ_tags 全量列表
primary_knowledge_type
secondary_knowledge_types
labels
```

Verifier 输出 JSON：

```json
{
  "accept": true,
  "answer_supported": true,
  "image_dependency": "high",
  "question_naturalness": "high",
  "benchmark_style_match": true,
  "multi_organ_ambiguity": false,
  "option_quality": "good",
  "text_leakage": false,
  "all_options_same_granularity": true,
  "multiple_correct_options": false,
  "reason": "答案能由图像和两页上下文共同支持，题面没有泄露证据。"
}
```

保留条件：

```text
accept = true
answer_supported = true
image_dependency in {high, medium}
question_naturalness in {high, medium}
benchmark_style_match = true
multi_organ_ambiguity = false
option_quality = good
text_leakage = false
all_options_same_granularity = true
multiple_correct_options = false
```

### 9.5 Local Validation

本地 validator 不判断医学语义，只检查结构和泄露风险：

```text
qid 唯一
split 合法
query_type 属于允许集合
answer 属于 A/B/C/D
answer_text == options[answer]
四个选项唯一
query_image_path 存在
origin_pdf 存在
page_idx 可追溯
pdf_context_pages 长度为 1 或 2
benchmark_used_for_training = false
同一 doc_id 不跨 split
question 不包含明显答案泄露词
```

通过 API verifier 和 local validator 的样本才进入 `qa_gold_4000.jsonl`。

## 10. Stage 1 采样与补齐策略

初始不要只抽 4,000 条 source，因为 API reject 会比较高。第一版按你的要求先构造 6,000 条原始输入用于问题构造；如果 pilot reject 率明显高，再扩展 source_pool。

```text
source_pool_v1: 6,000 条原始输入
API candidate generation: 分批运行
目标 accepted: 4,000 条
reject 率过高时再扩展到 12,000+
```

建议执行顺序：

1. 每种 query_type 先做 20 条 pilot，人工检查质量。
2. 通过后扩展到每类 200 条，统计 reject 原因和 accepted rate。
3. 根据 accepted rate 调整题型配比和 prompt。
4. 最后跑到 4,000 条 accepted。

补齐策略：

- 如果 `spatial_region_understanding` reject 率过高，先降比例。
- 如果 `procedure_or_operation_recognition` 源样本不足，减少该类目标量，不强行构造。
- 不为了凑数接受 image dependency low 的样本。
- 不为了凑数接受答案只由标签支持的样本。

## 11. Split 逻辑

目标切分：

```text
train: 3200
dev: 400
internal_test: 400
```

切分要求：

- 使用 `doc_id + seed` 稳定哈希决定 split。
- 同一 `doc_id` 只能进入一个 split。
- split 在 source pool 阶段就确定，后续 API 生成和验证都保留。
- 若某 split accepted 数不足，只从该 split 的 source pool 中补采，不跨 doc_id 回填。

这样避免同一本书、同一篇文章或同一 PDF 的样本同时出现在 train/dev/internal_test。

## 12. Stage 1 质量报告

`construction_report.json` 至少包含：

```json
{
  "source_pool_count": 0,
  "api_generated_count": 0,
  "api_rejected_count": 0,
  "verified_accept_count": 0,
  "verified_reject_count": 0,
  "final_count": 4000,
  "split_counts": {"train": 3200, "dev": 400, "internal_test": 400},
  "query_type_counts": {},
  "image_dependency_counts": {},
  "reject_reason_counts": {},
  "doc_id_count": 0
}
```

`validation_report.json` 至少包含：

```json
{
  "valid": true,
  "total": 4000,
  "errors": [],
  "warnings": [],
  "doc_id_split_leakage": [],
  "missing_images": 0,
  "missing_pdfs": 0,
  "invalid_answers": 0,
  "duplicate_options": 0
}
```

## 12.5 keep/drop 软标签构造（自动、无需人标）

用冻结 generator 当"探针"，看每条 top-k 证据把正确答案概率抬高还是压低。这是 SFT 阶段一的
keep/drop 监督来源，与 §15 的 answer-utility 奖励同源、口径一致。

流程：

1. **测基线**：只给"题目+题图+选项"（无检索证据），记正确选项概率，例 0.30。
2. **逐条测增量**：把 top-k 每条证据 `e_i` 单独加入再问一次，记新概率：
   - +第3条 -> 0.55（增量 +0.25）-> keep
   - +第2条 -> 0.20（增量 −0.10）-> drop
3. 对 k 条各做一遍 -> 得逐条 keep/drop 标签。

概率来源：generator 只输出单字母答案，从 vLLM logprobs 取该字母 token 的 logprob 转概率。
廉价版"逐条单加"忽略证据间交互但够暖启用；更准的 leave-one-out 更贵，非必需。

软标签输出（作为 Stage 1 gold 样本的附加字段或独立文件）：

```json
{
  "qid": "...",
  "p_correct_no_evidence": 0.30,
  "topk_soft_labels": [
    {"eid": "e1", "p_with": 0.36, "delta": 0.06, "label": "keep"},
    {"eid": "e2", "p_with": 0.20, "delta": -0.10, "label": "drop"},
    {"eid": "e3", "p_with": 0.55, "delta": 0.25, "label": "keep"}
  ]
}
```

## 13. Stage 2 GPT Agent Rollout 设计

Stage 2 的输入是 `qa_gold_4000.jsonl`，不是原始未验证 source。

每条 QA 构造一个 original low-information query：

```text
anatomical_site_recognition -> 这张图是什么部位？
lesion_or_finding_identification -> 图中是什么病变或异常？
procedure_or_operation_recognition -> 这一步在做什么操作？
spatial_region_understanding -> 图中异常或器械位于哪里？
```

如果 Stage 1 中已有 `low_information_query_seed`，优先使用该字段。

GPT agent 输入：

```json
{
  "qid": "...",
  "question": "...",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "original_query": "这张图是什么部位？",
  "query_image_path": "...",
  "candidate_evidence": [
    {
      "eid": "...",
      "modality": "text/image/image_text_pair",
      "text": "...",
      "image_path": "...",
      "score": 0.0
    }
  ]
}
```

GPT agent 输出（一次结构化 JSON，keep/drop 在前、action 在后，与小模型 policy 输出格式一致）：

```json
{
  "keep": [0, 2, 3],
  "drop": [1, 4],
  "action": "ACCEPT",
  "reason": "kept 证据已足够支持作答。"
}
```

或者：

```json
{
  "keep": [0],
  "drop": [1, 2, 3, 4],
  "action": "REWRITE",
  "rewrite_query": "结合图像中的腔道形态和黏膜特征，判断该部位属于哪段消化道？",
  "reason": "当前证据未覆盖可区分部位的视觉特征，需注入判别性临床词重检索。"
}
```

为构造 SFT 暖启标签，对 REWRITE 分支可让 GPT 生成 `K=4` 个 rewrite candidates 各跑完整闭环，
取 answer-utility 最高者作为监督目标；困难样本可扩展到 `K=8`。keep/drop 监督则直接来自 §keep/drop 软标签。

## 14. Stage 2 真实链路执行

每个 original query 和 rewritten query 都必须跑同一套冻结链路：

```text
query_text + query_image
 -> 密度感知路由（分配文本/图像检索预算，每轮重检索都重新路由）
 -> frozen coarse retrieval（文本/图像候选池，预算由路由或固定 top-k 决定）
 -> agent keep/drop（直接对初检候选逐条做答案效用判断）
 -> 冻结 generator（须开 logprobs）
 -> 取正确选项概率 P_G，用于 answer-utility 奖励
```

固定约束：

- retriever、可选密度路由和 generator 固定；唯一被训练的是 agent。
- generator 必须开 logprobs（只输出单字母答案，取该字母 token 概率）。
- GPT agent 不直接改答案，只输出 keep/drop + ACCEPT/REWRITE(+rewrite_query)。
- REWRITE 时把 dropped / 前几轮已见证据 ID 在**检索层**抑制，使多轮单调探索新证据。
- rewritten query 不能包含答案字母或直接泄露正确答案。
- 所有证据索引统一从 0 开始；query image 与候选图像证据都必须传入 Agent。

Stage 2 输出文件：

```text
outputs/gpt_agent_rollouts/gpt_agent_rollouts.jsonl
```

单条 rollout 结构（reward 用 answer-utility，即冻结 generator 对正确选项的概率提升）：

```json
{
  "qid": "...",
  "original_query": "这张图是什么部位？",
  "baseline": {
    "p_correct_no_evidence": 0.30
  },
  "original_result": {
    "candidate_evidence": [],
    "keep": [0, 2],
    "drop": [1, 3, 4],
    "kept_evidence": [],
    "p_correct_kept": 0.42,
    "utility_kept": 0.12
  },
  "agent_output": {
    "keep": [0, 2],
    "drop": [1, 3, 4],
    "action": "REWRITE",
    "rewrite_query": "..."
  },
  "rewrite_results": [
    {
      "rewrite_id": "...",
      "rewrite_query": "...",
      "candidate_evidence": [],
      "keep": [0, 1],
      "kept_evidence": [],
      "p_correct_kept": 0.71,
      "utility_kept": 0.41
    }
  ]
}
```

## 15. Stage 3 Reward 设计（answer-utility）

唯一主奖励来自冻结 generator 的答案效用，不来自检索相关性命中：

```text
r(a) = P_G(a* | q, I_q, E_a) − P_G(a* | q, I_q, ∅)
```

- `a*` 为正确选项 token；`P_G` 从冻结 generator 的 vLLM logprobs 取（generator 只输出单字母答案）。
- `E_a` 为动作 a 产出的证据集：ACCEPT 用 kept 子集；REWRITE 用重检索+再选择后的证据集。
- baseline `P_G(...|∅)` = 不给证据时的正确选项概率，是整组共享的减项。

轻量约束项（工程约束，论文里不作为主贡献，各扣少量分）：

```text
- invalid_json_penalty        : 输出不是合法 JSON / 动作无法解析
- leakage_penalty             : rewrite query 含答案字母或直接泄露 gold answer
- unnecessary_rewrite_penalty : 原证据已够（utility_kept 已高）仍 REWRITE
- length_penalty              : rewrite 过长
```

> 不要引入 evidence_delta / support_delta（gold doc/page 是否进 top-k）作为奖励主项——那是相关性
> 代理信号，会侵蚀护城河。gold doc/page 命中率只作为 §分析指标离线统计，不进 reward。

Stage 3 输出：

```text
outputs/agent_rewards/agent_group_rewards.jsonl
```

单条 group reward 结构（同一 query 一个 group，组内多动作跑完整闭环）：

```json
{
  "qid": "...",
  "group_id": "...",
  "original_query": "...",
  "p_correct_no_evidence": 0.30,
  "candidates": [
    {
      "candidate_id": "accept_topk",
      "action": "ACCEPT",
      "keep": [0, 1, 2, 3, 4],
      "p_correct": 0.34,
      "reward": 0.04
    },
    {
      "candidate_id": "keep_subset",
      "action": "ACCEPT",
      "keep": [0, 2],
      "p_correct": 0.42,
      "reward": 0.12
    },
    {
      "candidate_id": "rewrite_1",
      "action": "REWRITE",
      "rewrite_query": "...",
      "keep": [0, 1],
      "p_correct": 0.71,
      "reward": 0.41
    }
  ],
  "advantages": {"accept_topk": -0.7, "keep_subset": -0.2, "rewrite_1": 0.9},
  "best_candidate_id": "rewrite_1"
}
```

`advantages` 为组内相对优势 `A=(r−mean)/std`，即 GRPO 训练信号。

## 16. 训练数据使用方式（SFT 暖启 → GRPO）

两阶段训练，headline 是"policy 由 answer-utility RL 优化"，SFT 只是 cold-start 暖启。

### 16.1 阶段一：cold-start SFT 暖启（保持轻）

目标只是让模型出合法 JSON + 粗略正确的先验，不喂太饱（喂饱=RL 没空间=稀释主线）。
监督目标为完整 JSON（`keep/drop + action` 一起）：

- **keep/drop 标签** = §12.5 软标签（冻结 generator 自动生成，无需人工）。
- **ACCEPT/REWRITE 及 rewrite 文本标签** = 来自 GPT rollout（§13）里 answer-utility 真升的决策。

筛选 SFT 正样本：

```text
动作输出合法 JSON
无答案泄露、rewrite 长度合理
该动作的 r(a) > 0（answer-utility 真升）
keep/drop 与 §12.5 软标签一致
```

imitation-style 暖启样本：

```json
{
  "input": {
    "question": "...", "options": {}, "query_image_path": "...",
    "candidate_evidence": []
  },
  "output": {"keep": [0, 2], "drop": [1, 3, 4], "action": "REWRITE", "rewrite_query": "..."}
}
```

同时保留一部分 `ACCEPT` 样本，避免小模型学成"总是 rewrite"。

### 16.2 阶段二：GRPO（主线、增益来源）

每个 query 一个 group，组内包含多个动作各自跑完整闭环：

```text
{ ACCEPT(原始候选),  keep-子集,  rewrite 分支×N }
 每个动作 -> (REWRITE 则重检索) -> Agent 选择 -> generator -> r(a)
```

GRPO 用组内相对优势 `A=(r−mean)/std` 推动 policy 偏向高 answer-utility 的动作。训练时：

- GPT rollout reward bank 作为 warm-start / 离线 replay。
- 小模型在线生成的新动作作为真正 GRPO 训练样本，实时调用冻结链路打分。
- 先小规模在线评估，避免成本失控。

### 16.3 硬要求：SFT-only vs SFT+RL 消融

必须做 SFT-only vs SFT+RL 消融，用数字证明 RL 挣到了它的位置（也是核心卖点）。
SFT 只能模仿 GPT 决策、发现不了更优改写，也无组内反事实比较；RL 才是真正优化 answer-utility 的地方。

## 17. 当前优先实现顺序

当前先完成 Stage 1 构造链路，不启动 Stage 2 全量 rollout。

建议脚本顺序：

```text
1. prepare_qa_source_pool.py
   从 DB 抽取 image_text_pair 源样本，定位 image/PDF/page/context，生成 source_pool.jsonl。

2. route_qa_source_pool.py
   规则路由候选 query_type，按 doc_id split，生成 routed_source_pool.jsonl。

3. generate_qa_candidates_with_api.py（待实现，caller 必须传入图像像素）
   调用 API（带图）生成候选 question/options/candidate_answer/evidence_basis，复用 qa_api_prompts.py。

4. verify_qa_candidates_with_api.py（带图、独立 verifier）
   独立 API verifier（带图）严格过滤；对齐新流字段。

5. validate_qa_gold_dataset.py
   本地结构校验、split 泄露检查、统计报告。
```

> ⚠️ 生成与校验的 API 调用必须传入图像像素（用 `api_client.py` 的 `encode_image_path_to_data_url`），
> 否则"图像依赖"与 verifier 独立性都失效。图像 client 已就绪（`api_client.py`）。

推荐先跑 pilot：

```text
每类 20 条，共 80 条 candidate generation
人工检查 accepted 样本质量
再决定是否扩到每类 200 条
最后跑满 4,000 条
```

## 18. 已废弃的旧数据/脚本

旧版 `mcq_image_v2_4000` 及其生成脚本（`build_agentic_*_mcq_dataset.py`、
`verify_agentic_*_mcq_with_api.py`）已删除，不再作为任何训练/对照数据。删除原因：

- `organ_tags` 中第一个非"通用"标签被机械作为正确答案，无法处理多器官歧义。
- `image_content_type_identification` 是内部元分类题（本设计明令禁止，见 §8），不符合真实使用。
- 题型覆盖不足，缺少病变、操作、空间定位等 benchmark 风格难题。
- 没有经过图像 + PDF 两页上下文的生成式正确性确认，也不含 keep/drop 软标签与 answer-utility 口径。

新链路完全由 §9 起的 Stage 1 + §12.5 软标签 + §13 起的 answer-utility rollout 取代。

## 19. 风险与控制

### 19.1 API 幻觉

控制：Generator 只生成候选，Verifier 独立审核；本地 validator 再做硬检查；pilot 阶段人工抽查。

### 19.2 答案来自标签而不是图像

控制：`organ_tags` 和 `labels` 只作为提示；Verifier 必须判断答案是否由图像和上下文支持。

### 19.3 题面泄露证据

控制：禁止 caption/PDF 复述；本地规则扫描明显泄露；Verifier 输出 `text_leakage`。

### 19.4 选项粒度混乱

控制：Verifier 检查 `all_options_same_granularity` 和 `multiple_correct_options`。

### 19.5 训练时小模型只学会过度 rewrite

控制：Stage 2/3 保留 ACCEPT 样本，对不必要 rewrite 加 penalty。

### 19.6 离线 reward 与在线 GRPO 不一致

控制：先用 GPT rollout 做 warm-up，再小规模 online GRPO；论文中明确区分 offline reward bank 和 online policy optimization。

## 20. 当前决策点

在启动 Stage 1 代码和 API pilot 前，需要确认以下设计点：

1. 初始题型配比是否采用 `35/30/20/15`。
2. Stage 1 是否严格只用 `image_text_pair`，暂不混入 `text_only`。
3. 每个 source 默认最多保留 1 条 accepted QA，是否允许少量高质量 source 生成多题。
4. Pilot 是否按每类 20 条，共 80 条候选先跑；前置 source_pool_v1 默认 6,000 条。
5. Spatial region 题如果 reject 率高，是否允许自动降到 10%。

在这些点确认后，即可开始实现 Stage 1 构造脚本并跑 pilot。
