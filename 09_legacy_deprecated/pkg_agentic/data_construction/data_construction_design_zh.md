# MedAlign-RAG 数据构造与 Rewrite 训练工程设计

本文档记录当前确认后的完整工程设计。设计分为两个层次：

1. 当前立即落地的任务：构造 4,000 条可靠医学多模态 QA/MCQ 数据。
2. 后续训练链路：基于这 4,000 条可靠任务数据，让 GPT 作为真实 agent 接入冻结 RAG 链路，产生 rewrite rollout 和 reward，用于 reward-ranked offline warm-up / preference initialization（可选）与在线 GRPO。

核心边界：本轮数据构造只保证问题、选项、答案和来源证据正确；不在本轮臆造最终 rewrite 监督。rewrite 监督必须来自真实检索、重排、生成链路中的 original query 与 rewritten query 效果差异。

## 1. 总体目标

构建一套可追溯、可验证、可训练的医学多模态 agentic RAG 数据工程链路：

```text
Stage 1: Reliable QA Construction
自有书籍/期刊图文对 + PDF 两页上下文
 -> API 生成候选题面和选项；正确答案由源数据、图文上下文和验证逻辑共同约束

Stage 2: GPT Agent Rollout
可靠 QA/MCQ
 -> 构造低信息 original query
 -> 冻结 RAG 链路真实检索/重排/生成
 -> GPT agent 判断 ACCEPT/REWRITE 并产生多组 rewrite
 -> 每个 rewrite 真实跑完整链路

Stage 3: Reward Annotation & Training Data
original result vs rewrite result
 -> evidence reward / answer reward / delta reward / penalty
 -> 高质量轨迹用于可选 reward-ranked offline warm-up / preference initialization
 -> group-level reward 用于 GRPO 或 GRPO warm-start
```

这套设计的关键优势：

- 4,000 条 QA 数据先保证任务本身正确，避免训练信号建立在错误答案上。
- GPT rewrite 不是凭空生成标签，而是在真实冻结 RAG 链路中行动。
- 每个 rewrite 的 reward 来自真实检索和回答效果，给小模型明确监督信号。
- 小模型训练时不需要从完全随机探索开始，可以先学习 GPT 高质量 rollout，再进入 GRPO。

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

### 2.2 Stage 2 才产生 rewrite 轨迹

Stage 2 让 GPT 作为真实 agent，接入冻结 RAG 链路。GPT 根据 original query 的检索结果判断是否需要 rewrite，并产生多个 rewrite candidates。

Stage 2 产物是 `gpt_agent_rollouts.jsonl`，其中每条记录保存：

- original query
- original top-k evidence
- original generator answer
- GPT action: `ACCEPT` / `REWRITE`
- 多个 rewritten query candidates
- 每个 candidate 的 top-k evidence 和 generator answer

### 2.3 Stage 3 才产生训练 reward

Stage 3 比较 original 与 rewritten 的真实链路结果，得到 group-level reward。

Stage 3 产物是 `rewrite_group_rewards.jsonl`，用于：

- 可选 reward-ranked offline warm-up / preference initialization：用高 reward 轨迹初始化策略或偏好。
- preference learning：同一 qid 下比较好坏 rewrite。
- GRPO：使用同组多 candidate 的 reward 作为训练信号。

注意：如果训练时只使用离线保存的 GPT rollout reward，更准确地说是 offline RL / replay-based warm-up。如果训练时小模型生成的新 rewrite 仍然实时调用冻结 RAG 链路打分，则是更标准的 online GRPO。工程上建议先离线 warm-up，再在线小规模 GRPO。

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
/mnt/data_1/yds/多模态/agentic/outputs/qa_gold_4000/qa_gold_4000.jsonl
/mnt/data_1/yds/多模态/agentic/outputs/qa_gold_4000/train.jsonl
/mnt/data_1/yds/多模态/agentic/outputs/qa_gold_4000/dev.jsonl
/mnt/data_1/yds/多模态/agentic/outputs/qa_gold_4000/internal_test.jsonl
/mnt/data_1/yds/多模态/agentic/outputs/qa_gold_4000/construction_report.json
/mnt/data_1/yds/多模态/agentic/outputs/qa_gold_4000/validation_report.json
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
  "topk_evidence": [
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

GPT agent 输出：

```json
{
  "action": "REWRITE",
  "rewrite_candidates": [
    "这张内镜图显示的解剖部位是什么，重点关注黏膜形态和腔道结构？",
    "结合图像中的腔道形态和黏膜特征，判断该部位属于哪段消化道？"
  ],
  "failure_type": "missing_anatomical_site",
  "reason": "当前证据未明确覆盖可区分部位的视觉特征。"
}
```

或者：

```json
{
  "action": "ACCEPT",
  "rewrite_candidates": [],
  "failure_type": "none",
  "reason": "当前 top-k evidence 已足够回答。"
}
```

建议每条样本生成 `K=4` 个 rewrite candidates。对于需要更强探索的困难样本，可扩展到 `K=8`。

## 14. Stage 2 真实链路执行

每个 original query 和 rewritten query 都必须跑同一套冻结链路：

```text
query_text + optional query_image
 -> coarse retrieval
 -> multimodal reranker
 -> top-k evidence
 -> frozen generator
 -> answer prediction
```

固定约束：

- retriever 固定。
- reranker 固定。
- generator 固定。
- GPT agent 不直接改答案。
- GPT agent 只决定 ACCEPT/REWRITE 和 rewrite query。
- rewritten query 不能包含答案字母或直接泄露正确答案。

Stage 2 输出文件：

```text
/mnt/data_1/yds/多模态/agentic/outputs/gpt_agent_rollouts/gpt_agent_rollouts.jsonl
```

单条 rollout 结构：

```json
{
  "qid": "...",
  "original_query": "这张图是什么部位？",
  "original_result": {
    "topk": [],
    "generator_answer": "A",
    "generator_answer_text": "...",
    "answer_correct": false,
    "evidence_hit": false
  },
  "agent_output": {
    "action": "REWRITE",
    "failure_type": "missing_anatomical_site",
    "rewrite_candidates": []
  },
  "rewrite_results": [
    {
      "rewrite_id": "...",
      "rewrite_query": "...",
      "topk": [],
      "generator_answer": "B",
      "generator_answer_text": "...",
      "answer_correct": true,
      "evidence_hit": true
    }
  ]
}
```

## 15. Stage 3 Reward 设计

Reward 应同时考虑答案、证据、改写质量和成本。

建议初始 reward：

```text
reward = answer_delta
       + 0.3 * evidence_delta
       + 0.2 * support_delta
       - 0.2 * leakage_penalty
       - 0.1 * unnecessary_rewrite_penalty
       - 0.05 * length_penalty
```

其中：

```text
answer_delta = answer_correct_after - answer_correct_before
```

取值示例：

```text
before wrong, after right -> +1
before right, after right -> 0
before right, after wrong -> -1
before wrong, after wrong -> 0
```

`evidence_delta`：

```text
gold evidence / source doc / source page 是否进入 top-k 的变化
```

`support_delta`：

```text
verifier 判断 generator answer 是否被 top-k evidence 支持的变化
```

`leakage_penalty`：

```text
rewrite query 是否包含答案字母、直接答案文本、或明显泄露 gold answer
```

`unnecessary_rewrite_penalty`：

```text
original 已答对且证据充分，但 rewrite 没有提升或反而变差
```

`length_penalty`：

```text
rewrite 过长、塞入过多选项或证据描述
```

Stage 3 输出：

```text
/mnt/data_1/yds/多模态/agentic/outputs/rewrite_rewards/rewrite_group_rewards.jsonl
```

单条 group reward 结构：

```json
{
  "qid": "...",
  "group_id": "...",
  "original_query": "...",
  "candidates": [
    {
      "candidate_id": "orig",
      "query": "这张图是什么部位？",
      "answer_correct": false,
      "evidence_hit": false,
      "reward": 0.0
    },
    {
      "candidate_id": "rewrite_1",
      "query": "...",
      "answer_correct": true,
      "evidence_hit": true,
      "reward": 1.3
    }
  ],
  "best_candidate_id": "rewrite_1",
  "reward_components": {}
}
```

## 16. 训练数据使用方式

### 16.1 Reward-ranked Offline Warm-up / Preference Initialization（可选）

该步骤不是默认必须训练阶段，只是可选初始化方式。主训练目标仍是在线 GRPO。若使用离线初始化，可从 GPT rollout 中筛选：

```text
action = REWRITE
reward > 0
无答案泄露
rewrite 长度合理
answer_correct_after = true 或 evidence_hit_after = true
```

可构造成 imitation-style warm-up 样本：

```json
{
  "input": {
    "question": "...",
    "options": {},
    "original_query": "...",
    "topk_evidence": []
  },
  "output": {
    "action": "REWRITE",
    "rewrite_query": "..."
  }
}
```

同时保留一部分 `ACCEPT` 样本，避免小模型学成“总是 rewrite”。

### 16.2 Preference / DPO 可选

同一 qid 下：

```text
chosen = 高 reward rewrite
rejected = 低 reward rewrite 或 original query
```

可构造成 preference pair。

### 16.3 GRPO

小模型对同一输入生成多条 rewrite candidates，冻结 RAG 环境返回 reward。训练时可以用：

- GPT rollout reward bank 作为 warm-start。
- 小模型在线生成的新 candidates 作为真正 GRPO 训练样本。
- 先小规模在线评估，避免成本失控。

## 17. 当前优先实现顺序

当前先完成 Stage 1 构造链路，不启动 Stage 2 全量 rollout。

建议脚本顺序：

```text
1. prepare_qa_source_pool.py
   从 DB 抽取 image_text_pair 源样本，定位 image/PDF/page/context，生成 source_pool.jsonl。

2. route_qa_source_pool.py
   规则路由候选 query_type，按 doc_id split，生成 routed_source_pool.jsonl。

3. generate_qa_candidates_with_api.py
   调用 API 生成候选 question/options/candidate_answer/evidence_basis。

4. verify_qa_candidates_with_api.py
   独立 API verifier 严格过滤。

5. validate_qa_gold_dataset.py
   本地结构校验、split 泄露检查、统计报告。
```

推荐先跑 pilot：

```text
每类 20 条，共 80 条 candidate generation
人工检查 accepted 样本质量
再决定是否扩到每类 200 条
最后跑满 4,000 条
```

## 18. 旧 v2 数据的定位

旧版 `mcq_image_v2_4000` 只作为工程 smoke test 和对照参考，不作为最终训练数据。

旧 v2 问题：

- `organ_tags` 中第一个非“通用”标签被机械作为正确答案，无法处理多器官歧义。
- `image_content_type_identification` 是内部元分类题，不符合真实使用。
- 题型覆盖不足，缺少病变、操作、空间定位等 benchmark 风格难题。
- 没有经过图像 + PDF 两页上下文的生成式正确性确认。

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
