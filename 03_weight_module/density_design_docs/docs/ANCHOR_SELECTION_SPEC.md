# 锚点筛选规范（V1.1）

## 1. 目标
从原始 `final_description` 池中筛出可用于 query 链构造的 `anchor_text`。

注意：
- `anchor_text` 是语义事实载体，不是最终训练输入。
- 训练输入是后续生成/审查通过的 `query_text`。

## 2. 预清洗
先剔除以下样本再评分：
- 空文本/全空白
- 明显 OCR 脏片段（纯符号或乱码）
- 文本长度 `< 8` 或 `> 400`（可配置）
- 与其他样本完全重复（`text_norm_hash` 去重）

## 3. 设计思路（为什么要打分）
锚点评分不是为了“找到最学术的句子”，而是为了排序“对 query 构造是否有价值”的样本。

我们希望高分锚点具备三点：
1. 事实信息足够完整（能支撑 L2-L4）。
2. 语义有区分度（不是泛化模板句）。
3. 不过度依赖图像指代（否则可独立检索性差）。

因此公式被拆成：
- 正向信息量（anatomy/lesion/attribute/distribution/severity）
- 精确性加分（specificity）
- 泛化与指代惩罚（generic/deictic）

## 4. 锚点分层
### A 类（高信息锚点）
满足：
- 至少包含 1 个 `anatomy` 和 1 个 `lesion`
- 且有 `attribute/distribution/severity/context` 任一

用途：
- 主链 L2-L4 生成主力

### B 类（中信息锚点）
满足：
- 至少 1 个强约束槽位（`anatomy` 或 `lesion`）
- 但信息不够完整

用途：
- 生成 L1-L3，补齐中密度分布

### C 类（低信息锚点）
满足：
- 多为泛问、指代、短句，或弱语义文本

用途：
- 直接提供 L0/L1 天然样本，避免全靠生成

## 5. anchor_score 详细定义
### 5.1 总公式
为避免“槽位计数越多分越无上限”，先做归一化后再加权：

`anchor_score_raw =`
`0.24*A +`
`0.28*L +`
`0.16*AT +`
`0.12*D +`
`0.10*S +`
`0.10*SP -`
`0.10*GP -`
`0.08*DP`

`anchor_score = clamp(anchor_score_raw, 0, 1)`

变量含义：
- `A`：归一化 anatomy 信号
- `L`：归一化 lesion 信号
- `AT`：归一化 attribute 信号
- `D`：归一化 distribution 信号
- `S`：归一化 severity 信号
- `SP`：specificity_score
- `GP`：generic_penalty
- `DP`：deictic_penalty

### 5.2 计数项如何归一化
默认使用“截断 + 归一化”，防止堆词刷分：

- `A = min(anatomy_count, 2) / 2`
- `L = min(lesion_count, 2) / 2`
- `AT = min(attribute_count, 3) / 3`
- `D = min(distribution_count, 2) / 2`
- `S = min(severity_count, 2) / 2`

设计原因：
- anatomy / lesion 是核心检索条件，2 个后边际收益明显下降，故 cap=2。
- attribute 信息较细，常见合法并列略多，故 cap=3。
- distribution / severity 多数场景 1-2 个已足够表达，故 cap=2。

### 5.3 specificity_score（精确性加分）
`specificity_score in [0,1]`，用于奖励“具体、可定位、可区分”的表达。

建议实现：
- `term_signal = min(1.0, rare_term_count / 4.0)`（稀有专业词）
- `location_signal = 1.0 if 存在细粒度部位(如 小弯侧/下段/前壁) else 0.0`
- `quant_signal = 1.0 if 有定量/大小/范围(如 3cm/散在/局部) else 0.0`
- `specificity_score = clamp(0.5*term_signal + 0.3*location_signal + 0.2*quant_signal, 0, 1)`

为什么只给 `0.10` 权重：
- 精确性重要，但不能压过核心医学实体（anatomy/lesion）。
- 若权重过高，会让“术语堆砌句”异常高分。

### 5.4 generic_penalty（泛化惩罚）
`generic_penalty in [0,1]`，用于惩罚模板化泛问。

建议实现：
- 维护泛化短语词典：`有问题吗/正常吗/帮我看看/这是什么/是不是`
- `generic_penalty = min(1.0, generic_phrase_hits / 2.0)`

为什么惩罚权重 `0.10`：
- 泛化句会显著降低可检索性，惩罚力度需与 `specificity` 加分同量级。

### 5.5 deictic_penalty（指代惩罚）
`deictic_penalty in [0,1]`，惩罚“这里/该处/箭头所示”这类脱离图像难理解表达。

建议实现：
- 指代词典匹配得到 `deictic_hits`
- 若 `deictic_hits > 0` 且 `(anatomy_count + lesion_count) == 0`，设为强惩罚 `1.0`
- 否则 `deictic_penalty = min(1.0, deictic_hits / 2.0)`

为什么权重是 `0.08` 不是更高：
- 指代不总是坏样本；部分句子虽含“图中”，但仍有强实体可用。
- 所以略低于 generic penalty，避免误杀。

### 5.6 权重为什么这样分配
按“对检索收敛能力”的贡献排序：
1. `lesion (0.28)`：病变类型通常是最强区分信号。
2. `anatomy (0.24)`：部位是第一层过滤条件。
3. `attribute (0.16)`：形态特征提升区分度但不如前两者刚性。
4. `distribution (0.12)` / `severity (0.10)`：属于精修条件。
5. `specificity (+0.10)`：鼓励精准表达但防止术语刷分。
6. `generic (-0.10)` / `deictic (-0.08)`：抑制不可独立检索表达。

这组权重不是“理论最优”，而是工程上可解释、可调参的起点。后续可用 pilot 集做敏感性分析再微调。

### 5.7 评分分层阈值（建议）
- `anchor_score >= 0.60`：优先 A 候选
- `0.35 <= anchor_score < 0.60`：优先 B 候选
- `< 0.35`：优先 C 候选

注意：最终 A/B/C 仍以规则条件为主，分数用于排序和边界样本裁决，不单独决定类别。

### 5.8 计算示例
示例 1（高信息）：
- 文本：`胃窦小弯侧黏膜充血发红伴散在糜烂`
- 计数：anatomy=1, lesion=1, attribute=2, distribution=1, severity=0
- 归一化：A=0.5, L=0.5, AT=0.67, D=0.5, S=0
- 设 `SP=0.75, GP=0, DP=0`
- `score_raw ≈ 0.24*0.5 + 0.28*0.5 + 0.16*0.67 + 0.12*0.5 + 0 + 0.10*0.75 = 0.50`
- 结果：中高分，通常进入 A/B 边界，结合规则归 A。

示例 2（泛问）：
- 文本：`这是什么，有问题吗`
- 计数几乎为 0，`GP=1.0, DP=0.5`
- 分数接近 0，归 C。

示例 3（含指代但有实体）：
- 文本：`图中胃窦有糜烂吗`
- anatomy=1, lesion=1, deictic=1
- 有正向得分但受 `DP` 扣分，通常落在 B。

## 6. 采样策略
不要 pure top-k，使用“分层 + 配额”：
- A:B:C 建议配额 = `70:20:10`（pilot 阶段）
- 在每层内按分位数抽样，兼顾头部与长尾

## 7. 分层维度
- `doc_id`
- `organ`（或 anatomy 主类）
- `lesion` 主类
- 长度桶（短/中/长）
- 高频 vs 长尾术语

## 8. 防泄漏要求
- 锚点在切分前就要产出 `semantic_core_hash`
- 后续 split 必须按 `semantic_core_hash` 分组，不可跨集

## 9. 输出字段（anchor_pool）
- `anchor_id`
- `doc_id`
- `anchor_text`
- `anchor_text_norm`
- `anchor_score`
- `anchor_score_raw`
- `anchor_class`（A/B/C）
- `slot_summary`
- `semantic_core`
- `semantic_core_hash`

## 10. 建议记录的调试字段
- `score_components`：保存 A/L/AT/D/S/SP/GP/DP 逐项值
- `rule_decision_trace`：记录 A/B/C 规则判定路径
- `filter_reason`：被清洗剔除时的原因

这些字段会让后续你排查“为什么这个样本被判低分”时非常快。
