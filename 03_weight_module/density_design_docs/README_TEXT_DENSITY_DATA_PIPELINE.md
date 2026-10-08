# 文本语义密度模块方案（V1.1）

## 1. 任务定义
训练一个文本侧路由模块，输入 `user_query`，输出：
- `density_level`：L0-L4（离散密度等级）
- `density_score`：`[0,1]` 连续分（用于阈值路由）
- `image_dependency`：`{0,1,2}`（是否依赖图像解释）

本模块评估的是“文本独立检索价值”，不是图像内容理解本身。

## 2. 数据来源与定位
- 原始图文对：`data_house/origin_data/image_text/output_pairs_all_min_filtered.jsonl`
- 该文件里的 `final_description` 仅作为 `anchor_text`（语义锚点）。
- 训练输入样本必须是 `query_text`，不能直接拿 `final_description` 当 query。

## 3. 核心策略
采用“锚点驱动混合构造”：
1. 用 `final_description` 提供稳定医学事实。
2. 抽取槽位形成 `semantic_core`。
3. 在同一 `semantic_core` 下生成 L0-L4 query 链。
4. 加入风格变体和困难样本，补足真实用户表达分布。
5. 独立审查不过则回炉。

锚点评分与分层的完整解释（每项指标定义、权重原因、算例）见：
- `docs/ANCHOR_SELECTION_SPEC.md` 的第 5 节。

## 4. 标签与分数设计
### 4.1 density_level
详见 `docs/LABEL_GUIDELINE.md`。判定基于“约束信息量 + 可独立检索性”，不按句长硬判。

### 4.2 image_dependency
- `0`：文本足够独立检索
- `1`：部分依赖图像
- `2`：强依赖图像（大量指代表达或缺关键实体）

### 4.3 density_score（确定性规则）
先按 `density_level` 给基础分：
- `L0=0.10` `L1=0.30` `L2=0.50` `L3=0.70` `L4=0.90`

再计算规则密度分 `rule_density_raw`：
- `slot_score = min(1.0, 0.28*anatomy + 0.32*lesion + 0.18*attribute + 0.12*distribution + 0.10*severity)`
- `specificity_bonus = min(0.08, 0.02*rare_term_count)`
- `deictic_penalty = 0.10 if deictic_flag else 0.0`
- `generic_penalty = 0.06 if generic_flag else 0.0`
- `redundancy_penalty in [0, 0.08]`（模板化废话比例）
- `rule_density_raw = clamp(slot_score + specificity_bonus - deictic_penalty - generic_penalty - redundancy_penalty, 0, 1)`

偏移项：
- `offset = clip(0.16*(rule_density_raw - 0.5), -0.08, 0.08)`

最终分：
- `density_score = clamp(level_base + offset, 0, 1)`

## 5. 数据切分原则
- 切分单位：`semantic_core_hash`（必须）
- 目标比例：`train/val/test = 0.8/0.1/0.1`
- 禁止同一 `semantic_core_hash` 出现在不同 split。

## 6. 产物文件
- `density_cls.jsonl`：`query -> level/score`
- `dependency_cls.jsonl`：`query -> image_dependency`
- `density_rank.jsonl`：同核心 query 的排序对/组

## 7. 训练框架
- Encoder：`BGE-M3`（冻结）
- 输入：`embedding + structured features`
- 头部：两层 MLP + 多任务头（level/regression/dependency/ranking）
- 训练细节见 `docs/TRAINING_FRAMEWORK.md` 与 `configs/train_text_density_v1.yaml`

## 8. 质量门槛（放量前）
- `new_fact_rate < 1%`
- `non_monotonic_rate < 3%`
- `review_pass_rate > 85%`
- 人工抽检自然度达标后再从 pilot 扩到主集

## 9. 目录说明
- `prompts/`：槽位抽取、链生成、审查
- `docs/`：标注规范与训练框架
- `configs/`：数据生成与训练配置
- `scripts/`：执行步骤说明
