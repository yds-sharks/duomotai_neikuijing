# 轻量训练框架（V1.1）

## 1. 训练目标
只训练文本侧密度评估头，不微调文本 encoder。

输出四类信号：
1. `density_level`（L0-L4）
2. `density_score`（0-1）
3. `image_dependency`（0/1/2）
4. `pairwise_order`（同语义核内排序）

## 2. 输入与特征
### 2.1 编码输入
- 模型：`BGE-M3`
- 输入：`query_text`
- 输出：`embedding (1024-d)`
- 设置：`encoder_frozen=true`

### 2.2 结构化特征（建议）
- `char_len`
- `token_len`
- `anatomy_count`
- `lesion_count`
- `attribute_count`
- `distribution_count`
- `severity_count`
- `context_count`
- `deictic_flag`
- `generic_flag`
- `rare_term_count`
- `medical_term_ratio`
- `redundancy_ratio`
- `rule_density_raw`

## 3. 模型结构
`input = concat(embedding, structured_features)`

Backbone：
- `Linear(1024+d_f, 256) -> GELU -> Dropout`
- `Linear(256, 128) -> GELU -> Dropout`

Heads：
- `head_level`：5 类分类
- `head_reg`：1 维回归
- `head_dep`：3 类分类
- `head_rank`：共享 128-d 表示做 pairwise ranking

## 4. 损失函数
总损失：
`L = λ_ord*L_ord + λ_reg*L_reg + λ_dep*L_dep + λ_rank*L_rank + λ_cons*L_cons`

- `L_ord`：等级分类交叉熵（可带 class weights）
- `L_reg`：`SmoothL1`（目标为规则分引导后的 `density_score`）
- `L_dep`：依赖度分类交叉熵
- `L_rank`：pairwise hinge / logistic ranking loss
- `L_cons`：一致性约束（相邻 level 预测单调）

## 5. 数据组织
### 5.1 单样本数据
- `query_id`
- `query_text`
- `density_level`
- `density_score`
- `image_dependency`
- `semantic_core_hash`
- `split`

### 5.2 排序数据
- `query_id_a`
- `query_id_b`
- `label`（1 表示 a 密度 > b）
- `semantic_core_hash`
- `pair_type`（main_chain/style_variant/hard_case）

## 6. Baseline 顺序
1. `rule-only`
2. `frozen encoder + linear (level only)`
3. `frozen encoder + MLP (level+reg+dep)`
4. `frozen encoder + MLP + ranking + consistency`

## 7. 评测指标
- `Macro-F1`（L0-L4）
- `Weighted-F1`（L0-L4）
- `Pairwise Accuracy`
- `Spearman` / `Kendall`
- `MAE`（density_score）
- `Dependency Accuracy`

## 8. 早停与模型选择
- 早停主指标：
`val_joint_score = 0.45*MacroF1 + 0.25*PairAcc + 0.20*DepAcc + 0.10*(1-MAE)`
- 连续 `patience` 轮不提升则停止。

## 9. 风险与对策
- 风险：level 分布不均衡  
  对策：分层采样 + class weights
- 风险：生成数据风格漂移  
  对策：固定人工校准集做回归检查
- 风险：回归头被噪声拉偏  
  对策：先用规则分暖启动，再联合优化
