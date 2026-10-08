# 执行清单（V1.1）

## 阶段 0：准备
1. 确认输入文件：`output_pairs_all_min_filtered.jsonl`。
2. 创建输出目录：`text_density_design/outputs/v1.1`。
3. 固定随机种子：`20260422`。

## 阶段 1：清洗与锚点池
1. 规范化文本（去换行、去首尾空白、统一空格）。
2. 过滤极短/极长和低置信表达样本。
3. 以 `text_norm_hash` 去重。
4. 产出 `anchor_pool.jsonl`（含 A/B/C 分类与分数）。

## 阶段 2：槽位抽取
1. 对 `anchor_text` 调用 `prompts/extract_slots_prompt.md`。
2. 校验 JSON schema 与字段完整性。
3. 生成 `semantic_core_hash`。
4. 产出 `slot_records.jsonl`。

## 阶段 3：切分（必须先切再生成）
1. 按 `semantic_core_hash` 分组切分 train/val/test。
2. 检查是否有跨 split 泄漏。
3. 固化 split 清单。

## 阶段 4：链生成与审查
1. 用 `prompts/generate_chain_prompt.md` 生成 L0-L4 主链 + 变体 + hard case。
2. 用 `prompts/review_chain_prompt.md` 独立审查。
3. 对不通过样本按 `issues` 回炉重生。
4. 产出 `reviewed_chain.jsonl`。

## 阶段 5：训练数据导出
1. 导出 `density_cls.jsonl`。
2. 导出 `dependency_cls.jsonl`。
3. 导出 `density_rank.jsonl`。
4. 统计 level/dependency 分布并检查配额。

## 阶段 6：训练与校准
1. 预计算 BGE-M3 embedding。
2. 跑 baseline：rule -> linear -> mlp -> mlp+ranking。
3. 使用 `train_text_density_v1.yaml` 训练主模型。
4. 在 val 上做阈值校准并输出路由阈值表。

## 阶段 7：放量条件
1. `new_fact_rate < 1%`
2. `non_monotonic_rate < 3%`
3. `review_pass_rate > 85%`
4. 人工抽检通过后，再从 pilot 扩至 1.2w-1.4w 锚点。
