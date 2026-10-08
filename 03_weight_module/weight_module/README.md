# 权重模块处理流程（候选池 + 并发 LLM 打分）

本目录提供一套可直接执行的流程，用于：
1. 从原始图文对中清洗出候选池。
2. 并发调用 GPT 抽槽位并计算 `anchor_score`。
3. 产出可用于后续 A/B/C 分层和训练数据构造的锚点文件。
4. 基于锚点生成 L0-L4 query 链，并独立审查后导出训练数据。

## 1. 目录结构
- `pipeline_config.yaml`：统一配置（输入路径、清洗规则、API、并发、打分权重）
- `build_candidate_pool.py`：第一步清洗与候选池构建（不依赖 GPT）
- `llm_parallel_slot_scorer.py`：第二步并发 GPT 抽槽位 + anchor 打分
- `requirements.txt`：依赖
- `outputs/`：所有处理结果输出目录

## 2. 环境准备
```bash
cd /mnt/data_1/yds/多模态/权重模块
python3 -m pip install -r requirements.txt
```

设置 API Key（不要写死在代码里）：
```bash
export OPENAI_API_KEY='你的key'
```

如果你要使用其他中转地址或模型，在 `pipeline_config.yaml` 修改：
- `api.base_url`
- `api.primary_model`
- `api.fallback_models`

## 3. 第一步：构建候选池（推荐先执行）
```bash
python3 build_candidate_pool.py --config /mnt/data_1/yds/多模态/权重模块/pipeline_config.yaml
```

输出文件：
- `outputs/candidate_pool.cleaned.jsonl`
- `outputs/candidate_pool.filtered_examples.jsonl`
- `outputs/candidate_pool.stats.json`

## 4. 第二步：并发 LLM 槽位抽取与评分
```bash
python3 llm_parallel_slot_scorer.py --config /mnt/data_1/yds/多模态/权重模块/pipeline_config.yaml
```

可先小样本验证：
```bash
python3 llm_parallel_slot_scorer.py --max-samples 200
```

输出文件：
- `outputs/anchor_pool.scored.jsonl`
- `outputs/anchor_pool.errors.jsonl`

脚本支持：
- 并发（`request.concurrency`）
- 批处理（`request.batch_size`）
- 重试（`request.max_retries`）
- 模型降级（`primary_model + fallback_models`）
- 断点续跑（已处理 `candidate_id` 会自动跳过）

## 5. 模型与性能建议
- 快速低成本：`gpt-4o-mini`（推荐先跑）
- 更高质量：`gpt-4o`
- 建议策略：先 `mini` 全量跑，抽样复核后再对低置信样本二次用高质量模型重跑

## 6. 关键参数建议
- 吞吐优先：`batch_size=12~20`, `concurrency=8~16`
- 稳定优先：`batch_size=8~12`, `concurrency=4~8`
- 若接口限速明显，优先降低 `concurrency` 再降低 `batch_size`

## 7. 结果字段说明（第二步输出）
- `anchor_score_raw` / `anchor_score`
- `anchor_class_rule` / `anchor_class_score` / `anchor_class`
- `score_components`（A/L/AT/D/S/SP/GP/DP）
- `slots` / `counts` / `flags`
- `semantic_core`

## 8. 安全说明
- 当前代码不保存 API key 到文件。
- 请勿把明文 key 提交到 git、README 或日志。

## 9. 第三步：pilot 生成与审查
先构建 pilot 生成输入：
```bash
python3 /mnt/data_1/yds/多模态/权重模块/sample_generation_pilot.py
```

生成 L0-L4 问题链：
```bash
python3 /mnt/data_1/yds/多模态/权重模块/llm_chain_generator.py
```

独立审查生成结果：
```bash
python3 /mnt/data_1/yds/多模态/权重模块/llm_chain_reviewer.py
```

导出 pilot 训练三件套：
```bash
python3 /mnt/data_1/yds/多模态/权重模块/export_pilot_training_sets.py
```

第三步输出：
- `outputs/generation_input.pilot_1000.jsonl`
- `outputs/generated_chain.pilot.jsonl`
- `outputs/reviewed_chain.pilot.jsonl`
- `outputs/rejected_chain.pilot.jsonl`
- `outputs/density_cls.pilot.jsonl`
- `outputs/dependency_cls.pilot.jsonl`
- `outputs/density_rank.pilot.jsonl`
- `outputs/quality_report.pilot.json`

## 10. 当前主版本：v10 全量训练数据

v10 是当前已人工抽检认可的主版本。它基于 `anchor_pool.mainset_14000.split.jsonl` 中的 14000 条锚点生成 L0-L4 多等级问题链，其中成功生成 13992 条，失败 8 条。

核心输入：
- `outputs/generated_chain.mainset_14000.v10_balanced_forms.jsonl`

质量报告：
- `outputs/generated_chain.mainset_14000.v10_quality_report.json`
- `outputs/manual_audit_pack.v10.sample50.md`

导出三类训练数据：
```bash
python3 /mnt/data_1/yds/多模态/权重模块/export_training_sets_v10.py
```

导出结果：
- `outputs/density_cls.mainset_14000.v10.jsonl`：语义密度等级分类数据，输入是单条 query，标签是 `density_label=0..4`，对应 `L0..L4`。
- `outputs/dependency_cls.mainset_14000.v10.jsonl`：图像依赖度分类数据，输入是单条 query，标签是 `image_dependency=0..2`；`0` 表示文本自身较完整，`2` 表示高度依赖图像上下文。
- `outputs/density_rank.mainset_14000.v10.jsonl`：同一锚点内部的成对排序数据，输入是低密度 query 和高密度 query，`label=1` 表示 `query_high` 比 `query_low` 语义密度更高。
- `outputs/training_sets.mainset_14000.v10.report.json`：导出统计报告。

当前导出规模：
- 生成链：13992 条。
- `density_cls`：69960 条，L0-L4 每级 13992 条。
- `dependency_cls`：69960 条，依赖度分布为 `0:27984, 1:13992, 2:27984`。
- `density_rank`：139920 条，每条成功链产生 10 个有序等级对。

训练时建议：
- 第一阶段先训练 `density_cls`，验证模型是否能稳定区分 L0-L4。
- 第二阶段训练 `density_rank`，增强模型对“哪个问题更具体/更高密度”的相对判断。
- 第三阶段再训练或联合训练 `dependency_cls`，用于区分 query 是否必须依赖图像才能回答。

## 11. 训练脚本

训练代码位于：
- `train/train_text_density.py`
- `train/configs/density_cls_v10.yaml`
- `train/configs/dependency_cls_v10.yaml`
- `train/configs/density_rank_v10.yaml`
- `train/README_TRAINING.md`

安装训练依赖：
```bash
python3 -m pip install -r /mnt/data_1/yds/多模态/权重模块/train/requirements_train.txt
```

训练前冒烟测试：
```bash
bash /mnt/data_1/yds/多模态/权重模块/train/scripts/run_smoke_density_cls.sh
```

正式训练：
```bash
bash /mnt/data_1/yds/多模态/权重模块/train/scripts/run_train_density_cls.sh
bash /mnt/data_1/yds/多模态/权重模块/train/scripts/run_train_dependency_cls.sh
bash /mnt/data_1/yds/多模态/权重模块/train/scripts/run_train_density_rank.sh
```

训练默认只输入 `query_text`，不输入 `anchor_text`，目的是让模型学习问题本身的语义密度和图像依赖程度。

## 12. 推理接口

推理服务文件：
- `semantic_density_service.py`：Python 调用接口
- `infer_semantic_density.py`：命令行调用入口
- `build_rank_score_reference.py`：构建排序分数参考分布

先构建排序分数参考分布（只需一次）：
```bash
/mnt/data_1/yds/home/miniconda3/bin/python /mnt/data_1/yds/多模态/权重模块/build_rank_score_reference.py
```

单条 query 推理：
```bash
/mnt/data_1/yds/home/miniconda3/bin/python /mnt/data_1/yds/多模态/权重模块/infer_semantic_density.py \
  --query "这个情况正常吗？" \
  --pretty
```

两条 query 比较：
```bash
/mnt/data_1/yds/home/miniconda3/bin/python /mnt/data_1/yds/多模态/权重模块/infer_semantic_density.py \
  --query-a "这个情况正常吗？" \
  --query-b "胃窦黏膜粗糙伴结节样隆起提示什么病变？" \
  --pretty
```

接口输出包含三部分：
- `density`：密度等级分类结果，输出 `L0-L4`
- `image_dependency`：图像依赖度分类结果，输出 `0/1/2`
- `rank_signal`：排序模型给出的连续强弱信号，包含 `raw_score` 和 `percentile`

其中 `rank_signal` 的解释很重要：
- `raw_score` 是排序模型内部学到的单句标量，数值本身没有固定医学含义，只保证“更高通常代表更高密度”。
- `percentile` 是把 `raw_score` 放回训练分布后得到的位置，更适合直接使用。例如 `percentile=88` 可以理解为“这个问题在训练分布里大约比 88% 的 query 更密”。
- `closest_level_by_median` 是把该分数和训练集中各等级的中位数做最近匹配，作为辅助解释，不能替代正式的 `density` 分类输出。

推荐使用方式：
- 线上主输出：看 `density.label_name` 和 `image_dependency.label_name`
- 连续排序或边界 tie-break：看 `rank_signal.percentile`
- 两个 query 谁更具体：调用 `compare` / `--query-a + --query-b`，看 `pairwise_rank.prob_b_denser_than_a`

## 13. 统一权重模块入口

当前对外接口已经简化为“只做文本侧分析”。

也就是说，这一版接口只承担两件事：
- 输出文本信息密度标签：`L0-L4`
- 输出图片依赖等级标签：`R1-R3`

请直接使用下面这套接口：
- `weight_module_runtime.yaml`
- `multimodal_weight_service.py`
- `run_weight_module.py`

### 13.1 命令行入口

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/data_1/yds/home/miniconda3/bin/python \
  /mnt/data_1/yds/多模态/权重模块/run_weight_module.py \
  --query "这个情况正常吗？" \
  --pretty
```

输出核心字段：
- `density_level`
- `density_description`
- `image_dependency_level`
- `image_dependency_description`

### 13.2 标签定义

#### 信息密度标签

- `L0`：极低密度，几乎不提供稳定检索线索
- `L1`：低密度，语义方向较弱，仍然偏泛
- `L2`：中密度，已经包含部分有效约束
- `L3`：高密度，具备较完整判别线索
- `L4`：极高密度，问题本身已经非常具体

#### 图片依赖标签

当前对外不再使用训练时的 `0/1/2` 命名，而改为：

- `R1`：低图片依赖
- `R2`：中图片依赖
- `R3`：高图片依赖

对应关系是：

```text
训练标签 0 -> 对外标签 R1
训练标签 1 -> 对外标签 R2
训练标签 2 -> 对外标签 R3
```

这样做的原因是：
- `L0-L4` 已经在表示密度层级
- 图片依赖再用 `0/1/2` 容易和密度标签混淆
- `R1-R3` 更适合当作另一条独立维度

### 13.3 Python 调用

```python
from multimodal_weight_service import MultimodalWeightService

with MultimodalWeightService() as service:
    result = service.analyze(query_text="胃窦黏膜粗糙伴结节样隆起提示什么病变？")
```

返回结构示意：

```json
{
  "text_analysis": {
    "query_text": "...",
    "density_level": "L3",
    "density_description": "...",
    "image_dependency_level": "R1",
    "image_dependency_description": "...",
    "model_outputs": {
      "density_confidence": 0.91,
      "density_probabilities": {
        "L0": 0.00,
        "L1": 0.01,
        "L2": 0.08,
        "L3": 0.91,
        "L4": 0.00
      },
      "image_dependency_confidence": 0.97,
      "image_dependency_probabilities": {
        "R1": 0.97,
        "R2": 0.02,
        "R3": 0.01
      }
    }
  }
}
```

### 13.4 当前版本边界

这一版统一接口不再承担图片领域判别，也不直接决定是否走图片检索。

当前它只负责：
- 判断文本信息密度属于 `L0-L4` 哪一级
- 判断文本对图片依赖属于 `R1-R3` 哪一级

后续如果你要重新接回图片侧检索决策，可以把它当作一个上游文本分析模块来复用。
