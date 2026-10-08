# 语义密度模块训练方案 v10

本目录用于训练三个子任务：

- `density_cls`：输入单条问题 `query_text`，输出语义密度等级 `L0-L4`，对应标签 `0-4`。
- `dependency_cls`：输入单条问题 `query_text`，输出图像依赖度 `0/1/2`，其中 `2` 表示强依赖图像上下文。
- `density_rank`：输入同一锚点内的两个问题，判断第二个问题是否比第一个问题语义密度更高。

## 1. 当前默认训练策略

默认只使用 `query_text`，不使用 `anchor_text`。原因是我们要训练的是“问题本身的语义密度判断器”，如果把锚点说明一起输入，模型容易学到病例上下文匹配，而不是学问题密度。

默认编码器：

```bash
/mnt/data_1/yds/微调/models/mengzi-bert-base
```

这是一个相对轻量的中文 BERT 基线，适合今晚先跑通。如果显存充足，后续可以把配置中的 `model.name_or_path` 改成 `/mnt/data_1/yds/bge-m3` 做更强版本。

## 2. 训练输入

已导出的 v10 数据：

- `/mnt/data_1/yds/多模态/权重模块/outputs/density_cls.mainset_14000.v10.jsonl`
- `/mnt/data_1/yds/多模态/权重模块/outputs/dependency_cls.mainset_14000.v10.jsonl`
- `/mnt/data_1/yds/多模态/权重模块/outputs/density_rank.mainset_14000.v10.jsonl`

数据规模：

- `density_cls`：69960 条，L0-L4 完全均衡。
- `dependency_cls`：69960 条，依赖度分布为 `0:27984, 1:13992, 2:27984`。
- `density_rank`：139920 条正向排序对；训练时默认追加反向负样本，因此有效样本为 279840 条。

`dependency_cls` 默认启用 `class_weight: auto_balanced`，会自动提高类别 `1` 的损失权重，避免模型过度偏向 `0/2` 两端。

## 3. 环境准备

默认 `python3` 当前还缺少训练依赖。先安装：

```bash
python3 -m pip install -r /mnt/data_1/yds/多模态/权重模块/train/requirements_train.txt
```

如果你使用 conda 环境，请先激活环境再安装。

当前机器上可直接使用的训练解释器是：

```bash
/mnt/data_1/yds/home/miniconda3/bin/python
```

训练脚本已默认使用这个 Python；如果你想切到别的环境，可以在执行前指定：

```bash
export TRAIN_PYTHON=/你的/python
```

正式开跑前也可以先做一次总检查：

```bash
bash /mnt/data_1/yds/多模态/权重模块/train/scripts/preflight_check.sh
```

## 4. 先做冒烟测试

正式开跑前建议先跑 500 条小样本，确认 CUDA、模型路径和输出目录都正常：

```bash
bash /mnt/data_1/yds/多模态/权重模块/train/scripts/run_smoke_density_cls.sh
```

只检查数据和配置、不加载 torch：

```bash
python3 /mnt/data_1/yds/多模态/权重模块/train/train_text_density.py \
  --config /mnt/data_1/yds/多模态/权重模块/train/configs/density_cls_v10.yaml \
  --dry-run
```

## 5. 正式训练顺序

推荐今晚按这个顺序跑：

```bash
bash /mnt/data_1/yds/多模态/权重模块/train/scripts/run_train_density_cls.sh
bash /mnt/data_1/yds/多模态/权重模块/train/scripts/run_train_dependency_cls.sh
bash /mnt/data_1/yds/多模态/权重模块/train/scripts/run_train_density_rank.sh
```

也可以一键顺序跑三项，并自动保存日志：

```bash
bash /mnt/data_1/yds/多模态/权重模块/train/scripts/run_all_v10.sh
```

原因：

- 先跑 `density_cls`，它是主目标，数据均衡，最容易判断数据是否有效。
- 再跑 `dependency_cls`，补充“问题是否依赖图像”的辅助信号。
- 最后跑 `density_rank`，用于增强相对密度判断，训练量最大，最好放最后。

## 6. 输出目录

训练结果会写入：

- `/mnt/data_1/yds/多模态/权重模块/checkpoints/density_cls_v10_mengzi`
- `/mnt/data_1/yds/多模态/权重模块/checkpoints/dependency_cls_v10_mengzi`
- `/mnt/data_1/yds/多模态/权重模块/checkpoints/density_rank_v10_mengzi`

每个目录包含：

- `best_model/`：验证集最优模型权重和 tokenizer。
- `config.resolved.yaml`：实际使用的配置。
- `metrics.json`：训练历史、最佳验证指标、测试集指标。
- `predictions.test.jsonl`：测试集预测明细，便于抽查错例。

## 7. 关键审核点

- `density_cls` 的核心指标看 `test.macro_f1` 和混淆矩阵，重点关注相邻等级是否混淆，例如 L1/L2、L3/L4。
- `dependency_cls` 的核心指标看类别 `1` 的召回率，因为中等依赖样本较少。
- `density_rank` 的核心指标看 `test.accuracy`，它衡量同源问题对的相对密度方向是否判断正确。

## 8. 今晚建议参数

如果使用 `mengzi-bert-base`：

- `density_cls`: `batch_size=32`, `epochs=3`, `lr=2e-5`
- `dependency_cls`: `batch_size=32`, `epochs=3`, `lr=2e-5`
- `density_rank`: `batch_size=16`, `gradient_accumulation_steps=2`, `epochs=2`, `lr=1e-5`

如果显存不足：

- 先把分类任务 `batch_size` 从 32 降到 16。
- 排序任务把 `batch_size` 从 16 降到 8，保留 `gradient_accumulation_steps=2`。

## 9. 训练后如何调用

训练完成后，直接使用根目录下的推理接口：

```bash
/mnt/data_1/yds/home/miniconda3/bin/python /mnt/data_1/yds/多模态/权重模块/build_rank_score_reference.py
/mnt/data_1/yds/home/miniconda3/bin/python /mnt/data_1/yds/多模态/权重模块/infer_semantic_density.py --query "这个情况正常吗？" --pretty
```

建议：
- `density` 作为离散主标签
- `image_dependency` 作为是否需要依赖图像的辅助标签
- `rank_signal.percentile` 作为连续强弱信号，用于边界排序和 tie-break
