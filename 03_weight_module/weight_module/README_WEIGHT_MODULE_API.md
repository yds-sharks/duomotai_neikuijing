# 权重模块调用说明

这个文档只说明当前推荐的文本侧调用接口，不包含数据构造、训练流程或历史实验说明。

## 1. 当前推荐接口

当前推荐对外使用的是：

- 输入一条 `query` 文本
- 输出两类标签
  - `L_label`: 二分类粗粒度信息密度标签
  - `R_label`: 图片依赖标签

当前标签定义：

- `L0`: 由原内部标签 `L01` 映射而来
- `L1`: 由原内部标签 `L234` 映射而来
- `R1/R2/R3`: 保持不变

当前推荐接口不输出：

- `L0-L4` 五分类细粒度标签
- 排序分数
- 图像输入分析
- 批量文件接口

## 2. 代码位置

### 2.1 当前推荐对外接口

当前推荐直接使用这两个文件：

- `/mnt/data_1/yds/多模态/权重模块/run_query_lr.py`
- `/mnt/data_1/yds/多模态/权重模块/query_lr_service.py`

它们内部依赖训练好的 checkpoint：

- `density2_coarsefilter_mengzi`
- `routing_cls_api10k_mengzi`

### 2.2 底层依赖

底层公共推理工具仍然复用了这里的运行时依赖：

- `/mnt/data_1/yds/多模态/权重模块/semantic_density_service.py`

### 2.3 旧接口说明

下面这些文件仍然存在，但它们对应的是**旧的细粒度文本分析接口**，不是当前推荐入口：

- `/mnt/data_1/yds/多模态/权重模块/run_weight_module.py`
- `/mnt/data_1/yds/多模态/权重模块/multimodal_weight_service.py`
- `/mnt/data_1/yds/多模态/权重模块/weight_module_runtime.yaml`

旧接口输出的是：

- `L0-L4`
- `R1-R3`

如果你当前要的是“输入一句文本，返回 `L0/L1 + R1/R2/R3`”，**不要再用旧接口**。

## 3. 运行环境

推荐直接使用这个 Python：

```bash
/mnt/data_1/yds/home/miniconda3/bin/python
```

如果你要指定 GPU：

```bash
CUDA_VISIBLE_DEVICES=0
```

## 4. 命令行调用

### 4.1 最简单调用

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/data_1/yds/home/miniconda3/bin/python \
  /mnt/data_1/yds/多模态/权重模块/run_query_lr.py \
  --query "根据内镜图像，正在进行的手术操作是什么？" \
  --pretty
```

### 4.2 参数说明

- `--query`
  输入的问题文本，必填
- `--pretty`
  以格式化 JSON 输出，便于阅读
- `--root-dir`
  可选，默认：
  `/mnt/data_1/yds/多模态/权重模块`
- `--device`
  可选，默认 `auto`
  可填 `cpu`、`cuda`、`cuda:0` 等

## 5. Python 调用

### 5.1 最小调用

```python
from query_lr_service import QueryLRService

service = QueryLRService()
result = service.predict("根据内镜图像，正在进行的手术操作是什么？")
print(result)
```

### 5.2 返回值示例

```python
{
    "query_text": "根据内镜图像，正在进行的手术操作是什么？",
    "L_label": "L0",
    "L_confidence": 0.998321,
    "R_label": "R3",
    "R_confidence": 0.996842
}
```

## 6. 输入格式

输入就是一条字符串：

```text
query_text: "胃窦黏膜粗糙伴结节样隆起提示什么病变？"
```

当前推荐接口：

- 不接收图片
- 不接收图文对
- 不接收 batch 文件
- 不拼接 options
- 默认只按纯 `query` 推理

## 7. 输出格式

标准输出结构如下：

```json
{
  "query_text": "根据内镜图像，正在进行的手术操作是什么？",
  "L_label": "L0",
  "L_confidence": 0.998321,
  "R_label": "R3",
  "R_confidence": 0.996842
}
```

## 8. 输出字段说明

- `query_text`
  输入的原始文本

- `L_label`
  粗粒度文本信息密度标签，取值 `L0/L1`

- `L_confidence`
  当前 `L_label` 的置信度

- `R_label`
  图片依赖标签，取值 `R1/R2/R3`

- `R_confidence`
  当前 `R_label` 的置信度

## 9. 标签说明

### 9.1 L 标签

对外标签：

- `L0`
- `L1`

内部映射关系：

```text
L01 -> L0
L234 -> L1
```

可以简单理解为：

- `L0`：文本本身检索约束弱、信息密度较低
- `L1`：文本本身检索约束较强、信息密度较高

### 9.2 R 标签

- `R1`：低图片依赖
  仅靠文本通常也可以做判断

- `R2`：中图片依赖
  文本提供了一部分信息，但结合图片更稳

- `R3`：高图片依赖
  如果不看图片，通常难以可靠判断

## 10. 最小可交付调用信息

如果你只需要把当前可用调用方式发给别人，给这三条就够了：

1. 入口脚本：
`/mnt/data_1/yds/多模态/权重模块/run_query_lr.py`

2. 推荐 Python：
`/mnt/data_1/yds/home/miniconda3/bin/python`

3. 示例命令：

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/data_1/yds/home/miniconda3/bin/python \
  /mnt/data_1/yds/多模态/权重模块/run_query_lr.py \
  --query "请判断图中所示的消化器官是哪一个？" \
  --pretty
```

## 11. 旧接口保留说明

如果你后面还要回查历史逻辑，下面这套旧接口仍然可用：

- `/mnt/data_1/yds/多模态/权重模块/run_weight_module.py`
- `/mnt/data_1/yds/多模态/权重模块/multimodal_weight_service.py`
- `/mnt/data_1/yds/多模态/权重模块/semantic_density_service.py`

但它的目标是：

- 返回 `L0-L4` 五分类
- 返回 `R1-R3`

所以它不再是当前推荐对外入口。
