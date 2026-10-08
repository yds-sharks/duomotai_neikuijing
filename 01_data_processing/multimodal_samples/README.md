# multimodal_samples.db 说明

## 1. 设计目标

本目录中的统一主数据库为：

- `/mnt/data_1/yds/多模态/data_house/multimodal_samples.db`

该库作为唯一数据源（Source of Truth），用于：

- 统一存储纯文本样本与图文对样本
- 统一存储器官标签与知识标签
- 支持后续检索、过滤、切分与向量库构建
- 避免为纯文本与图文数据建立多个物理主库

## 2. 主表设计

数据库为 SQLite，核心主表只有一张：

- `multimodal_samples`

一个 `sample` 表示一条最小知识单元：

- 纯文本：一段文本
- 图文对：一组文本 + 一组图片

## 3. 字段说明

### 3.1 标识字段

- `sample_id`
  - 样本唯一 ID，主键
- `group_id`
  - 组 ID，用于绑定同一病例、同一组图文或后续需要聚合的数据
  - 语义上独立于 `sample_id`

### 3.2 来源字段

- `source_type`
  - `text_only` 或 `image_text_pair`
- `have_image`
  - 是否存在图片
  - 纯文本固定为 `0`
  - 图文对固定为 `1`
- `doc_id`
  - 文档 ID
- `doc_name`
  - 文档名
- `source_path`
  - 数据来源路径

### 3.3 定位字段

- `page_idx`
  - 页码，图文对可为空
- `block_id`
  - 原文本块 ID，图文对可为空
- `content_hash`
  - 内容哈希
  - 纯文本沿用原库字段
  - 图文对使用 `image_path + text` 生成哈希

### 3.4 主内容字段

- `text`
  - 统一文本入口
  - 纯文本样本存文本块内容
  - 图文对样本存图片描述文本
- `text_role`
  - `knowledge_block` 或 `image_description`

### 3.5 标签字段

- `organ_tags`
  - JSON 数组字符串
  - 表示器官标签（Level1）
- `primary_knowledge_type`
  - 主知识标签（Level2 主标签）
- `secondary_knowledge_types`
  - JSON 数组字符串
  - 次知识标签（Level2 辅标签）

### 3.6 统一标签结构

- `labels`
  - JSON 对象字符串
  - 统一 downstream 使用

格式为：

```json
{
  "level1": ["胃"],
  "level2_main": "病变特征",
  "level2_aux": ["诊断评估"]
}
```

### 3.7 检索辅助字段

- `retrieval_flags`
  - JSON 对象字符串
- `retrieval_meta`
  - JSON 对象字符串

`retrieval_flags` 格式：

```json
{
  "is_general": false,
  "is_mixed": false,
  "is_key_knowledge": false
}
```

生成规则：

- `is_general = "通用" in organ_tags`
- 若包含 `通用`，则 `is_mixed = false`
- 否则 `is_mixed = len([x for x in organ_tags if x != "通用"]) >= 2`
- `is_key_knowledge = false`

`retrieval_meta` 格式：

```json
{
  "body_site_main": "胃",
  "body_site_all": ["胃", "十二指肠"],
  "knowledge_type_main": "病变特征"
}
```

生成规则：

- 若包含 `通用`，则 `body_site_main = "通用"`
- 否则 `body_site_main = organ_tags[0]`，若为空则置空字符串
- `body_site_all = organ_tags`
- `knowledge_type_main = primary_knowledge_type`

### 3.8 图像字段

- `images`
  - JSON 数组字符串
  - 仅存图片路径等轻量元数据，不存图片二进制

格式为：

```json
[
  {
    "image_id": "img_xxx",
    "image_path": "/abs/path/to/image.jpg",
    "image_type": "unknown",
    "is_primary": true
  }
]
```

约束：

- 纯文本：`[]`
- 图文对：至少一张图

### 3.9 版面与扩展字段

- `bbox`
  - JSON 数组字符串或空
- `coord_sys`
  - 坐标系统
- `created_at`
  - 样本创建时间
- `extra`
  - JSON 对象字符串
  - 用于保留来源与补充信息

## 4. 标签约束

### 4.1 organ_tags 约束

`organ_tags` 为保留原始语义的多标签字段。

允许示例：

- `["食管"]`
- `["胃"]`
- `["胃", "十二指肠"]`
- `["通用"]`
- `["通用", "胃"]`
- `["通用", "食管", "胃"]`

解释规则：

- 只要包含 `通用`，该样本就视为通用类样本
- 当 `通用` 与具体器官同时出现时，具体器官标签仍然保留
- 当 `通用` 与具体器官同时出现时，不按跨部位 mixed 处理

### 4.2 知识标签约束

- `primary_knowledge_type` 为主标签
- `secondary_knowledge_types` 为辅标签数组

## 5. 当前数据来源

统一主库由以下两类来源构造：

### 5.1 纯文本来源

- `/mnt/data_1/yds/多模态/data_house/origin_data/blocks_classification/text_database_stage3_filtered_labeled.db`

映射规则：

- `source_type = text_only`
- `have_image = 0`
- `text = text_blocks.text`
- `text_role = knowledge_block`
- `organ_tags`、`primary_knowledge_type`、`secondary_knowledge_types` 直接继承原标注库
- `images = []`

### 5.2 图文对来源

- `/mnt/data_1/yds/多模态/data_house/origin_data/image_text/output_pairs_all_min_filtered.jsonl`

映射规则：

- `source_type = image_text_pair`
- `have_image = 1`
- `text = final_description`
- `text_role = image_description`
- `images` 中仅保存图片路径及轻量元信息
- 当前版本标签先置空：
  - `organ_tags = []`
  - `primary_knowledge_type = ""`
  - `secondary_knowledge_types = []`

## 6. 两类样本的大体格式

### 6.1 纯文本样本

```json
{
  "sample_id": "txt_a0281c14_15",
  "group_id": "grp_txt_a0281c14_15",
  "source_type": "text_only",
  "have_image": 0,
  "doc_id": "a0281c14",
  "page_idx": 2,
  "block_id": 15,
  "text": "本书中所涉及的药剂，在日本和中国，有可能会出现非医疗保险适应证的情况。",
  "text_role": "knowledge_block",
  "organ_tags": ["通用"],
  "primary_knowledge_type": "其他",
  "secondary_knowledge_types": [],
  "labels": {
    "level1": ["通用"],
    "level2_main": "其他",
    "level2_aux": []
  },
  "retrieval_flags": {
    "is_general": true,
    "is_mixed": false,
    "is_key_knowledge": false
  },
  "retrieval_meta": {
    "body_site_main": "通用",
    "body_site_all": ["通用"],
    "knowledge_type_main": "其他"
  },
  "images": []
}
```

### 6.2 图文对样本

```json
{
  "sample_id": "imgtxt_a0281c14_000001",
  "group_id": "grp_imgtxt_a0281c14_000001",
  "source_type": "image_text_pair",
  "have_image": 1,
  "doc_id": "a0281c14",
  "text": "食管M1型为主的0-IIc病变在喷洒碘后显示3cm大小的不染色区域。",
  "text_role": "image_description",
  "organ_tags": [],
  "primary_knowledge_type": "",
  "secondary_knowledge_types": [],
  "labels": {
    "level1": [],
    "level2_main": "",
    "level2_aux": []
  },
  "retrieval_flags": {
    "is_general": false,
    "is_mixed": false,
    "is_key_knowledge": false
  },
  "retrieval_meta": {
    "body_site_main": "",
    "body_site_all": [],
    "knowledge_type_main": ""
  },
  "images": [
    {
      "image_id": "img_a0281c14_000001",
      "image_path": "/abs/path/to/image.jpg",
      "image_type": "unknown",
      "is_primary": true
    }
  ]
}
```

## 7. 构造脚本

统一主库构造脚本为：

- `/mnt/data_1/yds/多模态/data_house/build_multimodal_samples.py`

运行方式：

```bash
cd /mnt/data_1/yds/多模态/data_house
python3 build_multimodal_samples.py
```

若需要覆盖已存在的输出库：

```bash
cd /mnt/data_1/yds/多模态/data_house
python3 build_multimodal_samples.py --overwrite
```
