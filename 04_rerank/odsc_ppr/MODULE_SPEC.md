# 重排序 模块规范

## 模块职责
对图文搜索结果进行重排序优化（ODSC: Option-Discriminative Submodular Composition）

## 目录结构

重排序/
├── MODULE_SPEC.md
├── code/
├── config/
├── data/
│   ├── tmp/
│   └── persistent/
│       └── bad_cases/
├── instructions/
└── notes/
    └── ops/                 # 操作记录（方案 + 步骤日志）

## 文件命名

所有文件以 `重排序_` 为前缀：
- 中间产物: `重排序_{序号}_{步骤}.ext`
- 最终产出: `重排序_{语义名}.ext`
- 日志: `重排序_{名称}_log.ext`

## 核心代码

| 文件 | 说明 |
|------|------|
| code/rerank_odsc.py | ODSC 重排序主脚本（新方法） |
| code/rerank_multimodal_ppr.py | PPR 重排序主脚本（baseline） |
| code/run_odsc_background.sh | ODSC 后台运行脚本 |
| code/run_rerank_background.sh | PPR 后台运行脚本 |
| code/merge_rerank_shards.py | 分片合并工具 |

## 上下游关系
- 输入来源: retrieval_export.jsonl（检索导出，每题 text_top20 + image_top20）
- 输出去向: 离线评测脚本 evaluate_from_retrieval_jsonl.py

## 关键注意事项
- PPR 方法效果为负提升，根因是：原始检索分数被丢弃、跨模态 embedding gap、图文配对失效
- ODSC 方法的方法说明见 instructions/odsc_method.md
- 图片候选编码必须用 {image + text} 联合编码，不能只传图片像素
