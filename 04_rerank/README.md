# 图文重排序

## 项目目标
面向医学多模态 RAG 场景（EndoBench 数据集），研究图文混合检索结果的重排序方法。当前重点是 ODSC（Option-Discriminative Submodular Composition）方法，目标是发论文。

## 子模块

| 子模块 | 职责 | 输入 | 输出 |
|--------|------|------|------|
| 重排序 | ODSC 图文重排序 + PPR baseline | retrieval_export.jsonl | 离线评测 retrieval.jsonl |

## 数据流

```
retrieval_export.jsonl (text_top20 + image_top20)
    → [ODSC 重排序 / PPR baseline]
    → retrieval.jsonl (top-3 证据)
    → [下游大模型 RAG 评测]
```

## 目录结构

```
图文重排序/
├── README.md
├── CLAUDE.md
├── .claude/commands/
└── 重排序/
    ├── MODULE_SPEC.md
    ├── code/
    ├── config/
    ├── data/
    │   ├── tmp/
    │   └── persistent/
    │       └── bad_cases/
    ├── instructions/
    └── notes/
        └── ops/
```
