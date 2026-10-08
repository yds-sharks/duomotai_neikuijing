# 图文重排序 — 环境配置文档

由 env_setup agent 维护。

## Python 环境
- 版本: 项目专用 venv
- 路径: /mnt/data_1/yds/venvs/qwen3vl-rerank/bin/python
- 注意: 不使用系统 Python，必须用此 venv

## 核心依赖（venv 内已安装）
| 包名 | 用途 |
|------|------|
| torch | 模型推理 |
| transformers | Qwen3-VL 模型加载 |
| sentence-transformers | 嵌入计算 |
| numpy | 数值计算 |
| Pillow | 图片处理 |

## 外部资源（不在仓库内）
| 资源 | 路径 | 大小 |
|------|------|------|
| 输入数据 | /mnt/data_10/mwx/workspace/.../retrieval_export.jsonl | 117M |
| 图片数据库 | /mnt/data_1/yds/多模态/data_house/multimodal_samples.db | 390M |
| Qwen3-VL 模型 | /mnt/data_1/yds/models/Qwen/Qwen3-VL-Embedding-2B | 4.3G |

## 已知问题
参见踩坑记录库: /mnt/workspace/yds/工作台/config/env_pitfalls.md
