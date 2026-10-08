# EndoBench 消融评测包（A100 服务器一键运行）

在 stratified_1000（1,002 题固定分层子集）上跑机制消融 C / D / G。
统一设置：u50 controller、Qwen3-VL-8B 冻结生成器、v2 权重模块 + 解剖路由、
最大 3 轮、同一候选缓存与索引、ParseFail/预算 → ACCEPT with current M+（空 M+ → closed-book）。

## 前置条件

- 2× A100 80GB（单卡即可放下整套模型，每卡跑一个配置）
- Python 3.10+，可访问 GitHub 与 HF（默认走 hf-mirror.com）
- 私有仓库需先 `gh auth login`（GitHub CLI）；或让仓库所有者临时设为 public

## 一键安装（约 30–60 分钟，含模型下载）

```bash
git clone https://github.com/yds-sharks/agentic.git
cd agentic/endobench_eval_pkg
bash setup_env.sh
```

setup_env.sh 会自动：建 venv → 装依赖 → 下载公开模型（Qwen3-VL-8B / bge-m3 /
mengzi-bert-base）→ 从 Release 下载并重组 u50 checkpoint、Milvus DB、权重模块、
候选缓存 → 写入路径配置 → 跑 10 题冒烟自检。

## 启动夜间任务

```bash
nohup bash run_overnight.sh > logs/overnight.log 2>&1 &
```

- wave1：C（GPU0）+ D（GPU1），约 2.5–3 小时
- wave2：G（GPU0），约 2.5–3 小时
- 结果：`results/<CFG>/agentic_samples.jsonl` + `agentic_summary.json`，日志在 `logs/`

## 手动单跑 / 换 controller

```bash
bash run_config.sh G 1                                   # 配置 G 在 GPU1
CTRL_CKPT=models/ckpt_ctrl_rft_v3 OUT_SUFFIX=_rft bash run_config.sh A 0
```

配置定义（对应论文消融表）：

| 配置 | 含义 | 开关 |
|---|---|---|
| A | Clean Full（M+ 与 M- 跨轮持久化，learned KEEP） | — |
| B | No Memory | `--no-mplus --no-mminus` |
| C | Selection-only（禁 REWRITE） | `--selection-only` |
| D | Rewrite/Search-only（固定 Top-k） | `--rewrite-only` |
| E | M+ Only | `--no-mminus` |
| F | M- Only | `--no-mplus` |
| G | Sequential rewrite→retrieval→score-rerank | `--sequential` |

## 目录结构

```
endobench_eval_pkg/
├── eval_endobench.py      # 主评测脚本
├── code/                  # controller/检索适配/打分器等模块
├── retrieval_mm/          # Milvus 多模态检索引擎
│   └── build_multimodal_milvus.py  # （可选）从 assets/multimodal_samples.db 重建索引；评测无需运行，DB 已由 Release 直接提供
├── weight_module/         # v2 权重模块（checkpoints 由 Release 提供）
├── assets/                # Milvus DB / 候选缓存 / 翻译缓存（下载后生成）
├── models/                # u50 与公开模型（下载后生成）
├── stratified_1000.jsonl  # 固定评测子集（QID 与本机其他配置完全一致）
└── smoke10.jsonl          # 10 题冒烟子集
```
