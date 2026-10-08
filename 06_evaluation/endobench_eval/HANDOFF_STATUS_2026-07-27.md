# 实验进度与代码状态交接文档（2026-07-27 晚）

> 用途：账户切换后的完整上下文恢复。所有路径基于 `/mnt/data_1/yds/多模态/endobench_eval/`。

---

## 一、当前论文实验设计（最终确定的 4 组消融）

| 配置 | 决策方式 | 记忆 | 最大轮数 | 回答的问题 |
|---|---|---:|---:|---|
| Config 1: One-shot RAG | 直接作答 | 无 | 1 | 普通 RAG 水平 |
| Config 2: Single-round Agent | ACCEPT/REWRITE 一次后作答 | 无 | 1 | 控制器是否有效 |
| Config 3: Iterative Agent (R3-NM) | 不断 REWRITE 直到 ACCEPT/预算耗尽 | 无 | 3 | 多轮探索是否有效 |
| Config 4: Memory-aware Agent | 不断 REWRITE 直到 ACCEPT/预算耗尽 | **M+ 与 M-** | 3 | 记忆是否进一步提升 |

旧版 7 组消融（A-G + SFT/RFT）**已废弃**，A100 包的 run_overnight.sh 计划也不再需要。

---

## 二、已确定的全量结果（6832 题）

| Configuration | Rounds | Persistent history | Acc. | 数据文件 |
|---|---|---|---|---|
| One-shot RAG | 1 | No | **38.85%** | `results_v2/vanilla_rag_samples.jsonl` |
| Single-round Agent | 1 | No | **40.12%** (2741/6832) | `results_v2/agentic_samples.jsonl` |
| Iterative Agent (R3-NM) | ≤3 | No | **跑到一半**（见第四节） | `results_full/R3_NM_shard{0,1}/` |
| Memory-aware Agent | ≤3 | Yes (det. M+) | **42.84%** (2927/6832) | rescue+harm 四个 shard 文件 |

其他参考基线：
- baseline（无检索，纯选项 argmax）：40.59%，`results/baseline_samples.jsonl`
- 旧 mr_full（2 轮 M+/M-，M+ 空转）：39.80%，`results_v2/mr_full/`（不再用于论文）

### Config 4（42.84%）的构成——五项已确认结果

1. **准确题数 = 2927/6832 = 42.8425%**
2. **四象限**（vs 单轮 40.12%）：
   - C→C = 2491（单轮对→3轮仍对）
   - C→W = 250（单轮对→3轮变错，harm）
   - W→C = 436（单轮错→3轮救回，rescue）
   - W→W = 3655
   - McNemar: b=250, c=436, χ²=49.89, p<0.0001，净增益 +186 题 = +2.72pp
3. **42.81% vs 42.84%**：harm 脚本 merge 代码硬编码 rescued=434（旧计数），实际 rescue 数据是 436。论文用 **42.84% (2927/6832)**。
4. **强制 top-3 M+ 触发率**：orig_action=REWRITE 的 3562/6832 = **52.1%** 触发强制 top-3 注入；ACCEPT 的 3270 题（47.9%）用模型自选 keep。
5. **Config 4 统计**：mean_rounds=2.39，mean_retr_calls=2.39，ParseFail=3.94%（269/6832）
   - Config 3 partial（前1862题）：mean_rounds=1.75，ParseFail=6.23%，多轮率46.5%

### Config 4 数据文件（rescue + harm，共 6832 题）

- 错题重跑（4091 题）：`rescue_full_u50_shard0.jsonl`（2045）+ `rescue_full_u50_shard1.jsonl`（2046）
- 对题重跑（2741 题）：`rescue_harm_u50_shard0.jsonl` + `rescue_harm_u50_shard1.jsonl`
- 运行脚本：`rescue_3round.py`（独立脚本），启动脚本 `run_rescue_full_u50.sh` / `run_harm_test_u50.sh`

---

## 三、四组配置详表（论文 Setup 部分用）

| 字段 | Config 1 | Config 2 | Config 3 (R3-NM) | Config 4 (rescue) |
|---|---|---|---|---|
| 检索前端权重+部位模块 | 权重✓+部位✓ | 权重✓+部位✓ | 权重✓+部位✓ | 部位✓，**权重✗**（rescue_3round.py 无此参数，固定 select_k=6/6） |
| 最大 controller 轮数 | 0 | 1 | 3 | 3 |
| REWRITE 后再调 controller | 否 | 否 | 是 | 是 |
| 是否执行 KEEP/DROP | 否 | 是 | 是 | 是 |
| empty KEEP 处理 | N/A | 全12候选注入（旧代码） | Top-5 | 不更新证据，继续下轮 |
| M+ 跨轮保留 | 否 | 否 | 否 | 是（REPLACE；REWRITE 强制 top-3，ACCEPT 用模型 keep） |
| M- 启用 | 否 | 否 | 否 | 是（无上限追加） |
| 预算耗尽处理 | N/A | N/A | Top-5 | 复用单轮原始答案 |
| ParseFail 处理 | N/A | 全12候选注入 | Top-5 | 跳过该轮，继续 |
| 最终证据规则 | 全部12初始候选 | ACCEPT→keep；REWRITE→top-5 | ACCEPT→keep(空→top5)；预算/ParseFail→top5 | 最后非空keep；全空→复用单轮答案（harm 组 840/2741=30.6% 触发，全部保持正确） |

**论文措辞要求（用户明确指定）**：Config 4 必须称为 *"memory-aware agent with deterministic M+ initialization"*，不能把强制 top-3 描述成控制器自主学会的证据保留，且要报告 52.1% 触发比例。

---

## 四、正在跑的实验：R3-NM（Config 3，全量 6832 题）

### 状态（2026-07-27 ~19:00 检查点）

| Shard | GPU | 进度 | 速度 | 预计完成 |
|---|---|---|---|---|
| shard0 | 0+1 | 1232/3416 (36%) | 6.6s/题 | ~21:00 |
| shard1 | 2+3 | 714/3416 (21%) | 10.8s/题 | ~01:00（次日凌晨） |

- 日志：`logs/r3nm_shard0.log`、`logs/r3nm_shard1.log`
- 输出：`results_full/R3_NM_shard0/`、`results_full/R3_NM_shard1/`
- 进程：两个 `eval_endobench.py`（PID 1050988 / 1056420，注意账户切换后确认进程仍存活）
- shard0 日志中偶发 Milvus GOAWAY too_many_pings 警告，**不影响运行**
- partial 结果（前1862题）acc=43.34%，但前段偏简单题，全量可能回落

### 启动命令（如需重启，脚本 `run_r3nm_full.sh`；注意 shard1 需手动加 milvus-db-path）

```bash
cd /mnt/data_1/yds/多模态/endobench_eval
PY=/mnt/data_1/yds/venvs/qwen35-train/bin/python
CKPT=/mnt/data_1/yds/多模态/rerank_image_and_text/agentic/train/ckpt_grpo_v4_u50

# shard0 (GPU 0/1)
CUDA_VISIBLE_DEVICES=0,1 $PY eval_endobench.py --mode agentic --ctrl-model $CKPT \
  --ctrl-device cuda:0 --gen-device cuda:1 \
  --retr-image-device cuda:0 --retr-text-device cuda:1 --weight-device cuda:0 \
  --cand-cache results_v2/cand_cache_v2.jsonl \
  --translate-cache endobench_translated_queries.jsonl \
  --use-organ-filter --use-weight-module --max-rounds 3 --latency \
  --no-mplus --no-mminus \
  --out-dir results_full/R3_NM_shard0 --offset 0 --limit 3416 \
  2>&1 | tee -a logs/r3nm_shard0.log &

# shard1 (GPU 2/3) —— 必须等 shard0 出现 "[retr] online" 再启动，且指定独立 DB 副本
CUDA_VISIBLE_DEVICES=2,3 $PY eval_endobench.py --mode agentic --ctrl-model $CKPT \
  --ctrl-device cuda:0 --gen-device cuda:1 \
  --retr-image-device cuda:0 --retr-text-device cuda:1 --weight-device cuda:0 \
  --cand-cache results_v2/cand_cache_v2.jsonl \
  --translate-cache endobench_translated_queries.jsonl \
  --use-organ-filter --use-weight-module --max-rounds 3 --latency \
  --no-mplus --no-mminus \
  --out-dir results_full/R3_NM_shard1 --offset 3416 --limit 3416 \
  --milvus-db-path /mnt/data_1/yds/多模态/data_house/milvus/multimodal_vector_indexes_shard1.db \
  2>&1 | tee -a logs/r3nm_shard1.log &
```

**脚本支持断点续跑**（samples.jsonl 里已有的 qid 自动跳过），中断后直接重跑同一命令即可。

### 跑完后的处理步骤

1. 合并统计两个 shard：
```python
import json
recs=[]
for sh in ['R3_NM_shard0','R3_NM_shard1']:
    recs += [json.loads(l) for l in open(f'results_full/{sh}/agentic_samples.jsonl')]
acc = sum(r['correct'] for r in recs)/len(recs)
print(f"R3-NM: {sum(r['correct'] for r in recs)}/{len(recs)} = {acc*100:.2f}%")
```
2. 计算两个关键量：
   - Δ_iteration = A_R3NM − 40.12（多轮探索的贡献）
   - Δ_memory = 42.84 − A_R3NM（记忆的贡献）
3. 统计 mean_rounds、mean_retr_calls、ParseFail 率、closed-book 数（应为 0 或接近 0）
4. 与 Config 2/4 做逐 qid McNemar 配对检验

---

## 五、本地代码修改记录（本轮会话）

### `eval_endobench.py`（核心评测脚本，699 行）——按 CLEAN_MEMORY_EXPERIMENT_SPEC 修改

规格文件：`CLEAN_MEMORY_EXPERIMENT_SPEC_2026-07-28.md`（当前打开的文件）

4 处关键修改（相对旧版）：
1. 新增 `_final_evidence()`（约 L278）：预算耗尽/ParseFail 回退 = dedup(M+ ∪ top-5)，上限 5，**绝不 closed-book**
2. ParseFail（约 L369）：回退 `_final_evidence()`（旧版是注入全部候选，虚高）
3. ACCEPT 空 keep（约 L383）：回退 `_fixed_topk_evidence()` top-5（旧版注入全部候选）
4. REWRITE 状态更新（约 L396）：M+ 改 **UNION+去重+cap 5**（旧版 REPLACE）；M- **cap 2**（旧版无限追加）

注意：**Config 2 的 40.12% 是旧代码跑的**（ParseFail/空keep 注入全 12 候选），Config 3 是新代码。两者 ParseFail 语义不同，但 Config 2 ParseFail 率仅 0.86%（59 题），影响 <0.1pp，可在论文脚注说明。

### `rescue_3round.py`（Config 4 专用脚本，504 行，未改动）

关键设计（与 eval_endobench.py 不同）：
- Round 0 ACCEPT **不 break**，强制所有题跑 2-3 轮
- REWRITE 时 M+ = **强制注入 top-3**（`--mplus-topk 3`）；ACCEPT 时 M+ = 模型 keep
- M+ 更新 = REPLACE；M- 无上限
- 无证据存活 → 复用单轮原始答案（`fallback_to_orig`）
- **无 `--use-weight-module` 参数**（与其他 Config 的一个不一致点）

### A100 部署包（`_pkg/agentic/endobench_eval_pkg/`，已 push 到 git）

- commit 6eb0750：补充遗漏的 `retrieval_mm/build_multimodal_milvus.py`（625 行），路径改环境变量
- commit a3937df：requirements.txt 加 torchvision/sentence-transformers/scipy/scikit-learn；run_config.sh 与 setup_env.sh 加 AGENTIC_ROOT；semantic_density_service.py 默认路径改环境变量
- **A100 现有的 A-G 消融计划已废弃，不需要再跑**（4 组数据本地全部凑齐）

---

## 六、重要结论备忘（论文写作依据）

1. **旧 M+/M- 机制的真相**：controller（u50）97.4% 的 REWRITE 轮 keep=[]，M+ 名存实亡（mr_full 的 mean_retained=0.0）；M- 自动注入所有多轮样本。旧 2 轮 mr_full 39.80% < 单轮 40.12% 的原因就是 M+ 空转 + 旧版 closed-book 回退。
2. **Config 4 增益全来自 REWRITE 题**：REWRITE 类 rescue 342 / harm 137 = 净+205；ACCEPT 类 rescue 94 / harm 113 = 净-19。多轮+强制 M+ 只对"需要改写的难题"有效。
3. **harm 组的 fallback**：840/2741（30.6%）3 轮无证据产出，复用单轮正确答案而保持正确。若严格改为 top-5 回退，Config 4 会掉到约 35-36%。这是设计选择，论文需说明"多轮失败时保持原答案"的语义。
4. **1轮 acc > 2/3轮 acc 是选择偏差**：单轮收尾的是简单题，多轮的是难题；正确比法是同题配对（四象限/McNemar）。
5. **权重模块**（mengzi-bert 密度/依赖分类器）通过 `WEIGHT_MODULE_DIR` 环境变量加载，R1/R2/R3 依赖级别自适应调整 text/image select_k（R3: 图+2文-2；R1: 文+2图-2）。

---

## 七、关键路径速查

| 内容 | 路径 |
|---|---|
| 评测主脚本 | `endobench_eval/eval_endobench.py` |
| Config 4 脚本 | `endobench_eval/rescue_3round.py` |
| 实验规格 | `endobench_eval/CLEAN_MEMORY_EXPERIMENT_SPEC_2026-07-28.md` |
| Python 环境 | `/mnt/data_1/yds/venvs/qwen35-train/bin/python` |
| Controller ckpt | `rerank_image_and_text/agentic/train/ckpt_grpo_v4_u50` |
| 生成器 | Qwen3-VL-8B-Instruct（DEFAULT_GEN_MODEL 见 eval 脚本 L50） |
| Milvus 主库 | `data_house/milvus/multimodal_vector_indexes.db` |
| Milvus 副本（shard1 并行用） | `data_house/milvus/multimodal_vector_indexes_shard1.db` |
| 候选缓存 | `endobench_eval/results_v2/cand_cache_v2.jsonl`（6832 题全缓存） |
| 翻译缓存 | `endobench_eval/endobench_translated_queries.jsonl` |
| 权重模块 | `/mnt/data_1/yds/多模态/权重模块`（env: WEIGHT_MODULE_DIR） |
| Agentic 代码根 | `rerank_image_and_text/agentic`（env: AGENTIC_ROOT） |
| 历史进度文档 | `endobench_eval/PROGRESS.md` |

### 已知坑

- Milvus Lite 同一 DB 文件不能两进程同时初始化 → 并行时第二个进程必须 `--milvus-db-path` 指向副本，且错开启动（等第一个打印 `[retr] online`）
- scipy 在系统 python3 没装，统计检验用 venv python 或手算 McNemar χ²=(|b−c|−1)²/(b+c)
- `python` 命令不存在，用 `/mnt/data_1/yds/venvs/qwen35-train/bin/python`

---

## 八、下一步待办

1. ⏳ 等 R3-NM 两 shard 跑完（shard1 预计次日 ~01:00）
2. 合并计算 R3-NM 全量 acc，填入主表，算 Δ_iteration 与 Δ_memory
3. 对 Config 2/3/4 做逐 qid McNemar 配对 + bootstrap 95% CI（可参考 `run_stat_significance.py`）
4. 主表定稿：

| Configuration | Rounds | Persistent history | Acc. | Δ Previous | Calls |
|---|---|---|---|---|---|
| One-shot RAG | 1 | No | 38.85 | – | 1.00 |
| Single-round Agent | 1 | No | 40.12 | +1.27 | 1.00 |
| Iterative Agent | ≤3 | No | 待出 | 待算 | ~1.75 |
| Memory-aware Agent | ≤3 | Yes (det. M+) | 42.84 | 待算 | 2.39 |

5. （可选，用户未决定）用统一的 eval_endobench.py 新语义重跑一版干净 R3-MEM 全量，替代 rescue+harm 拼合的 Config 4；当前结论是接受 42.84% 并在论文中如实描述 deterministic M+ initialization
