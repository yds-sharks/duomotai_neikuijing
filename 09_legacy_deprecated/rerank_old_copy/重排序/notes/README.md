# Multimodal PPR Rerank

This package contains the EndoBench multimodal rerank code prepared for git upload.

## Contents

- `src/rerank_multimodal_ppr.py`: main rerank pipeline.
- `scripts/run_rerank_background.sh`: background runner with shard support.
- `scripts/merge_rerank_shards.py`: merge sharded JSONL outputs by `index`.
- `docs/rerank说明.md`: original Chinese notes and experiment context.
- `data/input_manifest.json`: current full input paths, sizes, and target filenames.
- `data/sample/retrieval_export.sample.jsonl`: three-line input sample.
- `output/sample/`: smoke-test output examples.

## Runtime Inputs

The pipeline needs:

- `data/retrieval_export.jsonl`
- `data/multimodal_samples.db`
- `Qwen3-VL-Embedding-2B`, passed with `--model-path`

The current full input files are large, so they are not tracked in this git package. See `data/README.md` and `data/input_manifest.json`.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Use your existing environment if it already contains `sentence-transformers`, `torch`, `transformers`, and the Qwen3-VL remote code dependencies.

## Smoke Run

```bash
python src/rerank_multimodal_ppr.py \
  --retrieval-export-jsonl data/retrieval_export.jsonl \
  --main-db data/multimodal_samples.db \
  --model-path /path/to/Qwen3-VL-Embedding-2B \
  --device cuda:0 \
  --limit 1 \
  --batch-size 4 \
  --output-jsonl output/endobench_multimodal_ppr_top3_smoke.jsonl \
  --debug-jsonl output/endobench_multimodal_ppr_top3_smoke_debug.jsonl \
  --overwrite
```

## Full Run

```bash
python src/rerank_multimodal_ppr.py \
  --retrieval-export-jsonl data/retrieval_export.jsonl \
  --main-db data/multimodal_samples.db \
  --model-path /path/to/Qwen3-VL-Embedding-2B \
  --device cuda:0 \
  --batch-size 4 \
  --output-jsonl output/endobench_multimodal_ppr_top3.jsonl \
  --debug-jsonl output/endobench_multimodal_ppr_top3_debug.jsonl \
  --error-jsonl output/endobench_multimodal_ppr_top3_errors.jsonl \
  --progress-log output/endobench_multimodal_ppr_top3_progress.log
```

## Sharded Run

```bash
NUM_SHARDS=4 SHARD_INDEX=0 GPU_ID=0 \
PYTHON_BIN=/path/to/python \
MODEL_PATH=/path/to/Qwen3-VL-Embedding-2B \
RETRIEVAL_EXPORT_JSONL=/path/to/retrieval_export.jsonl \
MAIN_DB=/path/to/multimodal_samples.db \
bash scripts/run_rerank_background.sh
```

Run `SHARD_INDEX=0..3`, then merge:

```bash
python scripts/merge_rerank_shards.py \
  --inputs output/endobench_multimodal_ppr_top3.shard*.jsonl \
  --output output/endobench_multimodal_ppr_top3.jsonl
```

## Current Parameters

- text candidates: `retrieval.text_top20`
- image candidates: `retrieval.image_top20`
- `top_m_neighbors=5`
- `pair_boost=0.15`
- `pair_min_weight=0.45`
- `alpha=0.7`
- `ppr_iters=10`
- `select_k=3`
- `redundancy_weight=0.3`
- `pair_complete_bonus=0.1`
- `min_gain=0.05`
