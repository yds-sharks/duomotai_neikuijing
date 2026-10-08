# Agentic QA Data Construction

Source-grounded multimodal MCQ data construction for MedAlign-RAG.
Authoritative design: `data_construction_design_zh.md` (Stage 1 + soft labels + answer-utility rollout),
which aligns with the master framework `../medalign_paper_framework_zh.md` (v0.4, unified Agent controller).

## Flow

1. Analyze benchmark type distribution (benchmark samples held out from training).
2. `prepare_qa_source_pool.py` — extract `image_text_pair` sources from `multimodal_samples.db`,
   locate image / origin PDF / page / two-page context; split by `doc_id` hash (no leakage).
3. `route_qa_source_pool.py` — rule-route candidate `query_type`, emit routed candidates + pilot subset.
4. `generate_qa_candidates_with_api.py` — API generation, **must pass image pixels**,
   reuses prompts in `qa_api_prompts.py`. Streams accepted/rejected with resume.
5. `verify_qa_candidates_with_api.py` — blind-answer API verifier, **must pass image pixels**.
   Enforces strict quality gate, deduplicates per source, assembles gold splits.
6. `validate_qa_gold_dataset.py` — local schema / answer consistency / leakage / split checks.

Downstream (see design doc §12.5–16): frozen-generator probe produces keep/drop soft labels;
GPT agent rollout produces `keep/drop + ACCEPT/REWRITE` trajectories scored by answer-utility
(`P_G(a*|E) − P_G(a*|∅)`); training is cold-start SFT → GRPO. There is no standalone
reranker/PPR in this path.

## Scripts

- `analyze_benchmark_distribution.py` — EndoBench category/task/subtask distribution.
- `prepare_qa_source_pool.py` — deterministic source-pool extraction (no API).
- `route_qa_source_pool.py` — rule-based query-type routing + doc_id split + pilot subset.
- `qa_stage1_common.py` — shared helpers (PDF/context location, doc_id split, ids).
- `qa_api_prompts.py` — generation + verification prompt templates (reviewed before API runs).
- `api_client.py` — OpenAI-compatible client with `encode_image_path_to_data_url` (image-capable).
- `qa_api_common.py` — shared API helpers (config, image messages, JSON extraction, usage tracking, StreamWriter, resume).
- `generate_qa_candidates_with_api.py` — image-conditioned candidate generation with resume and usage reports.
- `verify_qa_candidates_with_api.py` — blind-answer verification and gold split assembly.
- `validate_qa_gold_dataset.py` — local schema / leakage / split validator.

## Example

```bash
python3 data_construction/prepare_qa_source_pool.py
python3 data_construction/route_qa_source_pool.py
python3 data_construction/generate_qa_candidates_with_api.py
python3 data_construction/verify_qa_candidates_with_api.py
python3 data_construction/validate_qa_gold_dataset.py \
  --input-jsonl outputs/qa_stage1_verified/qa_gold.jsonl \
  --report-json outputs/qa_stage1_verified/validation_report.json
```
