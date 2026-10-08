# Agentic MCQ Data Construction

This folder contains the source-grounded multiple-choice data construction tools for
MedAlign-RAG.

The intended flow is:

1. Analyze benchmark type distribution without using benchmark samples for training.
2. Build template MCQ candidates from `multimodal_samples.db`.
3. Validate local schema and leakage constraints.
4. Optionally run an API verifier to filter candidates using source evidence.
5. Use verified MCQs as the environment pool for GPT rewrite validation and later RL.

Benchmark samples should remain held out from training unless the experiment is
explicitly marked as a contaminated/debug run.

## Scripts

- `analyze_benchmark_distribution.py`
  - Reads an EndoBench retrieval export JSONL and reports category/task/subtask distribution.
- `build_agentic_mcq_dataset.py`
  - Builds template MCQs from the local multimodal SQLite database.
  - Splits by `doc_id` hash to avoid same-document leakage across splits.
- `validate_agentic_mcq_dataset.py`
  - Validates JSONL schema, answer consistency, leakage patterns, split counts, and doc leakage.
- `verify_agentic_mcq_with_api.py`
  - Optional OpenAI-compatible verifier for source-grounded filtering.

## Example

```bash
python3 data_construction/build_agentic_mcq_dataset.py \
  --total 4000 \
  --output-dir outputs/mcq_template_4000

python3 data_construction/validate_agentic_mcq_dataset.py \
  --input-jsonl outputs/mcq_template_4000/agentic_mcq_4000.jsonl \
  --report-json outputs/mcq_template_4000/validation_report.json
```
