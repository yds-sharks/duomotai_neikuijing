#!/usr/bin/env python3
"""Real-back-end entry point: run the harness over an EndoBench-style queries jsonl.

Requires vLLM (or any OpenAI-compatible server) for the brain/generator and the
Milvus index + BGE-M3 for retrieval (paths in harness_config.json).

    python run_harness.py --config harness_config.json \
        --queries ../05_agentic_rag/agentic/endobench_translated_queries.jsonl \
        --out runs/first10.jsonl --limit 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime.runner import HarnessRunner, summarize, write_summary  # noqa: E402


def load_queries(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> int:
    ap = argparse.ArgumentParser(description="Central-agent RAG harness")
    ap.add_argument("--config", default=str(Path(__file__).parent / "harness_config.json"))
    ap.add_argument("--queries", required=True, help="jsonl with qid/question/options/answer[/query_image_path]")
    ap.add_argument("--out", required=True, help="output trajectories jsonl (appended)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    runner = HarnessRunner(config, mock=False)
    agent = runner.make_agent()
    items = load_queries(args.queries)
    results = runner.run_batch(agent, items, out_path=args.out, limit=args.limit)
    runner.close()

    summary = summarize(results)
    summary_path = str(Path(args.out).with_suffix(".summary.json"))
    write_summary(
        summary_path,
        summary,
        extra={"queries": args.queries, "out": args.out, "config": args.config, "wall_seconds": runner.last_wall_seconds},
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"trajectories -> {args.out}\nsummary      -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
