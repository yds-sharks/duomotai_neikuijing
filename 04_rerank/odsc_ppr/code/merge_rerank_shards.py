#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge sharded rerank jsonl outputs by index.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged: Dict[int, dict] = {}
    for input_path in args.inputs:
        path = Path(input_path)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                merged[int(row["index"])] = row

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for index in sorted(merged):
            handle.write(json.dumps(merged[index], ensure_ascii=False) + "\n")

    print(f"merged={len(merged)} output={output_path}")


if __name__ == "__main__":
    main()
