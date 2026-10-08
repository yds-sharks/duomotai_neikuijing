#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from collections import Counter
from pathlib import Path


def iter_jsonl(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_glob", required=True, help="Glob for shard summary jsonl files.")
    args = ap.parse_args()

    files = sorted(Path().glob(args.summary_glob))
    if not files:
        raise SystemExit(f"No files matched: {args.summary_glob}")

    rows = []
    for f in files:
        rows.extend(iter_jsonl(f))

    status = Counter(r.get("status") for r in rows)
    msg = Counter(r.get("msg") for r in rows)
    print("files", len(files))
    print("rows", len(rows))
    print("status", dict(status))
    print("top_msgs", msg.most_common(10))


if __name__ == "__main__":
    main()
