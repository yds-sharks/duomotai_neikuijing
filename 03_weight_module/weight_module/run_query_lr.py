#!/usr/bin/env python3
"""CLI entrypoint for query-only coarse L + R inference."""

import argparse
import json

from query_lr_service import QueryLRService


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="输入文本 query，输出 L0/L1 与 R1/R2/R3 标签")
    parser.add_argument("--root-dir", default="/mnt/data_1/yds/多模态/权重模块")
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda/cuda:0 ...")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    service = QueryLRService(root_dir=args.root_dir, device=args.device)
    result = service.predict(args.query)

    if args.pretty:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
