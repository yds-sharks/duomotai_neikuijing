#!/usr/bin/env python3
"""CLI wrapper for semantic density inference."""

import argparse
import json

from semantic_density_service import SemanticDensityService


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", help="single query inference")
    parser.add_argument("--query-a", help="pairwise compare: query a")
    parser.add_argument("--query-b", help="pairwise compare: query b")
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda/cuda:0 ...")
    parser.add_argument("--root-dir", default="/mnt/data_1/yds/多模态/权重模块")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    service = SemanticDensityService(root_dir=args.root_dir, device=args.device)

    if args.query:
        result = service.predict(args.query)
    elif args.query_a and args.query_b:
        result = service.compare(args.query_a, args.query_b)
    else:
        raise SystemExit("请提供 --query，或同时提供 --query-a 和 --query-b。")

    if args.pretty:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
