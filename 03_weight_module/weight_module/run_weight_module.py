#!/usr/bin/env python3
"""CLI entrypoint for the current text-only weight module."""

import argparse
import json

from multimodal_weight_service import MultimodalWeightService


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="输入文本 query，输出 L0-L4 与 R1-R3 标签")
    parser.add_argument("--config", default="/mnt/data_1/yds/多模态/权重模块/weight_module_runtime.yaml")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    with MultimodalWeightService(config_path=args.config) as service:
        result = service.analyze(query_text=args.query)

    if args.pretty:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
