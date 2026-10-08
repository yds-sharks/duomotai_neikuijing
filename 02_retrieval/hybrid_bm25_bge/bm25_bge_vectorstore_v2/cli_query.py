import argparse
import json
import yaml

from .retrieval_service import HybridRetriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--topk", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    if args.topk is not None:
        cfg["retrieval"]["topk"] = args.topk
    svc = HybridRetriever(cfg)
    out = svc.search(args.query)
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()