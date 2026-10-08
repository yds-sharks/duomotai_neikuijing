import argparse
import yaml

from .insert_pipeline import Inserter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    ins = Inserter(cfg)
    ins.insert_dir(cfg["paths"]["input_root"])
    print("✅ Insert finished.")

if __name__ == "__main__":
    main()