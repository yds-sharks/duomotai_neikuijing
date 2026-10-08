#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_text(text: str, strip_newlines: bool, collapse_whitespace: bool) -> str:
    s = text or ""
    if strip_newlines:
        s = s.replace("\r", " ").replace("\n", " ")
    if collapse_whitespace:
        s = re.sub(r"\s+", " ", s)
    return s.strip()


def hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def keep_or_reason(
    text_norm: str,
    min_len: int,
    max_len: int,
    low_conf_patterns: List[str],
    seen_hashes: set,
) -> Tuple[bool, str]:
    if not text_norm:
        return False, "empty"
    if len(text_norm) < min_len:
        return False, "too_short"
    if len(text_norm) > max_len:
        return False, "too_long"
    for p in low_conf_patterns:
        if p and p in text_norm:
            return False, f"low_conf_pattern:{p}"
    h = hash_text(text_norm)
    if h in seen_hashes:
        return False, "duplicate_text_norm"
    return True, ""


def ensure_dirs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cleaned candidate pool from raw image-text pairs.")
    parser.add_argument(
        "--config",
        default="/mnt/data_1/yds/多模态/权重模块/pipeline_config.yaml",
        help="Path to pipeline config YAML.",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    input_path = Path(cfg["paths"]["raw_input_jsonl"])
    output_dir = Path(cfg["paths"]["outputs_dir"])
    clean_cfg = cfg["cleaning"]
    ensure_dirs(output_dir)

    candidate_out = output_dir / "candidate_pool.cleaned.jsonl"
    filtered_out = output_dir / "candidate_pool.filtered_examples.jsonl"
    stats_out = output_dir / "candidate_pool.stats.json"

    min_len = int(clean_cfg["min_char_len"])
    max_len = int(clean_cfg["max_char_len"])
    strip_newlines = bool(clean_cfg["strip_newlines"])
    collapse_ws = bool(clean_cfg["collapse_whitespace"])
    patterns = list(clean_cfg.get("drop_low_confidence_patterns", []))
    keep_filtered_examples = int(clean_cfg.get("keep_filtered_examples", 200))

    seen_hashes = set()
    stats = {
        "raw_total": 0,
        "kept_total": 0,
        "filtered_total": 0,
        "reason_counts": {},
    }
    filtered_examples = []

    with input_path.open("r", encoding="utf-8") as fin, candidate_out.open("w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin, start=1):
            stats["raw_total"] += 1
            line = line.strip()
            if not line:
                stats["filtered_total"] += 1
                stats["reason_counts"]["empty_line"] = stats["reason_counts"].get("empty_line", 0) + 1
                continue
            try:
                obj = json.loads(line)
            except Exception:
                stats["filtered_total"] += 1
                stats["reason_counts"]["invalid_json"] = stats["reason_counts"].get("invalid_json", 0) + 1
                continue

            image_path = obj.get("image_path", "")
            raw_text = obj.get("final_description", "")
            text_norm = normalize_text(raw_text, strip_newlines, collapse_ws)

            keep, reason = keep_or_reason(text_norm, min_len, max_len, patterns, seen_hashes)
            if not keep:
                stats["filtered_total"] += 1
                stats["reason_counts"][reason] = stats["reason_counts"].get(reason, 0) + 1
                if len(filtered_examples) < keep_filtered_examples:
                    filtered_examples.append(
                        {
                            "line_no": idx,
                            "reason": reason,
                            "image_path": image_path,
                            "text": raw_text,
                        }
                    )
                continue

            text_hash = hash_text(text_norm)
            seen_hashes.add(text_hash)
            stats["kept_total"] += 1

            rec = {
                "candidate_id": f"cand_{stats['kept_total']:08d}",
                "line_no": idx,
                "image_path": image_path,
                "anchor_text": raw_text,
                "anchor_text_norm": text_norm,
                "text_norm_hash": text_hash,
                "char_len": len(text_norm),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with filtered_out.open("w", encoding="utf-8") as f:
        for row in filtered_examples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats["keep_rate"] = round(stats["kept_total"] / max(stats["raw_total"], 1), 6)
    with stats_out.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"[done] raw={stats['raw_total']} kept={stats['kept_total']} filtered={stats['filtered_total']}")
    print(f"[out ] {candidate_out}")
    print(f"[out ] {filtered_out}")
    print(f"[out ] {stats_out}")


if __name__ == "__main__":
    main()

