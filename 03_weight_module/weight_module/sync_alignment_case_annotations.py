#!/usr/bin/env python3
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


MD_PATH = Path("/mnt/data_1/yds/多模态/权重模块/人工审核/06_人工标注输入/alignment_case_pack.v1.md")
JSONL_PATH = Path("/mnt/data_1/yds/多模态/权重模块/人工审核/06_人工标注输入/alignment_case_pack.v1.jsonl")
OUT_JSONL = Path("/mnt/data_1/yds/多模态/权重模块/人工审核/06_人工标注输入/alignment_case_pack.v1.reviewed.jsonl")
OUT_SUMMARY = Path("/mnt/data_1/yds/多模态/权重模块/人工审核/06_人工标注输入/alignment_case_pack.v1.reviewed.summary.json")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_md_annotations(md_text: str) -> List[Dict[str, Any]]:
    pattern = re.compile(
        r"## Case \d+\n- sample_id: (?P<sample_id>[^\n]+).*?- annotation_template:\n```json\n(?P<json>.*?)\n```",
        flags=re.S,
    )
    rows: List[Dict[str, Any]] = []
    for match in pattern.finditer(md_text):
        rows.append(
            {
                "sample_id": match.group("sample_id").strip(),
                "annotation_template": json.loads(match.group("json")),
            }
        )
    return rows


def fill_from_notes(annotation: Dict[str, Any]) -> Dict[str, Any]:
    ann = dict(annotation)
    notes = str(ann.get("notes", "") or "")

    if not ann.get("density_level"):
        match = re.search(r"\bL([0-4])\b", notes)
        if match:
            ann["density_level"] = f"L{match.group(1)}"

    routing = ann.get("routing_preference") or ann.get("image_dependency") or ""
    if not routing:
        match = re.search(r"\bR([1-3])\b", notes)
        if match:
            routing = f"R{match.group(1)}"
    ann["routing_preference"] = routing
    if routing and not ann.get("image_dependency"):
        ann["image_dependency"] = routing

    if not ann.get("retrieval_value"):
        if "毫无检索价值" in notes or "没有检索价值" in notes:
            ann["retrieval_value"] = "low"
        elif "检索价值弱" in notes:
            ann["retrieval_value"] = "low"
        elif any(key in notes for key in ["极高检索价值", "高价值检索", "检索质量", "极大锁定检索方向", "具有检索价值", "信息价值"]):
            ann["retrieval_value"] = "high"

    return ann


def main() -> None:
    base_rows = {row["sample_id"]: row for row in load_jsonl(JSONL_PATH)}
    md_rows = parse_md_annotations(MD_PATH.read_text(encoding="utf-8"))

    density_counter: Counter[str] = Counter()
    routing_counter: Counter[str] = Counter()
    retrieval_counter: Counter[str] = Counter()
    rows_written = 0

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for item in md_rows:
            sample_id = item["sample_id"]
            if sample_id not in base_rows:
                continue
            row = dict(base_rows[sample_id])
            ann = fill_from_notes(item["annotation_template"])
            row["annotation_template"] = ann
            if ann.get("density_level"):
                density_counter[ann["density_level"]] += 1
            if ann.get("routing_preference"):
                routing_counter[ann["routing_preference"]] += 1
            if ann.get("retrieval_value"):
                retrieval_counter[ann["retrieval_value"]] += 1
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows_written += 1

    summary = {
        "source_md": str(MD_PATH),
        "source_jsonl": str(JSONL_PATH),
        "output_jsonl": str(OUT_JSONL),
        "rows": rows_written,
        "density_distribution": dict(density_counter),
        "routing_distribution": dict(routing_counter),
        "retrieval_value_distribution": dict(retrieval_counter),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {OUT_JSONL}")
    print(f"[done] {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
