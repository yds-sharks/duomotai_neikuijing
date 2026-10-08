#!/usr/bin/env python3
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path("/mnt/data_1/yds/多模态/权重模块")
OUT = ROOT / "outputs"
SOURCE = OUT / "generated_chain.pilot_1000.v8.jsonl"
ERRORS = OUT / "generated_chain.pilot_1000.v8.errors.jsonl"
AUDIT_JSONL = OUT / "manual_audit_pack.v8.sample50.jsonl"
AUDIT_MD = OUT / "manual_audit_pack.v8.sample50.md"
REPORT = OUT / "manual_audit_pack.v8.sample50.report.json"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def chain_map(row: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in row.get("main_chain", []):
        out[str(item.get("level", ""))] = str(item.get("query_text", "")).strip()
    return out


def question_type(text: str) -> str:
    patterns = [
        ("normal", r"正常吗|有没有问题|有问题吗"),
        ("common", r"常见吗"),
        ("what", r"是什么情况|怎么回事"),
        ("indicate", r"提示什么|说明什么|意味着什么|代表什么"),
        ("consider", r"考虑什么|指向什么"),
        ("disease", r"什么病变|哪类病变"),
        ("location", r"附近|部位|有关吗"),
    ]
    for name, pattern in patterns:
        if re.search(pattern, text):
            return name
    return "other"


def summarize(rows: List[Dict[str, Any]], errors: List[Dict[str, Any]], sample: List[Dict[str, Any]]) -> Dict[str, Any]:
    class_counter = Counter(r.get("anchor_class", "UNK") for r in rows)
    sample_class_counter = Counter(r.get("anchor_class", "UNK") for r in sample)
    level_qtypes = defaultdict(Counter)
    malformed = Counter()

    for row in rows:
        for level, text in chain_map(row).items():
            level_qtypes[level][question_type(text)] += 1
            if not text.endswith("？"):
                malformed[f"{level}_not_question"] += 1
            if " " in text and len(text.split()) >= 3:
                malformed[f"{level}_keyword_like"] += 1

    return {
        "source": str(SOURCE),
        "total_success": len(rows),
        "total_errors": len(errors),
        "class_distribution": dict(class_counter),
        "sample_size": len(sample),
        "sample_class_distribution": dict(sample_class_counter),
        "question_type_distribution_by_level": {
            level: dict(counter) for level, counter in sorted(level_qtypes.items())
        },
        "malformed_checks": dict(malformed),
    }


def stratified_sample(rows: List[Dict[str, Any]], size: int = 50, seed: int = 20260423) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    by_class: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_class[row.get("anchor_class", "UNK")].append(row)

    quotas = {"A": 20, "B": 20, "C": 10}
    chosen: List[Dict[str, Any]] = []
    chosen_ids = set()

    for cls, quota in quotas.items():
        bucket = list(by_class.get(cls, []))
        rng.shuffle(bucket)
        for row in bucket[: min(quota, len(bucket))]:
            chosen.append(row)
            chosen_ids.add(row.get("candidate_id"))

    if len(chosen) < size:
        remaining = [r for r in rows if r.get("candidate_id") not in chosen_ids]
        rng.shuffle(remaining)
        chosen.extend(remaining[: size - len(chosen)])

    rng.shuffle(chosen)
    return chosen[:size]


def write_outputs(sample: List[Dict[str, Any]], report: Dict[str, Any]) -> None:
    with AUDIT_JSONL.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(sample, start=1):
            out = {
                "audit_id": f"v8_audit_{idx:03d}",
                "candidate_id": row.get("candidate_id"),
                "anchor_class": row.get("anchor_class"),
                "semantic_core": row.get("semantic_core"),
                "anchor_text": row.get("anchor_text"),
                "main_chain": chain_map(row),
                "reviewer_decision": {
                    "usable": "",
                    "level_boundary_ok": "",
                    "question_style_ok": "",
                    "diversity_ok": "",
                    "notes": "",
                },
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    md: List[str] = [
        "# Manual Audit Pack V8 Sample50",
        "",
        "用途：抽检 `pilot_1000 v8` 生成质量，重点看完整问题句、等级边界和设问多样性。",
        "",
        "建议填写：",
        "- `usable`: yes/no",
        "- `level_boundary_ok`: yes/no",
        "- `question_style_ok`: yes/no",
        "- `diversity_ok`: yes/no",
        "",
        "## Summary",
        f"- `total_success`: {report['total_success']}",
        f"- `total_errors`: {report['total_errors']}",
        f"- `class_distribution`: {report['class_distribution']}",
        f"- `sample_class_distribution`: {report['sample_class_distribution']}",
        f"- `malformed_checks`: {report['malformed_checks']}",
        "",
        "## Question Type Distribution",
        "```json",
        json.dumps(report["question_type_distribution_by_level"], ensure_ascii=False, indent=2),
        "```",
        "",
    ]

    for idx, row in enumerate(sample, start=1):
        chain = chain_map(row)
        md.extend(
            [
                f"## v8_audit_{idx:03d} `{row.get('candidate_id')}`",
                f"- `anchor_class`: {row.get('anchor_class')}",
                f"- `semantic_core`: {row.get('semantic_core')}",
                f"- `anchor_text`: {row.get('anchor_text')}",
                "",
                "### Main Chain",
            ]
        )
        for level in ["L0", "L1", "L2", "L3", "L4"]:
            md.append(f"- `{level}`: {chain.get(level, '')}")
        md.extend(
            [
                "",
                "### Reviewer Decision",
                "- `usable`: ",
                "- `level_boundary_ok`: ",
                "- `question_style_ok`: ",
                "- `diversity_ok`: ",
                "- `notes`: ",
                "",
                "---",
                "",
            ]
        )

    AUDIT_MD.write_text("\n".join(md), encoding="utf-8")
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    rows = load_jsonl(SOURCE)
    errors = load_jsonl(ERRORS)
    sample = stratified_sample(rows)
    report = summarize(rows, errors, sample)
    write_outputs(sample, report)
    print(f"[done] {AUDIT_JSONL}")
    print(f"[done] {AUDIT_MD}")
    print(f"[done] {REPORT}")


if __name__ == "__main__":
    main()
