#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path("/mnt/data_1/yds/多模态/权重模块")
OUT = ROOT / "outputs"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def chain_to_map(chain: List[Dict[str, Any]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in chain:
        level = str(item.get("level", "")).strip()
        text = str(item.get("query_text", "")).strip()
        if level:
            out[level] = text
    return out


def issue_brief_list(review_row: Dict[str, Any]) -> List[Dict[str, str]]:
    issues = review_row.get("issues", [])
    out: List[Dict[str, str]] = []
    if not isinstance(issues, list):
        return out
    for item in issues:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "code": str(item.get("code", "")),
                "location": str(item.get("location", "")),
                "message": str(item.get("message", "")),
            }
        )
    return out


def review_summary(review_row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "pass": bool(review_row.get("pass")),
        "summary": review_row.get("summary", {}),
        "quality_scores": review_row.get("quality_scores", {}),
        "issues": issue_brief_list(review_row),
    }


def md_chain(label: str, chain_map: Dict[str, str]) -> List[str]:
    lines = [f"### {label}"]
    for level in ["L0", "L1", "L2", "L3", "L4"]:
        lines.append(f"- `{level}`: {chain_map.get(level, '')}")
    return lines


def md_review(label: str, summary: Dict[str, Any]) -> List[str]:
    lines = [f"### {label}"]
    lines.append(f"- `pass`: {summary.get('pass')}")
    q = summary.get("quality_scores", {})
    if isinstance(q, dict) and q:
        lines.append(
            "- `quality_scores`: "
            + ", ".join(f"{k}={v}" for k, v in q.items())
        )
    issues = summary.get("issues", [])
    if issues:
        lines.append("- `issues`:")
        for item in issues:
            lines.append(
                f"  - `{item.get('code')}` {item.get('location')}: {item.get('message')}"
            )
    else:
        lines.append("- `issues`: none")
    return lines


def main() -> None:
    v5_path = OUT / "generated_chain.pilot_20.v5.jsonl"
    v6_path = OUT / "generated_chain.pilot_20.v6.jsonl"
    r5_path = OUT / "rejected_chain.pilot_20.v5b.jsonl"
    r6_path = OUT / "rejected_chain.pilot_20.v6.jsonl"

    out_jsonl = OUT / "manual_audit_pack.v1.jsonl"
    out_md = OUT / "manual_audit_pack.v1.md"

    v5_rows = {row["candidate_id"]: row for row in load_jsonl(v5_path)}
    v6_rows = {row["candidate_id"]: row for row in load_jsonl(v6_path)}
    r5_rows = {row["candidate_id"]: row for row in load_jsonl(r5_path)}
    r6_rows = {row["candidate_id"]: row for row in load_jsonl(r6_path)}

    candidate_ids = sorted(set(v5_rows) & set(v6_rows) & set(r5_rows) & set(r6_rows))
    audit_rows: List[Dict[str, Any]] = []

    for idx, cid in enumerate(candidate_ids, start=1):
        v5 = v5_rows[cid]
        v6 = v6_rows[cid]
        r5 = r5_rows[cid]
        r6 = r6_rows[cid]

        row = {
            "audit_id": f"audit_{idx:03d}",
            "candidate_id": cid,
            "anchor_class": v6.get("anchor_class"),
            "split": v6.get("split"),
            "semantic_core": v6.get("semantic_core"),
            "semantic_core_hash": v6.get("semantic_core_hash"),
            "anchor_text": v6.get("anchor_text"),
            "v5_main_chain": chain_to_map(v5.get("main_chain", [])),
            "v6_main_chain": chain_to_map(v6.get("main_chain", [])),
            "v5_review": review_summary(r5),
            "v6_review": review_summary(r6),
            "reviewer_decision": {
                "preferred_version": "",
                "anchor_usable": "",
                "reject_reason": "",
                "notes": "",
            },
        }
        audit_rows.append(row)

    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in audit_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    md_lines: List[str] = [
        "# Manual Audit Pack V1",
        "",
        "用途：人工校准 `query` 风格、`L0-L4` 边界和审查器标准。",
        "",
        "审核建议：",
        "- 先看 `anchor_text` 是否真的适合生成 query。",
        "- 再比较 `v5` 和 `v6` 哪版更像真实用户输入。",
        "- 如果两版都不理想，直接在备注里写你认可的 query 风格。",
        "- `preferred_version` 建议填：`v5` / `v6` / `neither`。",
        "- `anchor_usable` 建议填：`yes` / `no`。",
        "",
    ]

    for row in audit_rows:
        md_lines.extend(
            [
                f"## {row['audit_id']} `{row['candidate_id']}`",
                f"- `anchor_class`: {row['anchor_class']}",
                f"- `semantic_core`: {row['semantic_core']}",
                f"- `anchor_text`: {row['anchor_text']}",
                "",
            ]
        )
        md_lines.extend(md_chain("V5 Main Chain", row["v5_main_chain"]))
        md_lines.append("")
        md_lines.extend(md_review("V5 Review", row["v5_review"]))
        md_lines.append("")
        md_lines.extend(md_chain("V6 Main Chain", row["v6_main_chain"]))
        md_lines.append("")
        md_lines.extend(md_review("V6 Review", row["v6_review"]))
        md_lines.append("")
        md_lines.extend(
            [
                "### Reviewer Decision",
                "- `preferred_version`: ",
                "- `anchor_usable`: ",
                "- `reject_reason`: ",
                "- `notes`: ",
                "",
                "---",
                "",
            ]
        )

    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[done] {out_jsonl}")
    print(f"[done] {out_md}")
    print(f"[count] {len(audit_rows)}")


if __name__ == "__main__":
    main()
