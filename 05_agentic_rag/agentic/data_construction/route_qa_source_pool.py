#!/usr/bin/env python3
"""Route prepared source records to candidate QA query types."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from qa_stage1_common import candidate_id, clean_text, iter_jsonl, non_generic_organs

DEFAULT_INPUT = "/mnt/data_1/yds/多模态/agentic/outputs/qa_stage1_prepared/source_pool.jsonl"
DEFAULT_OUTPUT_DIR = "/mnt/data_1/yds/多模态/agentic/outputs/qa_stage1_prepared"

QUERY_TYPES = [
    "anatomical_site_recognition",
    "lesion_or_finding_identification",
    "procedure_or_operation_recognition",
    "spatial_region_understanding",
]

DEFAULT_LOW_INFORMATION_QUERY = {
    "anatomical_site_recognition": "这张图是什么部位？",
    "lesion_or_finding_identification": "图中是什么异常？",
    "procedure_or_operation_recognition": "图中在做什么操作？",
    "spatial_region_understanding": "图中异常在哪里？",
}

LESION_KEYWORDS = [
    "息肉", "腺瘤", "癌", "肿瘤", "肿物", "隆起", "凹陷", "溃疡", "糜烂", "狭窄", "出血", "炎症",
    "白斑", "红斑", "不染", "染色", "异型", "病变", "瘘", "穿孔", "气肿", "静脉曲张", "萎缩", "肠化",
    "黏膜改变", "异常", "结节", "囊肿", "坏死", "浸润", "分型", "分级",
]

PROCEDURE_KEYWORDS = [
    "切除", "剥离", "切开", "夹闭", "缝合", "注射", "止血", "活检", "染色", "标记", "牵引", "套扎",
    "扩张", "置入", "支架", "取石", "碎石", "ERCP", "ESD", "EMR", "POEM", "EUS", "治疗", "操作",
    "电刀", "圈套器", "止血夹", "器械", "导丝", "针刀", "透明帽",
]

SPATIAL_KEYWORDS = [
    "近端", "远端", "口侧", "肛侧", "前壁", "后壁", "左侧", "右侧", "小弯", "大弯", "上部", "中部", "下部",
    "边缘", "中心", "开口", "入口", "出口", "贲门", "幽门", "球部", "降部", "乳头", "齿状线", "回盲瓣",
    "位置", "区域", "侧壁", "顶端", "底部", "周围", "箭头", "所指",
]

ANATOMY_HINTS = [
    "食管", "胃", "胃窦", "胃体", "胃角", "贲门", "幽门", "十二指肠", "小肠", "回肠", "空肠",
    "结肠", "直肠", "盲肠", "回盲", "胆管", "胰腺", "乳头", "咽", "肛管",
]

ANATOMICAL_SUBSITE_HINTS = [
    "胃窦", "胃体", "胃角", "贲门", "幽门", "十二指肠球部", "降部", "乳头", "回盲部", "盲肠", "直肠", "齿状线"
]

ANATOMICAL_STRUCTURE_HINTS = ["皱襞", "乳头", "瓣", "开口", "管腔", "黏膜", "血管", "憩室"]


def has_any(text: str, keywords: List[str]) -> List[str]:
    return [kw for kw in keywords if kw in text]


def anatomical_subtype_hint(text: str) -> str:
    if has_any(text, ANATOMICAL_STRUCTURE_HINTS):
        return "structure"
    if has_any(text, ANATOMICAL_SUBSITE_HINTS):
        return "subsite"
    return "coarse_site"


def route_record(row: Dict[str, Any]) -> Tuple[List[str], Dict[str, List[str]], Dict[str, str]]:
    evidence_text = "\n".join([
        row.get("caption_or_pair_text", ""),
        row.get("image_local_context_text", ""),
        row.get("pdf_context_text", ""),
        json.dumps(row.get("labels") or {}, ensure_ascii=False),
    ])
    routing_text = "\n".join([
        evidence_text,
        row.get("primary_knowledge_type", ""),
        " ".join(row.get("secondary_knowledge_types") or []),
        " ".join(row.get("organ_tags") or []),
    ])
    evidence_text = clean_text(evidence_text, max_len=12000)
    routing_text = clean_text(routing_text, max_len=12000)
    primary = row.get("primary_knowledge_type", "")
    secondary = row.get("secondary_knowledge_types") or []
    organs = non_generic_organs(row.get("organ_tags") or [])

    reasons: Dict[str, List[str]] = defaultdict(list)
    subtype_hints: Dict[str, str] = {}

    if organs:
        reasons["anatomical_site_recognition"].append("non_generic_organ_tags: " + ",".join(organs[:5]))
    anatomy_hits = has_any(routing_text, ANATOMY_HINTS)
    if primary == "解剖特征" or len(anatomy_hits) >= 2:
        reasons["anatomical_site_recognition"].append("anatomy_context_hits: " + ",".join(anatomy_hits[:5]))
    if reasons.get("anatomical_site_recognition"):
        subtype_hints["anatomical_site_recognition"] = anatomical_subtype_hint(routing_text)

    lesion_hits = has_any(evidence_text, LESION_KEYWORDS)
    if primary in {"病变特征", "诊断评估"} or lesion_hits:
        reasons["lesion_or_finding_identification"].append("lesion_context_hits: " + ",".join(lesion_hits[:8]))
        subtype_hints["lesion_or_finding_identification"] = "prefer_same_site_hard_negatives"

    procedure_hits = has_any(evidence_text, PROCEDURE_KEYWORDS)
    if procedure_hits:
        reasons["procedure_or_operation_recognition"].append("procedure_context_hits: " + ",".join(procedure_hits[:8]))
        subtype_hints["procedure_or_operation_recognition"] = "requires_visible_instrument_or_explicit_step"
    elif primary in {"检查操作", "治疗操作"} or any(x in {"检查操作", "治疗操作"} for x in secondary):
        reasons["procedure_or_operation_recognition"].append("weak_operation_label_only_requires_api_recheck")
        subtype_hints["procedure_or_operation_recognition"] = "weak_route_requires_strict_verification"

    spatial_hits = has_any(evidence_text, SPATIAL_KEYWORDS)
    if spatial_hits:
        reasons["spatial_region_understanding"].append("spatial_context_hits: " + ",".join(spatial_hits[:8]))
        subtype_hints["spatial_region_understanding"] = "do_not_force_if_no_explicit_position"

    routed = [qt for qt in QUERY_TYPES if reasons.get(qt)]
    return routed, dict(reasons), subtype_hints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pilot-per-type", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    no_route = 0
    for row in iter_jsonl(Path(args.input)):
        routed, reasons, subtype_hints = route_record(row)
        if not routed:
            no_route += 1
            continue
        row = dict(row)
        row["candidate_query_types"] = routed
        row["routing_reasons"] = reasons
        row["routing_subtype_hints"] = subtype_hints
        source_rows.append(row)
        for qt in routed:
            cand = dict(row)
            cand["candidate_id"] = candidate_id(row["source_id"], qt)
            cand["candidate_query_type"] = qt
            cand["candidate_routing_reasons"] = reasons.get(qt, [])
            cand["candidate_subtype_hint"] = subtype_hints.get(qt, "")
            cand["low_information_query_seed"] = DEFAULT_LOW_INFORMATION_QUERY.get(qt, "这张图是什么？")
            cand["low_information_query_policy"] = "rule_first_default; API may suggest alternative but should not make it more specific by default"
            candidate_rows.append(cand)

    routed_path = out_dir / "routed_source_pool.jsonl"
    with routed_path.open("w", encoding="utf-8") as handle:
        for row in source_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    candidate_path = out_dir / "routed_qa_candidates.jsonl"
    with candidate_path.open("w", encoding="utf-8") as handle:
        for row in candidate_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    pilot_rows: List[Dict[str, Any]] = []
    per_type_counts: Counter = Counter()
    used_sources: set[str] = set()
    for row in candidate_rows:
        qt = row["candidate_query_type"]
        if per_type_counts[qt] >= args.pilot_per_type:
            continue
        key = f"{row['source_id']}:{qt}"
        if key in used_sources:
            continue
        pilot_rows.append(row)
        used_sources.add(key)
        per_type_counts[qt] += 1
        if all(per_type_counts[t] >= args.pilot_per_type for t in QUERY_TYPES):
            break

    pilot_path = out_dir / "pilot_qa_candidates.jsonl"
    with pilot_path.open("w", encoding="utf-8") as handle:
        for row in pilot_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "input": args.input,
        "routed_source_pool": str(routed_path),
        "routed_qa_candidates": str(candidate_path),
        "pilot_qa_candidates": str(pilot_path),
        "source_count": len(source_rows),
        "candidate_count": len(candidate_rows),
        "no_route_count": no_route,
        "candidate_query_type_counts": dict(Counter(row["candidate_query_type"] for row in candidate_rows)),
        "pilot_query_type_counts": dict(per_type_counts),
        "split_counts": dict(Counter(row["split"] for row in source_rows)),
    }
    report_path = out_dir / "routing_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
