#!/usr/bin/env python3
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Tuple


DEFAULT_QUERY_PATH = Path(
    "/mnt/data_10/mwx/workspace/multi_modal_rag/evaluate_benchmark/translate_benchmark/output/endobench_test_translated.jsonl"
)
DEFAULT_RESPONSE_PATH = Path("/mnt/data_1/yds/多模态/responses_run1.jsonl")
DEFAULT_LEGACY_PATH = Path("/mnt/data_1/yds/多模态/data_house/origin_data/output_pairs_all_min_filtered_labeled.jsonl")
DEFAULT_OUTPUT_DIR = Path("/mnt/data_1/yds/多模态/权重模块/outputs/rebuild_audit")


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def p90(values: List[int]) -> int:
    if not values:
        return 0
    idx = min(len(values) - 1, int(len(values) * 0.9))
    return sorted(values)[idx]


def options_text(row: Dict) -> str:
    parts = []
    for key in ("A", "B", "C", "D", "E"):
        value = row.get(key)
        if value is not None and str(value).strip():
            parts.append(f"{key}: {str(value).strip()}")
    return "\n".join(parts)


def query_group(question: str) -> str:
    rules = [
        ("organ_or_part_identification", ["哪个消化器官", "哪一个消化器官", "胃肠道器官名称", "哪个器官", "哪个部分", "哪种消化器官"]),
        ("anatomy_or_landmark_identification", ["哪个解剖结构", "具体解剖结构", "识别出的解剖结构", "标示的解剖结构"]),
        ("lesion_presence_or_type", ["病理发现", "病理诊断", "异常性质", "病变", "异常"]),
        ("histology_or_pathology", ["组织病理学分类", "组织学类型", "病理类型"]),
        ("lesion_count", ["息肉数量", "多少个", "数量", "总数", "数目"]),
        ("surgical_workflow", ["手术过程", "手术步骤", "手术操作", "操作阶段", "哪个阶段", "哪种治疗操作"]),
        ("quality_or_score", ["波士顿肠道准备", "BBPS", "评分", "Mayo评分"]),
        ("visual_grounding", ["坐标", "边界框", "[x1, y1, x2, y2]"]),
    ]
    for tag, keys in rules:
        if any(k in question for k in keys):
            return tag
    return "other_template_query"


def build_query_summary(query_rows: List[Dict], response_rows: List[Dict]) -> Dict:
    questions = [str(row.get("question", "")).strip() for row in query_rows]
    lengths = [len(q) for q in questions if q]
    meta_by_index = {row.get("index"): row for row in response_rows}

    category_counter = Counter()
    task_counter = Counter()
    subtask_counter = Counter()
    scene_counter = Counter()
    dataset_counter = Counter()
    group_counter = Counter()

    for qrow in query_rows:
        idx = qrow.get("index")
        meta = meta_by_index.get(idx, {})
        category_counter[str(meta.get("category", "")).strip()] += 1
        task_counter[str(meta.get("task", "")).strip()] += 1
        subtask_counter[str(meta.get("subtask", "")).strip()] += 1
        scene_counter[str(meta.get("scene", "")).strip()] += 1
        dataset_counter[str(meta.get("dataset", "")).strip()] += 1
        group_counter[query_group(str(qrow.get("question", "")).strip())] += 1

    return {
        "total_queries": len(query_rows),
        "unique_questions": len(set(questions)),
        "length_stats": {
            "min": min(lengths) if lengths else 0,
            "median": median(lengths) if lengths else 0,
            "p90": p90(lengths),
            "max": max(lengths) if lengths else 0,
        },
        "top_questions": [{"question": q, "count": c} for q, c in Counter(questions).most_common(20)],
        "category_distribution": dict(category_counter.most_common()),
        "task_distribution": dict(task_counter.most_common()),
        "subtask_distribution": dict(subtask_counter.most_common()),
        "scene_distribution": dict(scene_counter.most_common()),
        "dataset_distribution": dict(dataset_counter.most_common()),
        "query_group_distribution": dict(group_counter.most_common()),
    }


def sample_real_queries(
    query_rows: List[Dict],
    response_rows: List[Dict],
    sample_size: int,
    seed: int,
) -> List[Dict]:
    rng = random.Random(seed)
    meta_by_index = {row.get("index"): row for row in response_rows}
    buckets: Dict[str, List[Dict]] = defaultdict(list)
    seen_text = set()
    for qrow in query_rows:
        question = str(qrow.get("question", "")).strip()
        if not question:
            continue
        meta = meta_by_index.get(qrow.get("index"), {})
        task = str(meta.get("task", "")).strip() or "UNKNOWN"
        row = {
            "index": qrow.get("index"),
            "category": meta.get("category", ""),
            "task": task,
            "subtask": meta.get("subtask", ""),
            "scene": meta.get("scene", ""),
            "dataset": meta.get("dataset", ""),
            "question": question,
            "query_group": query_group(question),
            "options_text": options_text(qrow),
            "options": {k: qrow.get(k) for k in ("A", "B", "C", "D", "E") if qrow.get(k) is not None},
        }
        buckets[task].append(row)

    ordered_tasks = [task for task, _ in Counter(row.get("task", "") for row in response_rows).most_common()]
    for task, rows in buckets.items():
        rng.shuffle(rows)

    picked: List[Dict] = []
    task_pos = {task: 0 for task in buckets}
    while len(picked) < sample_size:
        progressed = False
        for task in ordered_tasks:
            rows = buckets.get(task, [])
            pos = task_pos.get(task, 0)
            while pos < len(rows) and rows[pos]["question"] in seen_text:
                pos += 1
            task_pos[task] = pos
            if pos >= len(rows):
                continue
            row = rows[pos]
            task_pos[task] += 1
            seen_text.add(row["question"])
            row["annotation_template"] = {
                "is_realistic_user_query": "",
                "retrieval_value": "",
                "density_level": "",
                "image_dependency": "",
                "retrieval_anchor_terms": [],
                "generic_non_anchor_terms": [],
                "anchor_scope": "",
                "rewrite_needed": "",
                "intent_family": "",
                "notes": "",
            }
            picked.append(row)
            progressed = True
            if len(picked) >= sample_size:
                break
        if not progressed:
            break
    return picked


def legacy_bucket(text: str, primary: str) -> str:
    workflow_keys = ["第一步", "第二步", "第三步", "流程图", "步骤", "顺序", "阶段"]
    concept_keys = ["原理", "示意图", "图A", "图B", "图C", "图D", "图E", "敏感度", "特异度", "标准", "分类系统"]
    multi_keys = ["第一幅图像", "第二幅图像", "第三幅图像", "第四幅图像", "上排图像", "下排图像", "编号", "依次展示", "由四幅"]
    factual_primary = primary in {"病变特征", "诊断评估", "解剖特征"}
    length = len(text)
    workflow = any(k in text for k in workflow_keys)
    concept = any(k in text for k in concept_keys)
    multi = any(k in text for k in multi_keys)

    if factual_primary and 12 <= length <= 80 and not workflow and not concept and not multi:
        return "direct_reuse"
    if factual_primary and 20 <= length <= 220 and not concept and not workflow:
        return "rewrite_reuse"
    if primary in {"检查操作", "治疗操作", "基础概念"} or workflow or concept or multi or length > 220:
        return "weak_or_discard"
    return "borderline"


def build_legacy_summary(legacy_rows: List[Dict]) -> Dict:
    texts = [str(row.get("final_description", "")).strip() for row in legacy_rows if str(row.get("final_description", "")).strip()]
    lengths = [len(t) for t in texts]
    primary_counter = Counter(str(row.get("primary_knowledge_type", "")).strip() for row in legacy_rows)
    bucket_counter = Counter()
    bucket_primary = defaultdict(Counter)

    for row in legacy_rows:
        text = str(row.get("final_description", "")).strip()
        primary = str(row.get("primary_knowledge_type", "")).strip()
        bucket = legacy_bucket(text, primary)
        bucket_counter[bucket] += 1
        bucket_primary[bucket][primary] += 1

    return {
        "total_legacy_rows": len(legacy_rows),
        "length_stats": {
            "min": min(lengths) if lengths else 0,
            "median": median(lengths) if lengths else 0,
            "p90": p90(lengths),
            "max": max(lengths) if lengths else 0,
        },
        "primary_knowledge_distribution": dict(primary_counter.most_common()),
        "reuse_bucket_distribution": dict(bucket_counter.most_common()),
        "reuse_bucket_by_primary": {
            bucket: dict(counter.most_common()) for bucket, counter in bucket_primary.items()
        },
    }


def sample_legacy_rows(legacy_rows: List[Dict], sample_size: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    buckets: Dict[str, List[Dict]] = defaultdict(list)
    for row in legacy_rows:
        text = str(row.get("final_description", "")).strip()
        if not text:
            continue
        primary = str(row.get("primary_knowledge_type", "")).strip()
        bucket = legacy_bucket(text, primary)
        buckets[bucket].append(
            {
                "image_path": row.get("image_path", ""),
                "primary_knowledge_type": primary,
                "secondary_knowledge_types": row.get("secondary_knowledge_types", []),
                "legacy_bucket": bucket,
                "final_description": text,
                "annotation_template": {
                    "reuse_decision": "",
                    "best_use": "",
                    "rewrite_query_example": "",
                    "candidate_retrieval_anchor_terms": [],
                    "generic_non_anchor_terms": [],
                    "anchor_scope": "",
                    "needs_manual_review": "",
                    "notes": "",
                },
            }
        )
    for rows in buckets.values():
        rng.shuffle(rows)

    order = ["direct_reuse", "rewrite_reuse", "weak_or_discard", "borderline"]
    picked: List[Dict] = []
    positions = {bucket: 0 for bucket in buckets}
    while len(picked) < sample_size:
        progressed = False
        for bucket in order:
            rows = buckets.get(bucket, [])
            pos = positions.get(bucket, 0)
            if pos >= len(rows):
                continue
            picked.append(rows[pos])
            positions[bucket] += 1
            progressed = True
            if len(picked) >= sample_size:
                break
        if not progressed:
            break
    return picked


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_real_query_md(path: Path, rows: List[Dict], summary: Dict) -> None:
    lines: List[str] = []
    lines.append("# Real Query Manual Pack")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- total_queries: {summary['total_queries']}")
    lines.append(f"- unique_questions: {summary['unique_questions']}")
    lines.append(
        f"- length_stats: min={summary['length_stats']['min']} median={summary['length_stats']['median']} p90={summary['length_stats']['p90']} max={summary['length_stats']['max']}"
    )
    lines.append("")
    for idx, row in enumerate(rows, start=1):
        lines.append(f"## Sample {idx}")
        lines.append(f"- index: {row['index']}")
        lines.append(f"- category: {row['category']}")
        lines.append(f"- task: {row['task']}")
        lines.append(f"- subtask: {row['subtask']}")
        lines.append(f"- scene: {row['scene']}")
        lines.append(f"- dataset: {row['dataset']}")
        lines.append(f"- query_group: {row['query_group']}")
        lines.append(f"- question: {row['question']}")
        if row["options_text"]:
            lines.append("- options:")
            lines.append("```text")
            lines.append(row["options_text"])
            lines.append("```")
        lines.append("- annotation_template:")
        lines.append("```json")
        lines.append(json.dumps(row["annotation_template"], ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_legacy_md(path: Path, rows: List[Dict], summary: Dict) -> None:
    lines: List[str] = []
    lines.append("# Legacy Reuse Manual Pack")
    lines.append("")
    lines.append("## Summary")
    for bucket, count in summary["reuse_bucket_distribution"].items():
        lines.append(f"- {bucket}: {count}")
    lines.append("")
    for idx, row in enumerate(rows, start=1):
        lines.append(f"## Sample {idx}")
        lines.append(f"- primary_knowledge_type: {row['primary_knowledge_type']}")
        lines.append(f"- legacy_bucket: {row['legacy_bucket']}")
        lines.append(f"- image_path: {row['image_path']}")
        lines.append("- final_description:")
        lines.append("```text")
        lines.append(row["final_description"])
        lines.append("```")
        lines.append("- annotation_template:")
        lines.append("```json")
        lines.append(json.dumps(row["annotation_template"], ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build audit packs for high-quality weight-module data reconstruction.")
    parser.add_argument("--query-jsonl", type=Path, default=DEFAULT_QUERY_PATH)
    parser.add_argument("--responses-jsonl", type=Path, default=DEFAULT_RESPONSE_PATH)
    parser.add_argument("--legacy-jsonl", type=Path, default=DEFAULT_LEGACY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--real-query-samples", type=int, default=60)
    parser.add_argument("--legacy-samples", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260427)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    query_rows = load_jsonl(args.query_jsonl)
    response_rows = load_jsonl(args.responses_jsonl)
    legacy_rows = load_jsonl(args.legacy_jsonl)

    real_summary = build_query_summary(query_rows, response_rows)
    legacy_summary = build_legacy_summary(legacy_rows)

    real_pack = sample_real_queries(query_rows, response_rows, args.real_query_samples, args.seed)
    legacy_pack = sample_legacy_rows(legacy_rows, args.legacy_samples, args.seed)

    real_summary_path = args.output_dir / "real_query_summary.v1.json"
    legacy_summary_path = args.output_dir / "legacy_reuse_summary.v1.json"
    real_jsonl_path = args.output_dir / "real_query_manual_pack.v1.jsonl"
    real_md_path = args.output_dir / "real_query_manual_pack.v1.md"
    legacy_jsonl_path = args.output_dir / "legacy_reuse_manual_pack.v1.jsonl"
    legacy_md_path = args.output_dir / "legacy_reuse_manual_pack.v1.md"

    real_summary_path.write_text(json.dumps(real_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    legacy_summary_path.write_text(json.dumps(legacy_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(real_jsonl_path, real_pack)
    write_jsonl(legacy_jsonl_path, legacy_pack)
    write_real_query_md(real_md_path, real_pack, real_summary)
    write_legacy_md(legacy_md_path, legacy_pack, legacy_summary)

    print(f"[done] {real_summary_path}")
    print(f"[done] {legacy_summary_path}")
    print(f"[done] {real_jsonl_path}")
    print(f"[done] {real_md_path}")
    print(f"[done] {legacy_jsonl_path}")
    print(f"[done] {legacy_md_path}")


if __name__ == "__main__":
    main()
