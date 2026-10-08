#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List

import yaml
from openai import AsyncOpenAI


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def extract_job_id(row: Dict[str, Any]) -> str:
    return str(row.get("job_id") or row.get("job", {}).get("job_id") or "").strip()


def load_completed_job_ids(*paths: Path) -> set[str]:
    done: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for row in load_jsonl(path):
            job_id = extract_job_id(row)
            if job_id:
                done.add(job_id)
    return done


def extract_json(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("{") and raw.endswith("}"):
        return json.loads(raw)
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        raise ValueError("No JSON object found in response")
    return json.loads(match.group(0))


def build_rewrite_prompt(prompt_template: str, job: Dict[str, Any]) -> str:
    payload = {
        "job_id": job["job_id"],
        "source_text": job["legacy_anchor"]["text"],
        "target_density_level": job["target_density_level"],
        "target_routing_preference": job["target_routing_preference"],
        "human_aligned_cases": job.get("human_aligned_cases", []),
    }
    return (
        prompt_template.strip()
        + "\n\n输入 JSON：\n```json\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n```"
    )


def build_label_prompt(
    prompt_template: str,
    job_id: str,
    query_text: str,
    human_aligned_cases: List[Dict[str, Any]],
) -> str:
    payload = {
        "job_id": job_id,
        "query_text": query_text,
        "human_aligned_cases": human_aligned_cases,
    }
    return (
        prompt_template.strip()
        + "\n\n输入 JSON：\n```json\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n```"
    )


async def call_model(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    timeout_seconds: int,
    temperature: float,
    use_json_response_format: bool,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "timeout": timeout_seconds,
        "temperature": temperature,
    }
    if use_json_response_format:
        kwargs["response_format"] = {"type": "json_object"}
    response = await client.chat.completions.create(**kwargs)
    content = (response.choices[0].message.content or "").strip()
    return extract_json(content)


def pick_legacy(rows: List[Dict[str, Any]], skip_buckets: List[str]) -> List[Dict[str, Any]]:
    picked: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("legacy_bucket") in set(skip_buckets):
            continue
        ann = row.get("annotation_template", {}) or {}
        decision = str(ann.get("reuse_decision", "")).strip().lower()
        if decision in {"direct_reuse", "rewrite_reuse", "keep"}:
            picked.append(row)
    return picked


def build_target_sequences(max_jobs: int) -> Dict[str, List[str]]:
    rng = random.Random(20260427)
    compatible_pairs = [
        ("L0", "R3"),
        ("L1", "R3"),
        ("L1", "R2"),
        ("L2", "R1"),
        ("L2", "R2"),
        ("L3", "R1"),
        ("L3", "R2"),
        ("L3", "R3"),
        ("L4", "R1"),
        ("L4", "R2"),
    ]
    weights = [10, 12, 8, 9, 10, 10, 12, 4, 9, 12]
    pairs: List[tuple[str, str]] = []
    for _ in range(max_jobs):
        pairs.append(rng.choices(compatible_pairs, weights=weights, k=1)[0])
    return {
        "density": [p[0] for p in pairs],
        "routing": [p[1] for p in pairs],
    }


def build_jobs(legacy_rows: List[Dict[str, Any]], max_jobs: int) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    total = min(max_jobs, len(legacy_rows))
    rng = random.Random(20260427)
    rows = legacy_rows[:]
    rng.shuffle(rows)
    target_seq = build_target_sequences(total)
    for idx, legacy in enumerate(rows[:total], start=1):
        jobs.append(
            {
                "job_id": f"job_{idx:06d}",
                "legacy_anchor": {
                    "text": legacy.get("final_description", ""),
                    "primary_knowledge_type": legacy.get("primary_knowledge_type", ""),
                    "secondary_knowledge_types": legacy.get("secondary_knowledge_types", []),
                    "legacy_bucket": legacy.get("legacy_bucket", ""),
                },
                "target_density_level": target_seq["density"][idx - 1],
                "target_routing_preference": target_seq["routing"][idx - 1],
            }
        )
    return jobs


def load_alignment_cases(path: Path, max_cases: int) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = load_jsonl(path)
    picked: List[Dict[str, Any]] = []
    for row in rows:
        ann = row.get("annotation_template", {}) or {}
        routing = ann.get("routing_preference") or ann.get("image_dependency")
        if not ann.get("density_level") and not routing:
            continue
        picked.append(
            {
                "sample_id": row.get("sample_id"),
                "text": row.get("text"),
                "density_level": ann.get("density_level"),
                "routing_preference": routing,
                "retrieval_value": ann.get("retrieval_value"),
                "retrieval_anchor_terms": ann.get("retrieval_anchor_terms", []),
                "generic_non_anchor_terms": ann.get("generic_non_anchor_terms", []),
                "notes": ann.get("notes", ""),
            }
        )
    return picked


def select_alignment_cases(
    all_cases: List[Dict[str, Any]],
    target_density_level: str,
    target_routing_preference: str,
    max_cases: int,
) -> List[Dict[str, Any]]:
    if not all_cases:
        return []
    exact = [
        x for x in all_cases
        if x.get("density_level") == target_density_level
        and x.get("routing_preference") == target_routing_preference
    ]
    if len(exact) >= max_cases:
        return exact[:max_cases]

    same_routing = [
        x for x in all_cases
        if x.get("routing_preference") == target_routing_preference
        and x not in exact
    ]
    same_density = [
        x for x in all_cases
        if x.get("density_level") == target_density_level
        and x not in exact
        and x not in same_routing
    ]
    others = [
        x for x in all_cases
        if x not in exact and x not in same_routing and x not in same_density
    ]
    picked = exact + same_routing + same_density + others
    return picked[:max_cases]


async def run_generation(cfg: Dict[str, Any], config_path: Path) -> None:
    paths = cfg["paths"]
    request_cfg = cfg["request"]
    api_cfg = cfg["api"]
    gen_cfg = cfg["generation"]

    output_dir = Path(paths["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = output_dir / "generated_candidates.v2.jsonl"
    error_jsonl = output_dir / "generated_candidates.v2.errors.jsonl"

    legacy_pack = load_jsonl(Path(paths["legacy_manual_pack_jsonl"]))
    legacy_rows = pick_legacy(legacy_pack, list(gen_cfg.get("skip_legacy_buckets", [])))

    if not legacy_rows:
        raise RuntimeError("No reusable legacy anchors found. Please fill legacy_reuse_manual_pack first.")

    jobs = build_jobs(legacy_rows, int(gen_cfg["max_jobs"]))
    completed_job_ids = load_completed_job_ids(output_jsonl)
    if completed_job_ids:
        jobs = [job for job in jobs if job["job_id"] not in completed_job_ids]
        print(
            f"[resume] skip completed jobs: {len(completed_job_ids)}; remaining jobs: {len(jobs)}"
        )
    else:
        print(f"[start] jobs to run: {len(jobs)}")
    all_alignment_cases = load_alignment_cases(
        Path(paths.get("alignment_case_pack_jsonl", "")),
        9999,
    )
    if all_alignment_cases:
        for job in jobs:
            job["human_aligned_cases"] = select_alignment_cases(
                all_cases=all_alignment_cases,
                target_density_level=job["target_density_level"],
                target_routing_preference=job["target_routing_preference"],
                max_cases=int(gen_cfg.get("max_alignment_cases_in_prompt", 8)),
            )
    project_root = config_path.parent.parent
    label_guideline_path = Path(paths.get("label_guideline_path") or (project_root / "text_density_design" / "docs" / "LABEL_GUIDELINE.md"))
    rewrite_prompt_template = read_text(config_path.parent / "prompts" / "rewrite_query_v2.md")
    label_prompt_template = read_text(config_path.parent / "prompts" / "label_query_v2.md")

    api_key = os.getenv(api_cfg["key_env"])
    if not api_key:
        raise RuntimeError(f"Missing env: {api_cfg['key_env']}")
    client = AsyncOpenAI(api_key=api_key, base_url=api_cfg.get("base_url"))

    semaphore = asyncio.Semaphore(int(request_cfg["concurrency"]))

    async def worker(job: Dict[str, Any]) -> None:
        rewrite_prompt = build_rewrite_prompt(rewrite_prompt_template, job)
        async with semaphore:
            try:
                rewrite_obj = await call_model(
                    client=client,
                    model=api_cfg["generation_model"],
                    prompt=rewrite_prompt,
                    timeout_seconds=int(request_cfg["timeout_seconds"]),
                    temperature=float(request_cfg["temperature_generation"]),
                    use_json_response_format=bool(request_cfg["use_json_response_format"]),
                )
                rewrite_items = rewrite_obj.get("items") or []
                if not rewrite_items:
                    raise ValueError("Rewrite step returned empty items")
                rewrite_item = rewrite_items[0]
                query_text = str(rewrite_item.get("query_text", "")).strip()
                if not query_text:
                    raise ValueError("Rewrite step returned empty query_text")

                label_prompt = build_label_prompt(
                    prompt_template=label_prompt_template,
                    job_id=job["job_id"],
                    query_text=query_text,
                    human_aligned_cases=job.get("human_aligned_cases", []),
                )
                label_obj = await call_model(
                    client=client,
                    model=api_cfg["generation_model"],
                    prompt=label_prompt,
                    timeout_seconds=int(request_cfg["timeout_seconds"]),
                    temperature=float(request_cfg["temperature_generation"]),
                    use_json_response_format=bool(request_cfg["use_json_response_format"]),
                )
                label_items = label_obj.get("items") or []
                if not label_items:
                    raise ValueError("Label step returned empty items")
                label_item = label_items[0]

                merged_item = {
                    "job_id": job["job_id"],
                    "query_text": query_text,
                    "density_level": label_item.get("density_level"),
                    "routing_preference": label_item.get("routing_preference"),
                    "retrieval_value": label_item.get("retrieval_value"),
                    "rewrite_confidence": rewrite_item.get("confidence"),
                    "label_confidence": label_item.get("confidence"),
                }
                with output_jsonl.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "job": job,
                                "response": {"items": [merged_item]},
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            except Exception as exc:
                with error_jsonl.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"job": job, "error": str(exc)}, ensure_ascii=False) + "\n")

    await asyncio.gather(*(worker(job) for job in jobs))
    print(f"[done] {output_jsonl}")
    print(f"[done] {error_jsonl}")


def main() -> None:
    parser = argparse.ArgumentParser(description="API-based generator for high-quality weight-module data v2.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/mnt/data_1/yds/多模态/权重模块/high_quality_data_pipeline_v2.yaml"),
    )
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    asyncio.run(run_generation(cfg, args.config))


if __name__ == "__main__":
    main()
