#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
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
            if line:
                rows.append(json.loads(line))
    return rows


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def extract_job_id(row: Dict[str, Any]) -> str:
    if "source" in row and isinstance(row["source"], dict):
        source = row["source"]
        return str(source.get("job", {}).get("job_id") or source.get("job_id") or "").strip()
    return str(row.get("job", {}).get("job_id") or row.get("job_id") or "").strip()


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


def build_prompt(prompt_template: str, row: Dict[str, Any], human_aligned_cases: List[Dict[str, Any]]) -> str:
    generated = (row.get("response") or {}).get("items") or [{}]
    item = generated[0] if generated else {}
    payload = {
        "job_id": row.get("job", {}).get("job_id") or row.get("job_id"),
        "query_text": item.get("query_text", ""),
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


async def main_async(config_path: Path, input_jsonl: Path, output_jsonl: Path, error_jsonl: Path) -> None:
    cfg = load_yaml(config_path)
    request_cfg = cfg["request"]
    api_cfg = cfg["api"]
    paths = cfg["paths"]

    rows = load_jsonl(input_jsonl)
    completed_job_ids = load_completed_job_ids(output_jsonl)
    if completed_job_ids:
        rows = [
            row for row in rows
            if str(row.get("job", {}).get("job_id") or row.get("job_id") or "").strip() not in completed_job_ids
        ]
        print(
            f"[resume] skip completed reviews: {len(completed_job_ids)}; remaining rows: {len(rows)}"
        )
    else:
        print(f"[start] review rows to run: {len(rows)}")
    alignment_rows = load_jsonl(Path(paths["alignment_case_pack_jsonl"])) if paths.get("alignment_case_pack_jsonl") else []
    human_aligned_cases: List[Dict[str, Any]] = []
    for row in alignment_rows:
        ann = row.get("annotation_template", {}) or {}
        routing = ann.get("routing_preference") or ann.get("image_dependency")
        if not ann.get("density_level") or not routing:
            continue
        human_aligned_cases.append(
            {
                "sample_id": row.get("sample_id"),
                "text": row.get("text"),
                "density_level": ann.get("density_level"),
                "routing_preference": routing,
                "retrieval_value": ann.get("retrieval_value"),
                "notes": ann.get("notes", ""),
            }
        )
        if len(human_aligned_cases) >= 8:
            break
    prompt_template = read_text(config_path.parent / "prompts" / "review_generated_query_v2.md")

    api_key = os.getenv(api_cfg["key_env"])
    if not api_key:
        raise RuntimeError(f"Missing env: {api_cfg['key_env']}")
    client = AsyncOpenAI(api_key=api_key, base_url=api_cfg.get("base_url"))
    semaphore = asyncio.Semaphore(int(request_cfg["concurrency"]))

    async def worker(row: Dict[str, Any]) -> None:
        prompt = build_prompt(prompt_template, row, human_aligned_cases)
        async with semaphore:
            try:
                obj = await call_model(
                    client=client,
                    model=api_cfg["review_model"],
                    prompt=prompt,
                    timeout_seconds=int(request_cfg["timeout_seconds"]),
                    temperature=float(request_cfg["temperature_review"]),
                    use_json_response_format=bool(request_cfg["use_json_response_format"]),
                )
                with output_jsonl.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"source": row, "review": obj}, ensure_ascii=False) + "\n")
            except Exception as exc:
                with error_jsonl.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"source": row, "error": str(exc)}, ensure_ascii=False) + "\n")

    await asyncio.gather(*(worker(row) for row in rows))
    print(f"[done] {output_jsonl}")
    print(f"[done] {error_jsonl}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Review generated v2 candidates with API.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/mnt/data_1/yds/多模态/权重模块/high_quality_data_pipeline_v2.yaml"),
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path("/mnt/data_1/yds/多模态/权重模块/outputs/v2_pipeline/generated_candidates.v2.jsonl"),
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("/mnt/data_1/yds/多模态/权重模块/outputs/v2_pipeline/reviewed_candidates.v2.jsonl"),
    )
    parser.add_argument(
        "--error-jsonl",
        type=Path,
        default=Path("/mnt/data_1/yds/多模态/权重模块/outputs/v2_pipeline/reviewed_candidates.v2.errors.jsonl"),
    )
    args = parser.parse_args()
    asyncio.run(main_async(args.config, args.input_jsonl, args.output_jsonl, args.error_jsonl))


if __name__ == "__main__":
    main()
