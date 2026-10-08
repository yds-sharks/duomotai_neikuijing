#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_json_text(raw: str) -> str:
    txt = raw.strip()
    if txt.startswith("{") and txt.endswith("}"):
        return txt
    m = re.search(r"\{[\s\S]*\}", txt)
    if m:
        return m.group(0)
    raise ValueError("No JSON found in response")


def parse_json_object(raw: str) -> Dict[str, Any]:
    return json.loads(extract_json_text(raw))


def build_prompt(template: str, row: Dict[str, Any]) -> str:
    payload = {
        "candidate_id": row.get("candidate_id"),
        "anchor_text": row.get("anchor_text", ""),
        "semantic_core": row.get("semantic_core", ""),
        "semantic_core_hash": row.get("semantic_core_hash", ""),
        "generated_chain": row,
    }
    return f"{template}\n\n待审查输入(JSON):\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n"


async def call_once(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    timeout_seconds: int,
    temperature: float,
    use_json_format: bool,
) -> str:
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "timeout": timeout_seconds,
    }
    if use_json_format:
        kwargs["response_format"] = {"type": "json_object"}
    resp = await client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()


async def call_with_retry(
    client: AsyncOpenAI,
    models: List[str],
    prompt: str,
    max_retries: int,
    timeout_seconds: int,
    temperature: float,
    use_json_format: bool,
) -> Tuple[str, str]:
    last_err = None
    for model in models:
        for attempt in range(max_retries):
            try:
                text = await call_once(client, model, prompt, timeout_seconds, temperature, use_json_format)
                return text, model
            except Exception as e:
                last_err = e
                await asyncio.sleep(min(8.0, (2 ** attempt) + random.random()))
    raise RuntimeError(f"all retries failed: {last_err}")


async def process_one(
    client: AsyncOpenAI,
    row: Dict[str, Any],
    template: str,
    models: List[str],
    stage_cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    prompt = build_prompt(template, row)
    try:
        raw, used_model = await call_with_retry(
            client=client,
            models=models,
            prompt=prompt,
            max_retries=int(stage_cfg["max_retries"]),
            timeout_seconds=int(stage_cfg["timeout_seconds"]),
            temperature=float(stage_cfg["temperature_review"]),
            use_json_format=bool(stage_cfg["use_json_response_format"]),
        )
        parsed = parse_json_object(raw)
        parsed["candidate_id"] = row.get("candidate_id")
        parsed["split"] = row.get("split")
        parsed["anchor_class"] = row.get("anchor_class")
        parsed["anchor_text"] = row.get("anchor_text")
        parsed["semantic_core"] = row.get("semantic_core")
        parsed["semantic_core_hash"] = row.get("semantic_core_hash")
        parsed["generated_chain"] = row
        parsed["review_model"] = used_model
        return parsed, {}
    except Exception as e:
        return {}, {
            "candidate_id": row.get("candidate_id"),
            "split": row.get("split"),
            "anchor_class": row.get("anchor_class"),
            "error": str(e),
        }


async def run(args: argparse.Namespace) -> None:
    cfg = load_config(Path(args.config))
    stage_cfg = cfg["generation_stage"]
    input_path = Path(args.input or cfg["stage3_outputs"]["generated_chain_pilot"])
    reviewed_path = Path(args.reviewed_output or cfg["stage3_outputs"]["reviewed_chain_pilot"])
    rejected_path = Path(args.rejected_output or cfg["stage3_outputs"]["rejected_chain_pilot"])
    err_path = Path(args.error_output or cfg["stage3_outputs"]["review_errors"])
    report_path = Path(args.report_output or cfg["stage3_outputs"]["quality_report_pilot"])
    template = load_text(Path("/mnt/data_1/yds/多模态/text_density_design/prompts/review_chain_prompt.md"))

    rows = load_jsonl(input_path)
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
    api_key = os.getenv(cfg["api"]["key_env"])
    if not api_key:
        raise RuntimeError(f"Missing env: {cfg['api']['key_env']}")
    client = AsyncOpenAI(api_key=api_key, base_url=cfg["api"]["base_url"])
    models = [stage_cfg["review_model"]] + list(cfg["api"].get("fallback_models", []))
    sem = asyncio.Semaphore(int(stage_cfg["concurrency"]))

    async def guarded(row: Dict[str, Any]):
        async with sem:
            return await process_one(client, row, template, models, stage_cfg)

    tasks = [asyncio.create_task(guarded(r)) for r in rows]

    stats = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "new_fact": 0,
        "non_monotonic_level": 0,
        "non_monotonic_score": 0,
        "unnatural": 0,
    }

    with reviewed_path.open("w", encoding="utf-8") as fpass, rejected_path.open("w", encoding="utf-8") as frej, err_path.open("w", encoding="utf-8") as ferr:
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Review chains"):
            good, bad = await coro
            if bad:
                ferr.write(json.dumps(bad, ensure_ascii=False) + "\n")
                continue
            stats["total"] += 1
            issues = good.get("issues", []) if isinstance(good.get("issues"), list) else []
            codes = {it.get("code") for it in issues if isinstance(it, dict)}
            if "NEW_FACT" in codes:
                stats["new_fact"] += 1
            if "NON_MONOTONIC_LEVEL" in codes:
                stats["non_monotonic_level"] += 1
            if "NON_MONOTONIC_SCORE" in codes:
                stats["non_monotonic_score"] += 1
            if "UNNATURAL_QUERY" in codes:
                stats["unnatural"] += 1

            if bool(good.get("pass")):
                stats["passed"] += 1
                fpass.write(json.dumps(good, ensure_ascii=False) + "\n")
            else:
                stats["failed"] += 1
                frej.write(json.dumps(good, ensure_ascii=False) + "\n")

    total = max(stats["total"], 1)
    report = {
        **stats,
        "pass_rate": round(stats["passed"] / total, 6),
        "new_fact_rate": round(stats["new_fact"] / total, 6),
        "non_monotonic_rate": round((stats["non_monotonic_level"] + stats["non_monotonic_score"]) / total, 6),
        "unnatural_rate": round(stats["unnatural"] / total, 6),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[done]", reviewed_path)
    print("[done]", rejected_path)
    print("[done]", err_path)
    print("[done]", report_path)
    print(report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Review pilot generated chains.")
    parser.add_argument(
        "--config",
        default="/mnt/data_1/yds/多模态/权重模块/pipeline_config.yaml",
    )
    parser.add_argument("--input", default=None)
    parser.add_argument("--reviewed-output", default=None)
    parser.add_argument("--rejected-output", default=None)
    parser.add_argument("--error-output", default=None)
    parser.add_argument("--report-output", default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
