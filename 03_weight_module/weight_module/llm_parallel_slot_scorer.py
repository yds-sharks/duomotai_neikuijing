#!/usr/bin/env python3
import argparse
import asyncio
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_done_ids(output_path: Path) -> set:
    if not output_path.exists():
        return set()
    done = set()
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                cid = obj.get("candidate_id")
                if cid:
                    done.add(cid)
            except Exception:
                continue
    return done


def chunk_list(rows: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
    return [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def ensure_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def ensure_list_map(value: Any) -> Dict[str, List[Any]]:
    if not isinstance(value, dict):
        return {}
    out = {}
    for k, v in value.items():
        if isinstance(v, list):
            out[k] = v
        elif v is None:
            out[k] = []
        else:
            out[k] = [v]
    return out


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def normalize_count(value: int, cap: int) -> float:
    return min(max(value, 0), cap) / float(cap)


def score_anchor(slot_obj: Dict[str, Any], scoring_cfg: Dict[str, Any]) -> Dict[str, Any]:
    caps = scoring_cfg["count_caps"]
    w = scoring_cfg["weights"]
    th = scoring_cfg["thresholds"]

    counts = ensure_dict(slot_obj.get("counts", {}))
    flags = ensure_dict(slot_obj.get("flags", {}))
    scores = ensure_dict(slot_obj.get("scores", {}))
    slots = ensure_list_map(slot_obj.get("slots", {}))

    anatomy_count = safe_int(counts.get("anatomy_count", 0), 0)
    lesion_count = safe_int(counts.get("lesion_count", 0), 0)
    attribute_count = safe_int(counts.get("attribute_count", 0), 0)
    distribution_count = safe_int(counts.get("distribution_count", 0), 0)
    severity_count = safe_int(counts.get("severity_count", 0), 0)
    context_count = safe_int(counts.get("context_count", 0), 0)

    A = normalize_count(anatomy_count, int(caps["anatomy"]))
    L = normalize_count(lesion_count, int(caps["lesion"]))
    AT = normalize_count(attribute_count, int(caps["attribute"]))
    D = normalize_count(distribution_count, int(caps["distribution"]))
    S = normalize_count(severity_count, int(caps["severity"]))

    specificity_score = safe_float(scores.get("specificity_score", 0.0), 0.0)
    generic_penalty = safe_float(scores.get("generic_penalty", 0.0), 0.0)
    deictic_penalty = safe_float(scores.get("deictic_penalty", 0.0), 0.0)

    specificity_score = clamp(specificity_score, 0.0, 1.0)
    generic_penalty = clamp(generic_penalty, 0.0, 1.0)
    deictic_penalty = clamp(deictic_penalty, 0.0, 1.0)

    raw = (
        w["anatomy"] * A
        + w["lesion"] * L
        + w["attribute"] * AT
        + w["distribution"] * D
        + w["severity"] * S
        + w["specificity"] * specificity_score
        - w["generic_penalty"] * generic_penalty
        - w["deictic_penalty"] * deictic_penalty
    )
    score = clamp(raw, 0.0, 1.0) if scoring_cfg.get("enable_score_clamp_01", True) else raw

    # Rule-first class decision
    has_strong = anatomy_count >= 1 or lesion_count >= 1
    has_detail = attribute_count >= 1 or distribution_count >= 1 or severity_count >= 1 or context_count >= 1
    if anatomy_count >= 1 and lesion_count >= 1 and has_detail:
        rule_class = "A"
    elif has_strong:
        rule_class = "B"
    else:
        rule_class = "C"

    # Score boundary adjustment
    if score >= float(th["class_A_min"]):
        score_class = "A"
    elif score >= float(th["class_B_min"]):
        score_class = "B"
    else:
        score_class = "C"

    final_class = rule_class
    if rule_class == "B" and score_class == "A":
        final_class = "A"
    elif rule_class == "C" and score_class in ("A", "B"):
        final_class = score_class

    return {
        "anchor_score_raw": round(raw, 6),
        "anchor_score": round(score, 6),
        "anchor_class_rule": rule_class,
        "anchor_class_score": score_class,
        "anchor_class": final_class,
        "score_components": {
            "A": round(A, 6),
            "L": round(L, 6),
            "AT": round(AT, 6),
            "D": round(D, 6),
            "S": round(S, 6),
            "SP": round(specificity_score, 6),
            "GP": round(generic_penalty, 6),
            "DP": round(deictic_penalty, 6),
        },
        "flags": flags,
        "counts": counts,
        "slots": slots,
        "semantic_core": slot_obj.get("semantic_core", ""),
    }


def build_prompt(batch: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append("你是内窥镜文本结构化分析器。")
    lines.append("任务：对每条文本抽取槽位并返回严格 JSON。")
    lines.append("不允许新增原文没有的事实。")
    lines.append("输出格式必须是：{\"items\":[...]}。")
    lines.append("每个 item 必须包含：")
    lines.append(
        "candidate_id, semantic_core, slots, counts, flags, scores。"
    )
    lines.append("字段约束：")
    lines.append("- slots: anatomy/lesion/attribute/distribution/severity/context/negation/deictic/generic/rare_terms 都是数组")
    lines.append("- counts: anatomy_count/lesion_count/attribute_count/distribution_count/severity_count/context_count 是整数")
    lines.append("- flags: deictic_flag/generic_flag 是布尔")
    lines.append("- scores: specificity_score/generic_penalty/deictic_penalty 是 [0,1] 浮点")
    lines.append("")
    lines.append("待处理样本：")
    for rec in batch:
        lines.append(f"- candidate_id: {rec['candidate_id']}")
        lines.append(f"  text: {rec['anchor_text_norm']}")
    lines.append("")
    lines.append("只返回 JSON，不要解释文本。")
    return "\n".join(lines)


def extract_json_text(raw: str) -> str:
    txt = raw.strip()
    if txt.startswith("{") and txt.endswith("}"):
        return txt
    m = re.search(r"\{[\s\S]*\}", txt)
    if m:
        return m.group(0)
    raise ValueError("No JSON object found in model response")


def parse_response(raw: str) -> Dict[str, Any]:
    js = extract_json_text(raw)
    obj = json.loads(js)
    if not isinstance(obj, dict) or "items" not in obj or not isinstance(obj["items"], list):
        raise ValueError("Invalid response schema: expected {\"items\": [...]} ")
    return obj


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


async def call_with_retry_and_fallback(
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
                wait_s = min(8.0, (2 ** attempt) + random.random())
                await asyncio.sleep(wait_s)
    raise RuntimeError(f"All model retries failed: {last_err}")


async def process_batch(
    client: AsyncOpenAI,
    batch_idx: int,
    batch: List[Dict[str, Any]],
    models: List[str],
    req_cfg: Dict[str, Any],
    scoring_cfg: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    prompt = build_prompt(batch)
    errs = []

    try:
        raw, used_model = await call_with_retry_and_fallback(
            client=client,
            models=models,
            prompt=prompt,
            max_retries=int(req_cfg["max_retries"]),
            timeout_seconds=int(req_cfg["timeout_seconds"]),
            temperature=float(req_cfg["temperature"]),
            use_json_format=bool(req_cfg["use_json_response_format"]),
        )
        parsed = parse_response(raw)
    except Exception as e:
        for rec in batch:
            errs.append(
                {
                    "candidate_id": rec["candidate_id"],
                    "batch_idx": batch_idx,
                    "error": f"request_or_parse_failed: {e}",
                }
            )
        return [], errs

    item_map = {}
    for it in parsed.get("items", []):
        cid = it.get("candidate_id")
        if cid:
            item_map[cid] = it

    out = []
    for rec in batch:
        cid = rec["candidate_id"]
        it = item_map.get(cid)
        if not it:
            errs.append(
                {"candidate_id": cid, "batch_idx": batch_idx, "error": "missing_item_in_model_output"}
            )
            continue

        try:
            scoring = score_anchor(it, scoring_cfg)
            out.append(
                {
                    **rec,
                    **scoring,
                    "semantic_core_hash": "",
                    "llm_model": used_model,
                }
            )
        except Exception as e:
            errs.append(
                {
                    "candidate_id": cid,
                    "batch_idx": batch_idx,
                    "error": f"scoring_failed: {e}",
                }
            )
    return out, errs


async def run(args: argparse.Namespace) -> None:
    cfg = load_config(Path(args.config))
    output_dir = Path(cfg["paths"]["outputs_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = output_dir / "candidate_pool.cleaned.jsonl"
    output_path = output_dir / "anchor_pool.scored.jsonl"
    error_path = output_dir / "anchor_pool.errors.jsonl"

    rows_all = load_jsonl(input_path)
    done_ids = load_done_ids(output_path)
    rows = [r for r in rows_all if r["candidate_id"] not in done_ids]

    req_cfg = cfg["request"]
    max_samples = req_cfg.get("max_samples")
    if args.max_samples is not None:
        max_samples = args.max_samples

    target_total = args.target_total_scored
    if target_total is not None:
        remaining_needed = int(target_total) - len(done_ids)
        if remaining_needed <= 0:
            print(
                f"[done] target already reached: scored={len(done_ids)} target={int(target_total)}"
            )
            return
        rows = rows[:remaining_needed]

    if max_samples:
        rows = rows[: int(max_samples)]

    if not rows:
        print("[done] no pending rows to process")
        return

    key_env = cfg["api"]["key_env"]
    api_key = os.getenv(key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key in env: {key_env}")

    base_url = cfg["api"]["base_url"]
    models = [cfg["api"]["primary_model"]] + list(cfg["api"].get("fallback_models", []))
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    batch_size = int(req_cfg["batch_size"])
    concurrency = int(req_cfg["concurrency"])
    sem = asyncio.Semaphore(concurrency)

    async def worker(i: int, b: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        async with sem:
            return await process_batch(client, i, b, models, req_cfg, cfg["anchor_scoring"])

    # Process in rounds so we can stop exactly at target_total_scored.
    cursor = 0
    batch_idx_base = 0
    progress_total_rows = len(rows)
    progress_done_rows = 0
    scored_now = 0
    new_done_ids = set()

    with output_path.open("a", encoding="utf-8") as fout, error_path.open("a", encoding="utf-8") as ferr:
        with tqdm(total=progress_total_rows, desc="LLM scoring") as pbar:
            while cursor < len(rows):
                if target_total is not None:
                    current_scored = len(done_ids) + scored_now
                    remain_need = int(target_total) - current_scored
                    if remain_need <= 0:
                        break
                    # Keep chunk bounded; never schedule more than needed to avoid overshoot.
                    chunk_rows = min(remain_need, len(rows) - cursor, batch_size * concurrency * 8)
                else:
                    chunk_rows = min(len(rows) - cursor, batch_size * concurrency * 8)

                if chunk_rows <= 0:
                    break

                chunk = rows[cursor : cursor + chunk_rows]
                batches = chunk_list(chunk, batch_size)
                tasks = [
                    asyncio.create_task(worker(batch_idx_base + i, b))
                    for i, b in enumerate(batches, start=1)
                ]

                for coro in asyncio.as_completed(tasks):
                    good, bad = await coro
                    for row in good:
                        cid = row.get("candidate_id")
                        if cid and cid not in done_ids and cid not in new_done_ids:
                            new_done_ids.add(cid)
                            scored_now += 1
                        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    for row in bad:
                        ferr.write(json.dumps(row, ensure_ascii=False) + "\n")

                cursor += chunk_rows
                batch_idx_base += len(batches)
                progress_done_rows += chunk_rows
                pbar.update(chunk_rows)

    final_scored = len(done_ids) + scored_now
    print(f"[done] wrote: {output_path}")
    print(f"[done] errors: {error_path}")
    print(f"[done] scored_total={final_scored}")
    if target_total is not None:
        print(f"[done] target_total_scored={int(target_total)} reached={final_scored >= int(target_total)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel LLM slot extraction + anchor scoring.")
    parser.add_argument(
        "--config",
        default="/mnt/data_1/yds/多模态/权重模块/pipeline_config.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Override max sample count for quick run.")
    parser.add_argument(
        "--target-total-scored",
        type=int,
        default=None,
        help="Stop when total successfully scored lines in output reaches this value.",
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
