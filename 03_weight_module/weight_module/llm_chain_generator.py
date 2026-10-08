#!/usr/bin/env python3
import argparse
import asyncio
import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm


LEVEL_BASE = {"L0": 0.10, "L1": 0.30, "L2": 0.50, "L3": 0.70, "L4": 0.90}


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


def load_done_ids(path: Path) -> set:
    if not path.exists():
        return set()
    done = set()
    with path.open("r", encoding="utf-8") as f:
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


def safe_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    return []


def choose_first(slots: Dict[str, List[str]], key: str) -> str:
    vals = safe_list(slots.get(key, []))
    return vals[0] if vals else ""


def choose_many(slots: Dict[str, List[str]], keys: List[str], limit: int) -> List[str]:
    out = []
    for key in keys:
        for val in safe_list(slots.get(key, [])):
            if val not in out:
                out.append(val)
            if len(out) >= limit:
                return out
    return out


def short_join(parts: List[str], sep: str = " ") -> str:
    return sep.join([p for p in parts if p]).strip()


def choose_second(slots: Dict[str, List[str]], key: str) -> str:
    vals = safe_list(slots.get(key, []))
    return vals[1] if len(vals) > 1 else ""


def join_with_cn(items: List[str], limit: int = 3) -> str:
    vals = [x for x in items if x][:limit]
    if not vals:
        return ""
    if len(vals) == 1:
        return vals[0]
    if len(vals) == 2:
        return f"{vals[0]}和{vals[1]}"
    return f"{vals[0]}、{vals[1]}和{vals[2]}"


def append_q(text: str) -> str:
    s = text.strip().rstrip("？?。")
    return s + "？" if s else ""


def stable_pick(seed: str, options: List[str]) -> str:
    if not options:
        return ""
    h = hashlib.sha1(seed.encode("utf-8")).hexdigest()
    idx = int(h[:8], 16) % len(options)
    return options[idx]


def dedup_items(items: List[str]) -> List[str]:
    out: List[str] = []
    for item in items:
        s = str(item).strip()
        if s and s not in out:
            out.append(s)
    return out


def clip_offset(raw: float) -> float:
    return max(-0.08, min(0.08, 0.16 * (raw - 0.5)))


def make_score(level: str, raw: float) -> float:
    return round(max(0.0, min(1.0, LEVEL_BASE[level] + clip_offset(raw))), 2)


def make_main_chain_skeleton(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidate_id = str(row.get("candidate_id", "unknown"))
    slots = row.get("slots", {}) if isinstance(row.get("slots"), dict) else {}
    anatomy = choose_first(slots, "anatomy")
    anatomy2 = choose_second(slots, "anatomy")
    lesion = choose_first(slots, "lesion")
    lesion2 = choose_second(slots, "lesion")
    attr = choose_first(slots, "attribute")
    dist = choose_first(slots, "distribution")
    sev = choose_first(slots, "severity")
    generic_flag = bool((row.get("flags") or {}).get("generic_flag"))

    l0_options = [
        "帮我看看这个有没有问题",
        "这个情况正常吗",
        "这看着像有问题吗",
        "这个地方是怎么回事",
        "这是什么病变",
        "这个像什么问题",
        "这需要重点排查什么",
        "这个表现说明什么",
        "这属于正常表现吗",
        "这个需要警惕吗",
    ]
    l0_q = stable_pick(f"{candidate_id}:L0", l0_options)
    if generic_flag:
        l0_q = stable_pick(
            f"{candidate_id}:L0:generic",
            ["帮我看看这是什么", "这看起来是什么", "这个到底是什么"],
        )

    if anatomy:
        l1_options = [
            f"这看起来像是{anatomy}附近的问题吗",
            f"这会是{anatomy}这块的异常吗",
            f"这和{anatomy}有关吗",
            f"这像是{anatomy}部位的表现吗",
            f"{anatomy}这里会是什么病变吗",
            f"{anatomy}这里需要排查什么",
            f"{anatomy}这个表现说明什么",
            f"{anatomy}这里需要警惕吗",
            f"{anatomy}这块更像什么问题",
            f"{anatomy}这个情况正常吗",
        ]
    else:
        l1_options = [
            "这看起来像是哪个部位的问题",
            "这更像哪个部位的异常",
            "这会是哪个部位出了问题",
            "这像是哪个位置的表现",
            "这个位置会是什么病变吗",
            "这个部位需要排查什么",
            "这个部位的表现说明什么",
            "这个位置需要警惕吗",
            "这个部位更像什么问题",
            "这个位置正常吗",
        ]
    l1_q = stable_pick(f"{candidate_id}:L1", l1_options)

    if anatomy and lesion:
        l2_options = [
            f"{anatomy}出现{lesion}，这种情况常见吗",
            f"{anatomy}如果有{lesion}，一般提示什么",
            f"{anatomy}看到{lesion}要紧吗",
            f"{anatomy}出现这种{lesion}，通常是什么情况",
            f"{anatomy}有{lesion}时，这是什么病变",
            f"{anatomy}出现{lesion}需要重点排查什么",
            f"{anatomy}看到{lesion}通常说明什么",
            f"{anatomy}有{lesion}时需要警惕吗",
            f"{anatomy}出现{lesion}更像什么问题",
            f"{anatomy}有{lesion}时下一步看什么",
        ]
    elif anatomy and attr:
        l2_options = [
            f"{anatomy}出现{attr}，这种情况常见吗",
            f"{anatomy}如果有{attr}，一般说明什么",
            f"{anatomy}看到这种{attr}正常吗",
            f"{anatomy}出现{attr}时通常考虑什么",
            f"{anatomy}有{attr}时，这是什么病变",
            f"{anatomy}出现{attr}需要重点排查什么",
            f"{anatomy}看到{attr}通常提示什么",
            f"{anatomy}有{attr}时需要警惕吗",
            f"{anatomy}出现{attr}更像什么问题",
            f"{anatomy}有{attr}时下一步看什么",
        ]
    elif lesion:
        l2_options = [
            f"这里像是{lesion}吗",
            f"这种表现会是{lesion}吗",
            f"这里出现的像不像{lesion}",
            f"这种情况要不要考虑{lesion}",
            f"这里有{lesion}时，这是什么病变",
            f"出现{lesion}需要重点排查什么",
            f"看到{lesion}通常说明什么",
            f"有{lesion}时需要警惕吗",
            f"出现{lesion}更像什么问题",
            f"有{lesion}时下一步看什么",
        ]
    else:
        weak_desc = join_with_cn(choose_many(slots, ["attribute", "distribution"], 2), limit=2) or "异常表现"
        l2_options = [
            f"这里出现{weak_desc}，这种情况常见吗",
            f"这种{weak_desc}一般说明什么",
            f"看到{weak_desc}需要考虑什么",
            f"这种{weak_desc}是常见表现吗",
            f"有{weak_desc}时，这是什么病变",
            f"出现{weak_desc}需要重点排查什么",
            f"看到{weak_desc}通常提示什么",
            f"有{weak_desc}时需要警惕吗",
            f"出现{weak_desc}更像什么问题",
            f"有{weak_desc}时下一步看什么",
        ]
    l2_q = stable_pick(f"{candidate_id}:L2", l2_options)

    l3_parts = []
    if anatomy:
        l3_parts.append(anatomy)
    if lesion:
        l3_parts.append(lesion)
    if attr:
        l3_parts.append(attr)
    elif dist:
        l3_parts.append(dist)
    l3_desc = join_with_cn(l3_parts, limit=3)
    if l3_desc:
        l3_options = [
            f"{l3_desc}同时出现，这是什么情况",
            f"如果有{l3_desc}这些表现，一般考虑什么",
            f"{l3_desc}一起出现时更像什么问题",
            f"出现{l3_desc}时通常提示什么",
            f"{l3_desc}同时出现时，这是什么病变",
            f"看到{l3_desc}这些线索，需要重点排查什么",
            f"{l3_desc}放在一起通常说明什么",
            f"如果同时有{l3_desc}，需要警惕什么",
            f"{l3_desc}这些表现更符合哪类情况",
            f"结合{l3_desc}，下一步应该看什么",
        ]
    else:
        l3_options = [
            "这种表现更像什么情况",
            "这种组合一般提示什么",
            "这些表现一起出现说明什么",
            "这种变化通常考虑什么问题",
            "这些表现一起出现时，这是什么病变",
            "看到这些线索需要重点排查什么",
            "这些表现放在一起通常说明什么",
            "这种组合需要警惕什么",
            "这些表现更符合哪类情况",
            "结合这些表现下一步应该看什么",
        ]
    l3_q = stable_pick(f"{candidate_id}:L3", l3_options)

    l4_parts = []
    l4_parts.extend(dedup_items([anatomy, anatomy2, lesion, attr or dist, sev or lesion2]))
    l4_desc = join_with_cn(l4_parts, limit=5)
    if l4_desc:
        l4_options = [
            f"如果{l4_desc}都存在，下一步该重点排查什么",
            f"当{l4_desc}同时出现时，更符合哪类情况",
            f"看到{l4_desc}这些表现时，临床上会优先考虑什么",
            f"{l4_desc}一起出现时，更像哪一类问题",
            f"这些{l4_desc}表现放在一起，应该往哪个方向判断",
            f"结合{l4_desc}这些线索，通常需要警惕什么",
            f"{l4_desc}同时存在时，和哪些疾病表现比较接近",
            f"如果同时看到{l4_desc}，该怎么理解这种表现",
            f"这种{l4_desc}的组合，通常会怎么判断",
            f"从{l4_desc}这些表现看，需要重点考虑哪种情况",
            f"{l4_desc}这些表现同时存在时，这是什么病变",
            f"综合{l4_desc}这些信息，最需要排除什么",
            f"如果{l4_desc}都能对应上，应该优先想到什么",
            f"把{l4_desc}这些细节合在一起看，说明什么",
            f"面对{l4_desc}这些表现，下一步应该确认什么",
        ]
    else:
        l4_options = [
            "如果这些表现同时存在，下一步该重点排查什么",
            "这些异常一起出现时，通常更符合哪类情况",
            "看到这些表现组合时，临床上会优先考虑什么",
            "这些变化同时存在时，应该往哪个方向判断",
            "这种表现组合通常需要警惕什么",
            "这些线索放在一起，该怎么理解",
            "这些表现同时存在时，这是什么病变",
            "综合这些信息，最需要排除什么",
            "如果这些线索都能对应上，应该优先想到什么",
            "把这些细节合在一起看，说明什么",
            "面对这些表现，下一步应该确认什么",
        ]
    l4_q = stable_pick(f"{candidate_id}:L4", l4_options)

    l0_q = append_q(l0_q)
    l1_q = append_q(l1_q)
    l2_q = append_q(l2_q)
    l3_q = append_q(l3_q)
    l4_q = append_q(l4_q)

    return [
        {
            "level": "L0",
            "query_text": l0_q,
            "kept_slots": [],
            "dropped_slots": ["anatomy", "lesion", "attribute", "distribution", "severity", "context"],
            "image_dependency": 2,
            "rule_density_raw": 0.12,
            "density_score": make_score("L0", 0.12),
            "style_tag": "deictic",
        },
        {
            "level": "L1",
            "query_text": l1_q,
            "kept_slots": [k for k in ["anatomy"] if choose_first(slots, k)],
            "dropped_slots": ["lesion", "attribute", "distribution", "severity", "context"],
            "image_dependency": 2,
            "rule_density_raw": 0.22,
            "density_score": make_score("L1", 0.22),
            "style_tag": "weak_constraint",
        },
        {
            "level": "L2",
            "query_text": l2_q,
            "kept_slots": [k for k in ["anatomy", "lesion"] if choose_first(slots, k)][:2],
            "dropped_slots": ["attribute", "distribution", "severity", "context"],
            "image_dependency": 1,
            "rule_density_raw": 0.44,
            "density_score": make_score("L2", 0.44),
            "style_tag": "keyword_question",
        },
        {
            "level": "L3",
            "query_text": l3_q,
            "kept_slots": [k for k in ["anatomy", "lesion", "attribute", "distribution"] if choose_first(slots, k)][:3],
            "dropped_slots": ["severity", "context"],
            "image_dependency": 0,
            "rule_density_raw": 0.67,
            "density_score": make_score("L3", 0.67),
            "style_tag": "clinical_colloquial",
        },
        {
            "level": "L4",
            "query_text": l4_q,
            "kept_slots": [k for k in ["anatomy", "lesion", "attribute", "distribution", "severity", "context"] if choose_first(slots, k)][:5],
            "dropped_slots": [],
            "image_dependency": 0,
            "rule_density_raw": 0.86,
            "density_score": make_score("L4", 0.86),
            "style_tag": "high_constraint",
        },
    ]


def make_variants_and_hardcases(row: Dict[str, Any], chain: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    l2 = next(item for item in chain if item["level"] == "L2")
    l3 = next(item for item in chain if item["level"] == "L3")
    l4 = next(item for item in chain if item["level"] == "L4")

    variants = [
        {"base_level": "L2", "query_text": l2["query_text"], "style_tag": "keyword"},
        {"base_level": "L3", "query_text": l3["query_text"], "style_tag": "spoken_question"},
        {"base_level": "L4", "query_text": l4["query_text"], "style_tag": "concise_professional"},
    ]
    hard_cases = [
        {
            "case_type": "long_but_low_density",
            "query_text": "这种情况到底是怎么回事？",
            "expected_level": "L1",
        },
        {
            "case_type": "short_but_high_density",
            "query_text": append_q(
                f"{join_with_cn([choose_first(row.get('slots', {}), 'anatomy'), choose_first(row.get('slots', {}), 'lesion'), choose_first(row.get('slots', {}), 'attribute')], limit=3)}同时出现说明什么"
            ),
            "expected_level": "L3",
        },
    ]
    return variants, hard_cases


def build_scaffold(row: Dict[str, Any]) -> Dict[str, Any]:
    main_chain = make_main_chain_skeleton(row)
    style_variants, hard_cases = make_variants_and_hardcases(row, main_chain)
    return {
        "anchor_text": row.get("anchor_text_norm") or row.get("anchor_text", ""),
        "semantic_core": row.get("semantic_core", ""),
        "semantic_core_hash": row.get("semantic_core_hash", ""),
        "constraints": {
            "semantic_core_locked": True,
            "must_not_add_facts": True,
        },
        "main_chain": main_chain,
        "style_variants": style_variants,
        "hard_cases": hard_cases,
    }


def looks_like_user_query(text: str) -> bool:
    if not isinstance(text, str):
        return False
    s = text.strip()
    if not s:
        return False
    if s.endswith("。"):
        return False
    if len(s) > 80:
        return False
    if not s.endswith("？") and not s.endswith("?"):
        return False
    return True


def validate_chain(parsed: Dict[str, Any]) -> List[str]:
    issues = []
    chain = parsed.get("main_chain", [])
    if not isinstance(chain, list) or len(chain) != 5:
        issues.append("main_chain_not_5")
        return issues
    expected_levels = ["L0", "L1", "L2", "L3", "L4"]
    prev_score = -1.0
    for idx, (item, level) in enumerate(zip(chain, expected_levels)):
        if item.get("level") != level:
            issues.append(f"wrong_level_at_{idx}")
        if not looks_like_user_query(item.get("query_text", "")):
            issues.append(f"unnatural_query_format_at_{idx}")
        score = item.get("density_score")
        if not isinstance(score, (int, float)):
            issues.append(f"missing_score_at_{idx}")
        else:
            if float(score) <= prev_score:
                issues.append(f"non_monotonic_score_at_{idx}")
            prev_score = float(score)
    l3 = str(chain[3].get("query_text", ""))
    l4 = str(chain[4].get("query_text", ""))
    # L4 should be at least as specific as L3, but natural question forms do not
    # always make the highest-density question strictly longer in characters.
    if len(l4) + 8 < len(l3):
        issues.append("l3_not_shorter_than_l4")
    return issues


def validate_rewrites(parsed: Dict[str, Any]) -> List[str]:
    issues = []
    rewrites = parsed.get("rewrites")
    if not isinstance(rewrites, dict):
        return ["missing_rewrites"]
    for level in ["L0", "L1", "L2", "L3", "L4"]:
        text = rewrites.get(level)
        if not looks_like_user_query(text):
            issues.append(f"bad_rewrite_{level}")

    variants = parsed.get("variants")
    if not isinstance(variants, dict):
        issues.append("missing_variants")
    else:
        for level in ["L2", "L3", "L4"]:
            if not looks_like_user_query(variants.get(level, "")):
                issues.append(f"bad_variant_{level}")

    hard_cases = parsed.get("hard_cases")
    if not isinstance(hard_cases, dict):
        issues.append("missing_hard_cases")
    else:
        for key in ["long_but_low_density", "short_but_high_density"]:
            if not looks_like_user_query(hard_cases.get(key, "")):
                issues.append(f"bad_hard_case_{key}")
    return issues


def merge_rewrites(scaffold: Dict[str, Any], parsed: Dict[str, Any]) -> Dict[str, Any]:
    out = json.loads(json.dumps(scaffold, ensure_ascii=False))
    rewrites = parsed["rewrites"]
    variants = parsed["variants"]
    hard_cases = parsed["hard_cases"]

    for item in out["main_chain"]:
        level = item["level"]
        item["query_text"] = str(rewrites[level]).strip()
    for item in out["style_variants"]:
        item["query_text"] = str(variants[item["base_level"]]).strip()
    for item in out["hard_cases"]:
        item["query_text"] = str(hard_cases[item["case_type"]]).strip()
    return out


def build_prompt(template: str, row: Dict[str, Any], scaffold: Dict[str, Any]) -> str:
    payload = {
        "candidate_id": row.get("candidate_id", ""),
        "anchor_text": row.get("anchor_text_norm") or row.get("anchor_text", ""),
        "slots": row.get("slots", {}),
        "counts": row.get("counts", {}),
        "flags": row.get("flags", {}),
        "semantic_core": row.get("semantic_core", ""),
        "semantic_core_hash": row.get("semantic_core_hash", ""),
        "anchor_class": row.get("anchor_class", ""),
        "scaffold_chain": scaffold,
    }
    return f"{template}\n\n输入样本(JSON):\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n"


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
    scaffold = build_scaffold(row)
    prompt = build_prompt(template, row, scaffold)
    try:
        raw, used_model = await call_with_retry(
            client=client,
            models=models,
            prompt=prompt,
            max_retries=int(stage_cfg["max_retries"]),
            timeout_seconds=int(stage_cfg["timeout_seconds"]),
            temperature=float(stage_cfg["temperature_generate"]),
            use_json_format=bool(stage_cfg["use_json_response_format"]),
        )
        parsed = parse_json_object(raw)
        local_issues = validate_rewrites(parsed)
        if local_issues:
            raise ValueError("local_validation_failed:" + ",".join(local_issues))
        merged = merge_rewrites(scaffold, parsed)
        final_issues = validate_chain(merged)
        if final_issues:
            raise ValueError("final_validation_failed:" + ",".join(final_issues))
        merged["candidate_id"] = row.get("candidate_id")
        merged["split"] = row.get("split")
        merged["anchor_class"] = row.get("anchor_class")
        merged["llm_model"] = used_model
        merged["scaffold_chain"] = scaffold
        return merged, {}
    except Exception as e:
        return {}, {
            "candidate_id": row.get("candidate_id"),
            "split": row.get("split"),
            "anchor_class": row.get("anchor_class"),
            "error": str(e),
            "scaffold_chain": scaffold,
        }


async def run(args: argparse.Namespace) -> None:
    cfg = load_config(Path(args.config))
    stage_cfg = cfg["generation_stage"]
    input_path = Path(args.input or cfg["stage3_outputs"]["pilot_anchor_input"])
    out_path = Path(args.output or cfg["stage3_outputs"]["generated_chain_pilot"])
    err_path = Path(args.error_output or cfg["stage3_outputs"]["generated_chain_errors"])
    template = load_text(Path("/mnt/data_1/yds/多模态/权重模块/rewrite_chain_prompt.md"))

    rows = load_jsonl(input_path)
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
    done_ids = load_done_ids(out_path)
    rows = [r for r in rows if r.get("candidate_id") not in done_ids]

    api_key = os.getenv(cfg["api"]["key_env"])
    if not api_key:
        raise RuntimeError(f"Missing env: {cfg['api']['key_env']}")
    client = AsyncOpenAI(api_key=api_key, base_url=cfg["api"]["base_url"])
    models = [stage_cfg["generation_model"]] + list(cfg["api"].get("fallback_models", []))
    sem = asyncio.Semaphore(int(stage_cfg["concurrency"]))

    async def guarded(row: Dict[str, Any]):
        async with sem:
            return await process_one(client, row, template, models, stage_cfg)

    tasks = [asyncio.create_task(guarded(r)) for r in rows]
    with out_path.open("a", encoding="utf-8") as fout, err_path.open("a", encoding="utf-8") as ferr:
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generate chains"):
            good, bad = await coro
            if good:
                fout.write(json.dumps(good, ensure_ascii=False) + "\n")
            if bad:
                ferr.write(json.dumps(bad, ensure_ascii=False) + "\n")

    print("[done]", out_path)
    print("[done]", err_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pilot L0-L4 chains.")
    parser.add_argument("--config", default="/mnt/data_1/yds/多模态/权重模块/pipeline_config.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--error-output", default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
