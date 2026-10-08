#!/usr/bin/env python3
"""Batch-translate EndoBench English questions to Chinese for retrieval.

Uses the same translation prompt as retrieval/test/utils.py:
  1. Remove option letters (A. B. C. D.) but keep option content
  2. Translate to Chinese
  3. Rewrite for better retrieval

Output: eval/endobench_translated_queries.jsonl  (one line per question, indexed)
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# --- API config ---
API_CONFIG_PATH = Path("/mnt/data_1/yds/多模态/agentic/data_construction/api_config.local.json")

def load_api_config() -> dict:
    return json.loads(API_CONFIG_PATH.read_text(encoding="utf-8"))

TRANSLATION_PROMPT = """请帮我改写以下用于医学问答检索的 query。

要求：
1. 去掉选项前的字母标识（如 A. B. C. D. 等），但是要保留选项内容
2. 将 query 翻译成中文
3. 可以适当使用更易于检索的表述改写

原始 query：
{query}

请直接输出改写后的 query，不要包含任何解释或其他内容。"""


async def translate_one(client, model: str, query: str, idx: int) -> dict:
    """Translate a single query via API."""
    prompt = TRANSLATION_PROMPT.format(query=query)
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        result = resp.choices[0].message.content.strip() if resp.choices else ""
        return {"index": idx, "query_en": query, "query_zh": result, "ok": bool(result)}
    except Exception as e:
        return {"index": idx, "query_en": query, "query_zh": "", "ok": False, "error": str(e)}


async def main(args):
    from openai import AsyncOpenAI
    from datasets import load_dataset

    cfg = load_api_config()
    base_url = cfg["base_url"].rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    client = AsyncOpenAI(api_key=cfg["api_key"], base_url=base_url)
    model = cfg.get("model", "gpt-5.4")

    # Load EndoBench
    ds = load_dataset(args.benchmark, split=args.split)
    n = len(ds)
    print(f"[data] {args.benchmark}[{args.split}] n={n}")

    # Load existing translations (resume)
    done = {}
    out_path = Path(args.output)
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").strip().split("\n"):
            row = json.loads(line)
            done[row["index"]] = row
        print(f"[resume] {len(done)} already translated")

    # Build queries
    tasks = []
    for i in range(n):
        if i in done and done[i].get("ok"):
            continue
        ex = ds[i]
        q = str(ex.get("question") or "").strip()
        # Build options
        options = {}
        for L in "ABCDEFGH":
            t = ex.get(f"option_{L}")
            if t and str(t).strip():
                options[L] = str(t).strip()
        opt_text = "\n".join(f"{L}: {t}" for L, t in options.items())
        query = f"{q}\n{opt_text}" if opt_text else q
        tasks.append((client, model, query, i))

    print(f"[todo] {len(tasks)} queries to translate")
    if not tasks:
        print("[done] all translations complete")
        return

    # Concurrent translation
    sem = asyncio.Semaphore(args.concurrency)
    async def bounded(t):
        async with sem:
            return await translate_one(*t)

    results = []
    batch_size = args.concurrency * 2
    for start in range(0, len(tasks), batch_size):
        batch = tasks[start:start + batch_size]
        batch_results = await asyncio.gather(*[bounded(t) for t in batch])
        results.extend(batch_results)
        # Append to output file
        with open(out_path, "a", encoding="utf-8") as f:
            for r in batch_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        ok_count = sum(1 for r in results if r["ok"])
        print(f"[progress] {len(results)}/{len(tasks)} done (ok={ok_count})")

    # Merge with existing
    all_results = dict(done)
    for r in results:
        all_results[r["index"]] = r

    # Write final sorted output
    with open(out_path, "w", encoding="utf-8") as f:
        for i in sorted(all_results.keys()):
            f.write(json.dumps(all_results[i], ensure_ascii=False) + "\n")

    ok_total = sum(1 for r in all_results.values() if r.get("ok"))
    print(f"[done] {ok_total}/{n} translated successfully -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="Saint-lsy/EndoBench")
    ap.add_argument("--split", default="test")
    ap.add_argument("--output", default=str(HERE / "endobench_translated_queries.jsonl"))
    ap.add_argument("--concurrency", type=int, default=20)
    main_args = ap.parse_args()
    asyncio.run(main(main_args))
