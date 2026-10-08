#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块名：qwen3_test.py

功能：
- 从指定 JSONL（默认：results_merged-8B_top5_rag.jsonl）逐行读取样本；
- 每条样本使用 description + top5 构造 3 个不同 Prompt；
- 调用 SiliconFlow 上的 Qwen3 模型生成 rag_answer1/2/3；
- 边处理边写：每处理完一条，立即写入输出 JSONL 一行；
- 控制台打印进度：Sample i/total。
"""

import argparse
import json
import os
import pathlib
from typing import Dict, Any, List

from openai import OpenAI

# ==================== Qwen3 API 配置 ====================

# 推荐用环境变量保存 key：export SILICONFLOW_API_KEY="sk-xxxx"
API_KEY = "sk-wcxgnlhmttndxzccprdqdgkbrlfakjkxjkbdtlbpilnfeevy"
BASE_URL = "https://api.siliconflow.cn/v1"
MODEL_NAME = "Qwen/Qwen3-VL-32B-Instruct"  # 按你实际可用的模型改

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)


def call_qwen3(prompt: str) -> str:
    """
    给定一个 prompt，调用 SiliconFlow 上的 Qwen3 模型，返回生成文本。
    """
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful pathology assistant.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        max_tokens=1024,
        temperature=0.2,
    )

    content = resp.choices[0].message.content
    if isinstance(content, List):
        text_out = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    else:
        text_out = str(content)

    return text_out.strip()


# ==================== 三个 Prompt 模板 ====================

PROMPT_TEMPLATE_1 = """You are a breast pathology specialist. You are given an original English description of a ROI (region of interest) and several retrieved reference passages. Your task is to produce an evidence-grounded refinement of the original description.

[Original description]
{description}

[Retrieved reference passages (from textbooks, guidelines, or pathology reports)]
{contexts}

Please:
1. Stay strictly within the information that can be reasonably supported by the reference passages. Do NOT invent new pathological facts.
2. Prefer standard pathological terminology from the reference passages (for stroma, cellular morphology, architectural patterns, nuclear features, etc.) to replace vague or colloquial expressions in the original description.
3. Focus only on microscopic morphology: architecture, cell types, nuclear features, stromal components, inflammation/necrosis, etc. Do NOT discuss treatment, prognosis, or clinical recommendations.
4. Produce a single, coherent, concise **English microscopic description** in one paragraph, without bullet points, meta-comments, or explanations.

Now provide the refined English microscopic description.
"""

PROMPT_TEMPLATE_2 = """You are an experienced pathologist. Based on the original ROI description and several retrieved reference passages, you need to rewrite the ROI description to make it more accurate, standardized, and well-structured.

[Original ROI description]
{description}

[Retrieved reference passages (for reference and support)]
{contexts}

Please follow these principles:
1. Preserve the useful content of the original description and integrate it with the key information from the reference passages, forming a clearer and more logically organized description.
2. Only add or modify details when they are reasonably supported by the reference passages. If there is insufficient evidence, keep the wording neutral and cautious.
3. Emphasize: architectural patterns (e.g., lobular, ductal, fibrous stroma), cellular composition and degree of atypia, nuclear features, spatial distribution, and the presence or absence of necrosis, inflammation, or fibrosis.
4. Do NOT jump to a diagnostic label (e.g., benign/malignant). Stay at the level of morphological description only.
5. Output exactly **one paragraph in English**, and do not mention “passages above”, “references”, or your reasoning process.

Please provide the final integrated English microscopic description.
"""

PROMPT_TEMPLATE_3 = """You are acting as an attending pathologist writing a standardized English microscopic description for a ROI. You have the original description and several retrieved reference passages. Do your reasoning internally and only output the final description.

[Original description]
{description}

[Retrieved reference passages]
{contexts}

Your goals:
- Describe the ROI as accurately as possible without going beyond what can be reasonably inferred from the reference passages.
- Prioritize objective, reproducible, observation-based language rather than subjective speculation.
- Use standard pathological terms to refine or replace ambiguous expressions in the original description.

Requirements:
1. First (internally) organize the overall background: type of stroma (e.g., fibrous), presence of ducts/glands/lobules/nodular structures, and the main cell populations, including whether there is obvious atypia.
2. Then integrate nuclear features, cytoplasmic features, and distribution patterns, and clearly describe any key abnormal changes (e.g., hypercellularity, pleomorphism, mitotic activity, necrosis, inflammation).
3. Finally, output **one English microscopic description paragraph** with an objective and neutral tone. Do not include “maybe”, “likely”, or treatment/prognosis suggestions.

Only output the final English microscopic description, and do not explain your reasoning.
"""


# ==================== 工具函数 ====================

def count_nonempty_lines(path: pathlib.Path) -> int:
    """统计非空行数，用于进度显示。"""
    cnt = 0
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                cnt += 1
    return cnt


def build_context_from_top5(top5: Any) -> str:
    """把 top5 文段拼成上下文字符串。"""
    segs = []
    if not isinstance(top5, list):
        return ""
    for i, h in enumerate(top5):
        if not isinstance(h, dict):
            continue
        text = (h.get("text") or "").strip()
        if not text:
            continue
        segs.append(f"[{i+1}] {text}")
    return "\n\n".join(segs)


# ==================== 主流程 ====================

def main():
    DEFAULT_IN = "/mnt/data_1/yds/RAG/Hybrid_milvus/总版/insert/数据临时站/results_merged-8B_top5_rag.jsonl"

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default=DEFAULT_IN,
        help=f"输入 JSONL 文件路径（默认：{DEFAULT_IN}）",
    )
    ap.add_argument(
        "--output",
        default="",
        help="输出 JSONL 文件路径（默认在 input 文件名后加 _filled.jsonl）",
    )
    args = ap.parse_args()

    in_path = pathlib.Path(args.input).resolve()
    if args.output:
        out_path = pathlib.Path(args.output).resolve()
    else:
        out_path = in_path.with_name(in_path.stem + "_32B_filled.jsonl")

    if not in_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {in_path}")

    total = count_nonempty_lines(in_path)
    print(f"[INFO] 输入文件: {in_path}")
    print(f"[INFO] 输出文件: {out_path}")
    print(f"[INFO] 非空样本行数: {total}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 边读边写
    with in_path.open("r", encoding="utf-8") as fin, \
            out_path.open("w", encoding="utf-8") as fout:

        idx = 0
        for ln in fin:
            if not ln.strip():
                continue

            idx += 1
            try:
                item: Dict[str, Any] = json.loads(ln)
            except Exception as e:
                print(f"[WARN] 行 {idx} JSON 解析失败，原样写回: {e}")
                fout.write(ln)
                fout.flush()
                continue

            desc = (item.get("description") or "").strip()
            top5 = item.get("top5") or []
            contexts = build_context_from_top5(top5)

            print(f"\n========== Sample {idx}/{total} ==========")
            print("[DESCRIPTION]")
            print(desc)
            print("\n[CONTEXTS]")
            print(contexts)

            # 默认先设为空，避免异常时 key 缺失
            ans1 = item.get("rag_answer1", "")
            ans2 = item.get("rag_answer2", "")
            ans3 = item.get("rag_answer3", "")

            if desc and contexts:
                prompt1 = PROMPT_TEMPLATE_1.format(description=desc, contexts=contexts)
                prompt2 = PROMPT_TEMPLATE_2.format(description=desc, contexts=contexts)
                prompt3 = PROMPT_TEMPLATE_3.format(description=desc, contexts=contexts)

                try:
                    ans1 = call_qwen3(prompt1)
                    print("\n[rag_answer1]")
                    print(ans1)
                except Exception as e:
                    print(f"[WARN] 样本 {idx} rag_answer1 调用失败: {e}")
                    ans1 = item.get("rag_answer1", "")

                try:
                    ans2 = call_qwen3(prompt2)
                    print("\n[rag_answer2]")
                    print(ans2)
                except Exception as e:
                    print(f"[WARN] 样本 {idx} rag_answer2 调用失败: {e}")
                    ans2 = item.get("rag_answer2", "")

                try:
                    ans3 = call_qwen3(prompt3)
                    print("\n[rag_answer3]")
                    print(ans3)
                except Exception as e:
                    print(f"[WARN] 样本 {idx} rag_answer3 调用失败: {e}")
                    ans3 = item.get("rag_answer3", "")

            item["rag_answer1"] = ans1
            item["rag_answer2"] = ans2
            item["rag_answer3"] = ans3

            # 立刻写入一行
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"\n[OK] 流式处理完成，结果已写入：{out_path}")


if __name__ == "__main__":
    main()
