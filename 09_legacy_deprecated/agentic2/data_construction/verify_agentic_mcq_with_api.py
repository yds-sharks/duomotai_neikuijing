#!/usr/bin/env python3
"""Optional source-grounded MCQ verifier using an OpenAI-compatible API.

This script filters template-generated MCQs. It does not ask the model to create
or choose the gold answer from scratch; it asks whether the existing answer is
supported by the provided source evidence.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_BASE_URL = "https://api.openai.com/v1"


SYSTEM_PROMPT = """你是医学多模态选择题数据质检员。给定一道已构造的四选一题、结构化正确答案、来源证据文本和可选图像，判断该答案是否被来源证据支持。

你不能重新发明答案，只能核验给定答案是否正确、是否有歧义、是否存在答案泄露或干扰项明显错误。
只输出 JSON，不要解释性文本。"""


USER_PROMPT_TEMPLATE = """请核验下面的选择题。

问题：{question}
选项：{options}
给定正确答案：{answer}. {answer_text}

来源信息：
- doc_id: {doc_id}
- doc_name: {doc_name}
- source_path: {source_path}
- origin_pdf_path: {origin_pdf_path}
- page_idx: {page_idx}
- image_path: {image_path}
- level1: {level1}
- level2: {level2}

证据文本：
{evidence_text}

请只输出如下 JSON：
{{
  "supported": true/false,
  "answer_correct": true/false,
  "distractors_plausible_but_wrong": true/false,
  "ambiguous": true/false,
  "leakage": true/false,
  "needs_pdf_page": true/false,
  "confidence": 0.0-1.0,
  "reason": "不超过80字"
}}"""


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc


def write_jsonl_row(handle: Any, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def image_to_data_uri(path: str, max_bytes: int) -> str:
    if not path:
        return ""
    image_path = Path(path)
    if not image_path.exists() or not image_path.is_file():
        return ""
    if image_path.stat().st_size > max_bytes:
        return ""
    mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
    data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{data}"


def build_messages(row: Dict[str, Any], include_image: bool, max_image_bytes: int) -> List[Dict[str, Any]]:
    source = row.get("source") or {}
    user_text = USER_PROMPT_TEMPLATE.format(
        question=row.get("question", ""),
        options=json.dumps(row.get("options", {}), ensure_ascii=False),
        answer=row.get("answer", ""),
        answer_text=row.get("answer_text", ""),
        doc_id=source.get("doc_id", ""),
        doc_name=source.get("doc_name", ""),
        source_path=source.get("source_path", ""),
        origin_pdf_path=source.get("origin_pdf_path", ""),
        page_idx=source.get("page_idx", ""),
        image_path=source.get("image_path", ""),
        level1=source.get("level1", ""),
        level2=source.get("level2", ""),
        evidence_text=source.get("evidence_text", ""),
    )

    content: Any = user_text
    if include_image:
        data_uri = image_to_data_uri(str(source.get("image_path") or row.get("query_image_path") or ""), max_image_bytes)
        if data_uri:
            content = [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ]

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def post_chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    timeout: int,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def parse_verdict(text: str) -> Dict[str, Any]:
    loaded = json.loads(text)
    return {
        "supported": bool(loaded.get("supported", False)),
        "answer_correct": bool(loaded.get("answer_correct", False)),
        "distractors_plausible_but_wrong": bool(loaded.get("distractors_plausible_but_wrong", False)),
        "ambiguous": bool(loaded.get("ambiguous", True)),
        "leakage": bool(loaded.get("leakage", True)),
        "needs_pdf_page": bool(loaded.get("needs_pdf_page", False)),
        "confidence": float(loaded.get("confidence", 0.0) or 0.0),
        "reason": str(loaded.get("reason", ""))[:200],
    }


def keep_verdict(verdict: Dict[str, Any], min_confidence: float) -> bool:
    return (
        verdict.get("supported") is True
        and verdict.get("answer_correct") is True
        and verdict.get("distractors_plausible_but_wrong") is True
        and verdict.get("ambiguous") is False
        and verdict.get("leakage") is False
        and float(verdict.get("confidence", 0.0)) >= min_confidence
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--accepted-jsonl", required=True)
    parser.add_argument("--rejected-jsonl", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--include-image", action="store_true")
    parser.add_argument("--max-image-bytes", type=int, default=3_500_000)
    parser.add_argument("--min-confidence", type=float, default=0.70)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key env: {args.api_key_env}")

    accepted_path = Path(args.accepted_jsonl)
    rejected_path = Path(args.rejected_jsonl)
    accepted_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {"seen": 0, "accepted": 0, "rejected": 0, "errors": 0}
    with accepted_path.open("w", encoding="utf-8") as accepted, rejected_path.open("w", encoding="utf-8") as rejected:
        for row in iter_jsonl(Path(args.input_jsonl)):
            if args.limit and stats["seen"] >= args.limit:
                break
            stats["seen"] += 1
            messages = build_messages(row, args.include_image, args.max_image_bytes)
            error = ""
            verdict: Dict[str, Any] = {}
            for attempt in range(1, args.max_retries + 1):
                try:
                    content = post_chat_completion(
                        base_url=args.base_url,
                        api_key=api_key,
                        model=args.model,
                        messages=messages,
                        timeout=args.timeout,
                    )
                    verdict = parse_verdict(content)
                    break
                except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, ValueError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    if attempt < args.max_retries:
                        time.sleep(min(2 ** attempt, 10))
            row["verification"] = {
                "status": "accepted" if verdict and keep_verdict(verdict, args.min_confidence) else "rejected",
                "verifier_model": args.model,
                "verdict": verdict,
                "error": error,
            }
            row.setdefault("provenance", {})["api_verified"] = row["verification"]["status"] == "accepted"
            if row["verification"]["status"] == "accepted":
                stats["accepted"] += 1
                write_jsonl_row(accepted, row)
            else:
                stats["rejected"] += 1
                if error:
                    stats["errors"] += 1
                write_jsonl_row(rejected, row)
            if args.sleep:
                time.sleep(args.sleep)
            if stats["seen"] % 50 == 0:
                print(json.dumps(stats, ensure_ascii=False), flush=True)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
