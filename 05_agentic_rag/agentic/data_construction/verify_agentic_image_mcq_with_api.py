#!/usr/bin/env python3
"""Verify image-query MCQs with an OpenAI-compatible multimodal API."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import tempfile
from io import BytesIO
import time
import urllib.error
import urllib.request

import requests
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, Dict, Iterable, List

import fitz  # PyMuPDF
from PIL import Image


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_CONFIG_JSON = "/mnt/data_1/yds/多模态/agentic/data_construction/api_config.local.json"

SYSTEM_PROMPT = """你是医学多模态选择题数据质检员。
给定一道低信息图像选择题、query image、候选答案、图文描述和相关 PDF 两页上下文，判断给定正确答案是否被图像与来源证据支持。

注意：
- 题目本身不应依赖证据描述，模型作答时只能看到题目、选项和图像。
- 来源证据只用于你核验数据质量。
- 不要重新创造答案，只核验给定答案是否可靠。
- 如果图像/证据不足、标签过粗、部位不一致、多个选项都可能对，应判为不通过。
- 只输出 JSON。"""


USER_PROMPT_TEMPLATE = """请核验下面的图像选择题是否适合作为训练样本。

问题：{question}
选项：{options}
给定正确答案：{answer}. {answer_text}
query_type: {query_type}

来源元信息：
- doc_id: {doc_id}
- doc_name: {doc_name}
- image_path: {image_path}
- origin_pdf_path: {origin_pdf_path}
- page_idx: {page_idx}
- context_page_indices: {context_page_indices}
- level1: {level1}
- level2: {level2}

图文描述/图注候选：
{evidence_text}

相关 PDF 两页文本：
{pdf_context_text}

请严格输出 JSON：
{{
  "supported": true/false,
  "answer_correct": true/false,
  "image_sufficient_for_question": true/false,
  "distractors_plausible_but_wrong": true/false,
  "ambiguous": true/false,
  "label_granularity_ok": true/false,
  "leakage": true/false,
  "confidence": 0.0-1.0,
  "reason": "不超过100字"
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


def file_to_data_uri(path: str, max_bytes: int) -> str:
    if not path:
        return ""
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return ""
    try:
        with Image.open(file_path) as image:
            rgb = image.convert("RGB")
            if max(rgb.size) > 1536:
                rgb.thumbnail((1536, 1536), Image.LANCZOS)
            buffer = BytesIO()
            rgb.save(buffer, format="JPEG", quality=90, optimize=True)
            data_bytes = buffer.getvalue()
    except Exception:
        if file_path.stat().st_size > max_bytes:
            return ""
        data_bytes = file_path.read_bytes()
    if len(data_bytes) > max_bytes:
        return ""
    data = base64.b64encode(data_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{data}"


def render_pdf_pages(pdf_path: str, page_indices: List[int], dpi: int, max_pages: int) -> List[str]:
    if not pdf_path or not page_indices:
        return []
    path = Path(pdf_path)
    if not path.exists():
        return []
    rendered: List[str] = []
    doc = fitz.open(str(path))
    try:
        for page_idx in page_indices[:max_pages]:
            if page_idx < 0 or page_idx >= doc.page_count:
                continue
            page = doc.load_page(page_idx)
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_p{page_idx}.jpg")
            tmp.close()
            pix.save(tmp.name)
            rendered.append(tmp.name)
    finally:
        doc.close()
    return rendered


def build_messages(row: Dict[str, Any], args: argparse.Namespace) -> tuple[List[Dict[str, Any]], List[str]]:
    source = row.get("source") or {}
    user_text = USER_PROMPT_TEMPLATE.format(
        question=row.get("question", ""),
        options=json.dumps(row.get("options", {}), ensure_ascii=False),
        answer=row.get("answer", ""),
        answer_text=row.get("answer_text", ""),
        query_type=row.get("query_type", ""),
        doc_id=source.get("doc_id", ""),
        doc_name=source.get("doc_name", ""),
        image_path=source.get("image_path", ""),
        origin_pdf_path=source.get("origin_pdf_path", ""),
        page_idx=source.get("page_idx", ""),
        context_page_indices=source.get("context_page_indices", []),
        level1=source.get("level1", ""),
        level2=source.get("level2", ""),
        evidence_text=source.get("evidence_text", ""),
        pdf_context_text=source.get("pdf_context_text", ""),
    )

    content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
    query_image_uri = file_to_data_uri(str(row.get("query_image_path") or source.get("image_path") or ""), args.max_image_bytes)
    if query_image_uri:
        content.append({"type": "image_url", "image_url": {"url": query_image_uri}})

    temp_files: List[str] = []
    if args.include_pdf_page_images:
        page_indices = source.get("context_page_indices") or []
        rendered = render_pdf_pages(
            str(source.get("origin_pdf_path") or ""),
            [int(x) for x in page_indices],
            dpi=args.pdf_dpi,
            max_pages=args.max_pdf_pages,
        )
        temp_files.extend(rendered)
        for page_image in rendered:
            page_uri = file_to_data_uri(page_image, args.max_pdf_page_image_bytes)
            if page_uri:
                content.append({"type": "image_url", "image_url": {"url": page_uri}})

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ], temp_files


def post_chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    timeout: int,
) -> str:
    def candidate_chat_urls(value: str) -> List[str]:
        normalized = (value or "").strip().rstrip("/")
        if not normalized:
            return []
        if normalized.endswith("/chat/completions"):
            return [normalized]
        parsed = urlparse(normalized)
        path = parsed.path.rstrip("/")
        urls = [f"{normalized}/chat/completions"]
        if not path.endswith("/v1"):
            urls.append(f"{normalized}/v1/chat/completions")
        deduped: List[str] = []
        seen = set()
        for url in urls:
            if url in seen:
                continue
            seen.add(url)
            deduped.append(url)
        return deduped

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    fallback_payload = dict(payload)
    fallback_payload.pop("response_format", None)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    session = requests.Session()
    session.trust_env = False
    last_error: Exception | None = None
    for request_payload in (payload, fallback_payload):
        for url in candidate_chat_urls(base_url):
            try:
                response = session.post(url, json=request_payload, headers=headers, timeout=timeout)
                if response.status_code >= 400:
                    raise RuntimeError(f"HTTP {response.status_code} for {url}: {response.text[:1000]}")
                data = response.json()
                message = data["choices"][0]["message"]
                return str(message.get("content") or message.get("reasoning_content") or "")
            except Exception as exc:  # noqa: BLE001
                last_error = exc
    assert last_error is not None
    raise last_error


def parse_verdict(text: str) -> Dict[str, Any]:
    loaded = json.loads(text)
    return {
        "supported": bool(loaded.get("supported", False)),
        "answer_correct": bool(loaded.get("answer_correct", False)),
        "image_sufficient_for_question": bool(loaded.get("image_sufficient_for_question", False)),
        "distractors_plausible_but_wrong": bool(loaded.get("distractors_plausible_but_wrong", False)),
        "ambiguous": bool(loaded.get("ambiguous", True)),
        "label_granularity_ok": bool(loaded.get("label_granularity_ok", False)),
        "leakage": bool(loaded.get("leakage", True)),
        "confidence": float(loaded.get("confidence", 0.0) or 0.0),
        "reason": str(loaded.get("reason", ""))[:240],
    }


def keep_verdict(verdict: Dict[str, Any], min_confidence: float) -> bool:
    return (
        verdict.get("supported") is True
        and verdict.get("answer_correct") is True
        and verdict.get("image_sufficient_for_question") is True
        and verdict.get("distractors_plausible_but_wrong") is True
        and verdict.get("ambiguous") is False
        and verdict.get("label_granularity_ok") is True
        and verdict.get("leakage") is False
        and float(verdict.get("confidence", 0.0)) >= min_confidence
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-json",
        default=DEFAULT_CONFIG_JSON,
        help="Optional local JSON config containing api_key/base_url/model and verifier options.",
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--accepted-jsonl", required=True)
    parser.add_argument("--rejected-jsonl", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--include-pdf-page-images", action="store_true")
    parser.add_argument("--pdf-dpi", type=int, default=120)
    parser.add_argument("--max-pdf-pages", type=int, default=2)
    parser.add_argument("--max-image-bytes", type=int, default=4_000_000)
    parser.add_argument("--max-pdf-page-image-bytes", type=int, default=4_000_000)
    parser.add_argument("--min-confidence", type=float, default=0.70)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=3)
    return parser.parse_args()



def load_config(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        return {}
    try:
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid config JSON: {config_path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise SystemExit(f"Config JSON must be an object: {config_path}")
    return loaded


def apply_config(args: argparse.Namespace, config: Dict[str, Any]) -> argparse.Namespace:
    # Config fills defaults; explicit command-line values win for these fields.
    string_defaults = {"model": "", "base_url": "", "api_key": ""}
    for key, value in config.items():
        if not hasattr(args, key):
            continue
        current = getattr(args, key)
        if key in string_defaults and current != string_defaults[key]:
            continue
        setattr(args, key, value)
    return args


def main() -> None:
    args = parse_args()
    args = apply_config(args, load_config(args.config_json))
    args.base_url = args.base_url or os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL)
    if not args.model:
        raise SystemExit("Missing model. Set --model or config field `model`.")
    api_key = args.api_key or os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key. Set config field `api_key`, --api-key, or env {args.api_key_env}.")

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
            messages, temp_files = build_messages(row, args)
            error = ""
            verdict: Dict[str, Any] = {}
            try:
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
            finally:
                for temp_file in temp_files:
                    try:
                        Path(temp_file).unlink(missing_ok=True)
                    except Exception:
                        pass

            accepted_flag = bool(verdict) and keep_verdict(verdict, args.min_confidence)
            row["verification"] = {
                "status": "accepted" if accepted_flag else "rejected",
                "verifier_model": args.model,
                "verdict": verdict,
                "error": error,
            }
            row.setdefault("provenance", {})["api_verified"] = accepted_flag
            if accepted_flag:
                stats["accepted"] += 1
                write_jsonl_row(accepted, row)
            else:
                stats["rejected"] += 1
                if error:
                    stats["errors"] += 1
                write_jsonl_row(rejected, row)
            if args.sleep:
                time.sleep(args.sleep)
            if stats["seen"] % 20 == 0:
                print(json.dumps(stats, ensure_ascii=False), flush=True)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
