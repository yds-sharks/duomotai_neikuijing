#!/usr/bin/env python3
"""Shared helpers for Stage-1 API caller scripts (generation + verification).

Provides: config loading, image-attached multimodal message building, robust JSON
extraction, API usage extraction, resume support, and a lock-guarded streaming
JSONL writer (代码三件套: timing / streaming append+fsync / resume).
"""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from api_client import ChatResult, OpenAICompatibleClient, encode_image_path_to_data_url


def load_api_config(path: str) -> Dict[str, Any]:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    if not cfg.get("api_key") or cfg["api_key"] == "PASTE_YOUR_KEY_HERE":
        raise RuntimeError(f"api_config {path} has no real api_key; copy api_config.example.json and fill it.")
    if not cfg.get("base_url"):
        raise RuntimeError(f"api_config {path} missing base_url.")
    if not cfg.get("model"):
        raise RuntimeError(f"api_config {path} missing model.")
    return cfg


def build_client(cfg: Dict[str, Any]) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        timeout_s=int(cfg.get("timeout", 180)),
        max_retries=int(cfg.get("max_retries", 3)),
    )


def build_image_messages(system_prompt: str, user_text: str, image_path: str) -> List[Dict[str, Any]]:
    """Build chat messages with the query image attached as pixels (data URL).

    Passing image pixels is mandatory: without it 'image dependency' and verifier
    independence both collapse into text-only judgements.
    """
    data_url = encode_image_path_to_data_url(Path(image_path))
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_content(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON extraction from a model reply (tolerates code fences/prose)."""
    if not text:
        return None
    stripped = text.strip()
    for candidate in (stripped, _strip_code_fence(stripped)):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    match = _JSON_OBJ_RE.search(stripped)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None


def _strip_code_fence(text: str) -> str:
    if text.startswith("```"):
        body = text.split("```", 2)
        if len(body) >= 2:
            inner = body[1]
            if inner.lstrip().lower().startswith("json"):
                inner = inner.lstrip()[4:]
            return inner.strip()
    return text


def extract_usage(result: ChatResult) -> Dict[str, int]:
    usage = (result.raw_response or {}).get("usage") or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
    }


def accumulate_usage(total: Dict[str, int], one: Dict[str, int]) -> None:
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        total[key] = total.get(key, 0) + int(one.get(key, 0))


def load_done_ids(path: Path, id_key: str) -> Set[str]:
    """Collect ids already present in an output JSONL so we can --resume."""
    done: Set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            value = row.get(id_key)
            if value:
                done.add(str(value))
    return done


class StreamWriter:
    """Append-mode JSONL writer with a lock and fsync per record.

    Append + resume is the crash-recovery model here: a killed run leaves valid
    lines on disk; the next run skips ids already written.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._handle = self.path.open("a", encoding="utf-8")

    def write(self, row: Dict[str, Any]) -> None:
        line = json.dumps(row, ensure_ascii=False)
        with self._lock:
            self._handle.write(line + "\n")
            self._handle.flush()
            os.fsync(self._handle.fileno())

    def close(self) -> None:
        with self._lock:
            self._handle.close()
