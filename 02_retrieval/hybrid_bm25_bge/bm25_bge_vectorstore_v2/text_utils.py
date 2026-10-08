import hashlib
import json
import os
import re
from typing import List, Tuple

_CH_RE = re.compile(r"[\u4e00-\u9fa5]")
_EN_RE = re.compile(r"[A-Za-z]")


def clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\u3000", " ").replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def detect_lang(text: str) -> str:
    if not text:
        return "mix"
    ch = len(_CH_RE.findall(text))
    en = len(_EN_RE.findall(text))
    total = max(1, ch + en)
    ch_ratio = ch / total
    en_ratio = en / total
    if ch_ratio >= 0.7:
        return "zh"
    if en_ratio >= 0.6:
        return "en"
    return "mix"


def sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def read_json_lines(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                try:
                    # 兼容单行 json 文件
                    obj = json.loads(line)
                    yield obj
                except Exception:
                    continue


def iter_json_records(file_path: str):
    # 支持 .jsonl（多行）或 .json（一行一个对象）
    if file_path.endswith(".jsonl"):
        yield from read_json_lines(file_path)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            txt = f.read().strip()
        try:
            data = json.loads(txt)
            if isinstance(data, list):
                for obj in data:
                    yield obj
            elif isinstance(data, dict):
                yield data
            else:
                # 逐行尝试
                for line in txt.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except Exception:
                        pass
        except Exception:
            # 回退到逐行
            for line in txt.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    pass