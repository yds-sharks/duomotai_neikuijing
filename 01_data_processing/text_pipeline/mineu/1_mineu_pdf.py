#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: 1_mineu_pdf.py
Purpose:
  Batch parse PDFs with MinerU CLI, with automatic language routing.

Key fixes:
  - MinerU 2.6.7 does NOT accept --lang auto.
  - This script accepts --lang auto for user convenience, but will auto-map to a valid MinerU lang per PDF.
  - Auto-detect primarily distinguishes Chinese vs English; includes basic JP/KR hints.
  - If MinerU still fails due to lang invalid, auto fallback retry (ch -> en).

Example:
  python 1_mineu_pdf.py \
    --input_dir /mnt/data_1/yds/多模态/data/test \
    --out_root /mnt/data_1/yds/多模态/data/output/test \
    --backend pipeline --method auto \
    --device cuda:0 --vram_gb 20 --model_source modelscope \
    --enable_formula --enable_table

Recommended:
  Do NOT pass --lang, let it auto.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# =========================
# Module: mineru_lang_router
# =========================

# MinerU 2.6.7 accepted languages (from your stderr)
MINERU_ALLOWED_LANGS = {
    "ch", "ch_server", "ch_lite", "en",
    "korean", "japan", "chinese_cht",
    "ta", "te", "ka", "th", "el", "latin", "arabic",
    "east_slavic", "cyrillic", "devanagari",
}

# Heuristic language groups
_RE_CJK = re.compile(r"[\u4e00-\u9fff]")
_RE_JP = re.compile(r"[\u3040-\u30ff]")          # Hiragana/Katakana
_RE_KR = re.compile(r"[\uac00-\ud7af]")          # Hangul


def _safe_ratio(num: int, den: int) -> float:
    return num / den if den > 0 else 0.0


def extract_pdf_text_sample(pdf_path: Path, max_pages: int = 2, max_chars: int = 6000) -> str:
    """
    Try to extract text from the first N pages using pypdf.
    If PDF is scanned or extraction fails, returns empty string.
    """
    try:
        from pypdf import PdfReader  # pip install pypdf
        reader = PdfReader(str(pdf_path))
        parts: List[str] = []
        n = min(max_pages, len(reader.pages))
        for i in range(n):
            t = reader.pages[i].extract_text() or ""
            t = t.strip()
            if t:
                parts.append(t)
            if sum(len(x) for x in parts) >= max_chars:
                break
        sample = "\n".join(parts)
        return sample[:max_chars].strip()
    except Exception:
        return ""


def detect_lang_for_mineru(pdf_path: Path) -> str:
    """
    Decide MinerU lang for this PDF.

    Strategy (practical & stable):
    - If extractable text exists:
        - If contains Japanese kana -> 'japan'
        - Else if contains Hangul -> 'korean'
        - Else if CJK ratio >= 1% -> 'ch'
        - Else -> 'en'
    - If no extractable text (scanned / image PDF): default 'ch' (more tolerant).
    """
    sample = extract_pdf_text_sample(pdf_path, max_pages=2, max_chars=6000)
    if not sample:
        return "ch"

    total = len(sample)
    cjk = len(_RE_CJK.findall(sample))
    jp = len(_RE_JP.findall(sample))
    kr = len(_RE_KR.findall(sample))

    if jp >= 5:
        return "japan"
    if kr >= 5:
        return "korean"

    cjk_ratio = _safe_ratio(cjk, total)
    # 1% is a conservative threshold; mixed EN+CN papers usually exceed this if truly Chinese.
    return "ch" if cjk_ratio >= 0.01 else "en"


def normalize_lang(requested_lang: str, pdf_path: Path) -> str:
    """
    Convert user requested --lang to an actual MinerU supported lang.
    - If requested_lang == 'auto': detect per PDF.
    - Else: if invalid, fallback to 'ch' or 'en' based on detection.
    """
    lang = (requested_lang or "auto").strip().lower()
    if lang == "auto":
        lang = detect_lang_for_mineru(pdf_path)

    if lang not in MINERU_ALLOWED_LANGS:
        # If user passed something unsupported, fallback to a detected one
        det = detect_lang_for_mineru(pdf_path)
        lang = det if det in MINERU_ALLOWED_LANGS else "ch"

    return lang


# =========================
# Module: batch_runner_utils
# =========================

def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def safe_stem(name: str) -> str:
    s = Path(name).stem
    s = re.sub(r"[^\w\-]+", "_", s, flags=re.UNICODE).strip("_")
    return s or "pdf"


def find_pdfs(input_dir: Path) -> List[Path]:
    pdfs = sorted([p for p in input_dir.rglob("*.pdf") if p.is_file()])
    return pdfs


def load_pdf_list(input_list: Path) -> List[Path]:
    pdfs: List[Path] = []
    for line in input_list.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        p = Path(line).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"PDF listed in input_list not found: {p}")
        pdfs.append(p)
    return pdfs


def has_mineru_success(out_dir: Path) -> bool:
    if not out_dir.exists():
        return False
    # MinerU output should include *_content_list.json
    for f in out_dir.rglob("*_content_list.json"):
        if f.is_file() and f.stat().st_size > 0:
            return True
    return False


def resolve_mineru_executable(user_specified: str = "") -> str:
    """
    Resolve a runnable MinerU CLI path without relying on the caller's PATH.
    """
    candidates: List[Path] = []

    if user_specified:
        candidates.append(Path(user_specified).expanduser())

    env_bin = os.environ.get("MINERU_BIN", "").strip()
    if env_bin:
        candidates.append(Path(env_bin).expanduser())

    which_bin = shutil.which("mineru")
    if which_bin:
        candidates.append(Path(which_bin))

    # Prefer the console script that belongs to the Python interpreter running this wrapper.
    py_bin = Path(sys.executable).resolve()
    candidates.append(py_bin.parent / "mineru")

    conda_prefix = os.environ.get("CONDA_PREFIX", "").strip()
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "bin" / "mineru")

    # Historical runs in this workspace used the anyRAG conda env.
    candidates.append(Path("/home/yds/miniconda3/envs/anyRAG/bin/mineru"))

    checked: List[str] = []
    seen = set()
    for cand in candidates:
        cand = cand.resolve()
        cand_str = str(cand)
        if cand_str in seen:
            continue
        seen.add(cand_str)
        checked.append(cand_str)
        if cand.is_file() and os.access(cand, os.X_OK):
            return cand_str

    raise FileNotFoundError(
        "No runnable MinerU CLI found. Checked: " + "; ".join(checked)
    )


def build_cmd(
    mineru_bin: str,
    pdf_path: Path,
    out_dir: Path,
    backend: str,
    method: str,
    lang: str,
    enable_formula: bool,
    enable_table: bool,
) -> List[str]:
    cmd = [
        mineru_bin,
        "-p", str(pdf_path),
        "-o", str(out_dir),
        "-b", backend,
        "-m", method,
        "-l", lang,
        "-f", "true" if enable_formula else "false",
        "-t", "true" if enable_table else "false",
    ]
    return cmd


def run_cmd_to_logs(
    cmd: List[str],
    out_dir: Path,
    env: Dict[str, str],
    timeout_sec: int,
) -> Tuple[int, str]:
    """
    Run cmd, write stdout/stderr to files, return (returncode, msg).
    """
    log_out = out_dir / "mineru_stdout.log"
    log_err = out_dir / "mineru_stderr.log"
    try:
        with log_out.open("w", encoding="utf-8") as fo, log_err.open("w", encoding="utf-8") as fe:
            proc = subprocess.run(
                cmd,
                stdout=fo,
                stderr=fe,
                env=env,
                timeout=timeout_sec if timeout_sec > 0 else None,
                check=False,
            )
        return proc.returncode, "OK"
    except subprocess.TimeoutExpired:
        return 124, f"TIMEOUT({timeout_sec}s)"
    except FileNotFoundError:
        return 127, "mineru_not_found_in_PATH"
    except Exception as e:
        return 1, f"EXCEPTION({type(e).__name__}: {e})"


def stderr_contains_invalid_lang(out_dir: Path) -> bool:
    err = out_dir / "mineru_stderr.log"
    if not err.exists():
        return False
    txt = err.read_text(encoding="utf-8", errors="ignore")
    return ("Invalid value for '-l' / '--lang'" in txt) or ("--lang" in txt and "Invalid value" in txt)


def run_one_pdf(
    pdf_path: Path,
    out_dir: Path,
    backend: str,
    method: str,
    mineru_bin: str,
    requested_lang: str,
    enable_formula: bool,
    enable_table: bool,
    env: Dict[str, str],
    timeout_sec: int,
) -> Tuple[bool, str]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve actual lang (no 'auto' reaches MinerU)
    resolved_lang = normalize_lang(requested_lang, pdf_path)

    cmd = build_cmd(mineru_bin, pdf_path, out_dir, backend, method, resolved_lang, enable_formula, enable_table)

    meta = {
        "pdf_path": str(pdf_path),
        "out_dir": str(out_dir),
        "mineru_bin": mineru_bin,
        "requested_lang": requested_lang,
        "resolved_lang": resolved_lang,
        "cmd": cmd,
        "env_subset": {
            "MINERU_DEVICE_MODE": env.get("MINERU_DEVICE_MODE"),
            "MINERU_VIRTUAL_VRAM_SIZE": env.get("MINERU_VIRTUAL_VRAM_SIZE"),
            "MINERU_MODEL_SOURCE": env.get("MINERU_MODEL_SOURCE"),
        },
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "run_cmd.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    rc, msg = run_cmd_to_logs(cmd, out_dir, env, timeout_sec)

    # Success condition
    if rc == 0 and has_mineru_success(out_dir):
        return True, "OK"

    # If failed due to invalid lang, retry with robust fallbacks (ch -> en)
    if rc != 0 and stderr_contains_invalid_lang(out_dir):
        for fallback_lang in ["ch", "en"]:
            if fallback_lang == resolved_lang:
                continue
            cmd2 = build_cmd(mineru_bin, pdf_path, out_dir, backend, method, fallback_lang, enable_formula, enable_table)
            meta["retry"] = {"fallback_lang": fallback_lang, "cmd": cmd2, "retry_at": time.strftime("%Y-%m-%d %H:%M:%S")}
            (out_dir / "run_cmd.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            rc2, _ = run_cmd_to_logs(cmd2, out_dir, env, timeout_sec)
            if rc2 == 0 and has_mineru_success(out_dir):
                return True, f"OK(retry_lang={fallback_lang})"
        return False, f"FAILED(returncode={rc}, invalid_lang)"

    if rc == 0:
        return False, "FAILED(returncode=0, missing_content_list)"

    return False, f"FAILED(returncode={rc}, reason={msg})"


# =========================
# Module: main_cli
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="", help="Folder containing PDFs (recursive).")
    ap.add_argument("--input_list", default="", help="Optional text file with one absolute PDF path per line.")
    ap.add_argument("--out_root", required=True, help="Output root folder for MinerU results.")
    ap.add_argument("--backend", default="pipeline",
                    choices=["pipeline", "vlm-transformers", "vlm-sglang-engine"], help="MinerU backend.")
    ap.add_argument("--method", default="auto", choices=["auto", "txt", "ocr"], help="Parse method.")

    # IMPORTANT: default is auto; user does NOT need to pass this.
    ap.add_argument("--lang", default="auto",
                    help=f"Language routing. Use 'auto' to detect per PDF. "
                         f"Or set one of: {sorted(MINERU_ALLOWED_LANGS)}")

    ap.add_argument("--device", default="cuda:0", help="MINERU_DEVICE_MODE, e.g., cuda:0 or cpu.")
    ap.add_argument("--vram_gb", type=int, default=20, help="MINERU_VIRTUAL_VRAM_SIZE (GB).")
    ap.add_argument("--model_source", default="modelscope", choices=["huggingface", "modelscope"], help="MINERU_MODEL_SOURCE.")
    ap.add_argument("--mineru_bin", default="", help="Optional absolute path to MinerU CLI. If omitted, auto-detect.")
    ap.add_argument("--enable_formula", action="store_true", help="Enable formula parsing (-f true).")
    ap.add_argument("--enable_table", action="store_true", help="Enable table parsing (-t true).")
    ap.add_argument("--force", action="store_true", help="Re-run even if output already exists.")
    ap.add_argument("--timeout_sec", type=int, default=0, help="Per-PDF timeout seconds (0 means no timeout).")
    ap.add_argument("--summary_path", default="", help="Optional custom batch summary jsonl path.")
    ap.add_argument("--failed_path", default="", help="Optional custom failed-list path.")
    args = ap.parse_args()

    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if not args.input_dir and not args.input_list:
        print("[ERROR] One of --input_dir or --input_list is required.")
        sys.exit(1)
    if args.input_dir and args.input_list:
        print("[ERROR] Use only one of --input_dir or --input_list.")
        sys.exit(1)

    if args.input_list:
        input_list = Path(args.input_list).expanduser().resolve()
        pdfs = load_pdf_list(input_list)
        input_desc = str(input_list)
    else:
        input_dir = Path(args.input_dir).expanduser().resolve()
        pdfs = find_pdfs(input_dir)
        input_desc = str(input_dir)

    if not pdfs:
        print(f"[ERROR] No PDFs found from: {input_desc}")
        sys.exit(1)

    # Prepare env
    env = os.environ.copy()
    env["MINERU_DEVICE_MODE"] = args.device
    env["MINERU_VIRTUAL_VRAM_SIZE"] = str(args.vram_gb)
    env["MINERU_MODEL_SOURCE"] = args.model_source

    try:
        mineru_bin = resolve_mineru_executable(args.mineru_bin)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(127)

    print(f"[INFO] Using MinerU CLI: {mineru_bin}")

    summary_path = Path(args.summary_path).expanduser().resolve() if args.summary_path else (out_root / "_batch_summary.jsonl")
    failed_path = Path(args.failed_path).expanduser().resolve() if args.failed_path else (out_root / "_failed_list.txt")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(pdfs)
    ok_cnt = 0
    skip_cnt = 0
    fail_cnt = 0

    for i, pdf in enumerate(pdfs, 1):
        stem = safe_stem(pdf.name)
        short = sha1_text(str(pdf))[:8]
        out_dir = out_root / f"{stem}__{short}"

        if (not args.force) and has_mineru_success(out_dir):
            rec = {
                "pdf": str(pdf),
                "out_dir": str(out_dir),
                "status": "SKIP_EXISTS",
                "index": i,
                "total": total,
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            summary_path.open("a", encoding="utf-8").write(json.dumps(rec, ensure_ascii=False) + "\n")
            skip_cnt += 1
            print(f"[{i}/{total}] SKIP (already parsed): {pdf}")
            continue

        print(f"[{i}/{total}] RUN : {pdf}")
        t0 = time.time()

        ok, msg = run_one_pdf(
            pdf_path=pdf,
            out_dir=out_dir,
            backend=args.backend,
            method=args.method,
            mineru_bin=mineru_bin,
            requested_lang=args.lang,          # may be 'auto'
            enable_formula=args.enable_formula,
            enable_table=args.enable_table,
            env=env,
            timeout_sec=args.timeout_sec,
        )
        dt = round(time.time() - t0, 3)

        # Try to read resolved lang from run_cmd.json for reporting
        resolved_lang = None
        run_cmd_path = out_dir / "run_cmd.json"
        if run_cmd_path.exists():
            try:
                resolved_lang = json.loads(run_cmd_path.read_text(encoding="utf-8")).get("resolved_lang")
            except Exception:
                resolved_lang = None

        rec = {
            "pdf": str(pdf),
            "out_dir": str(out_dir),
            "status": "OK" if ok else "FAIL",
            "msg": msg,
            "backend": args.backend,
            "method": args.method,
            "requested_lang": args.lang,
            "resolved_lang": resolved_lang,
            "device": args.device,
            "vram_gb": args.vram_gb,
            "model_source": args.model_source,
            "mineru_bin": mineru_bin,
            "enable_formula": args.enable_formula,
            "enable_table": args.enable_table,
            "seconds": dt,
            "index": i,
            "total": total,
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        summary_path.open("a", encoding="utf-8").write(json.dumps(rec, ensure_ascii=False) + "\n")

        if ok:
            ok_cnt += 1
            print(f"[{i}/{total}] OK  ({dt}s): {out_dir}")
        else:
            fail_cnt += 1
            failed_path.open("a", encoding="utf-8").write(str(pdf) + "\n")
            print(f"[{i}/{total}] FAIL({dt}s): {msg} -> {out_dir}")

    print("\n========== BATCH DONE ==========")
    print(f"Total: {total}, OK: {ok_cnt}, SKIP: {skip_cnt}, FAIL: {fail_cnt}")
    print(f"Summary: {summary_path}")
    if fail_cnt:
        print(f"Failed list: {failed_path}")


if __name__ == "__main__":
    main()
