#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: run_pipeline.py
One-command runner for:
  Stage1: MinerU parse PDFs (1_mineu_pdf.py) with resume-by-skip
  Stage2: Build AssetPack JSONL (2_jsonl.py) with atomic output

Resume strategy:
  - Stage1: each PDF has its own out_dir; rerun skips directories already containing *_content_list.json
  - Stage2: writes to a temp file, then atomically replaces final jsonl on success
"""

import argparse
import subprocess
from pathlib import Path
import time
import os

def run(cmd, env=None):
    print("\n[CMD]", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, env=env)
    if p.returncode != 0:
        raise SystemExit(p.returncode)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_root", required=True)

    # Stage1 options (passed through)
    ap.add_argument("--backend", default="pipeline")
    ap.add_argument("--method", default="auto")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--vram_gb", type=int, default=20)
    ap.add_argument("--model_source", default="modelscope")
    ap.add_argument("--mineru_bin", default="")
    ap.add_argument("--enable_formula", action="store_true", default=True)
    ap.add_argument("--enable_table", action="store_true", default=True)
    ap.add_argument("--timeout_sec", type=int, default=0)
    ap.add_argument("--force", action="store_true", default=False)

    # Stage2 outputs
    ap.add_argument("--assetpack_jsonl", default="")
    ap.add_argument("--assetpack_manifest", default="")

    args = ap.parse_args()

    input_dir = Path(args.input_dir).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    stage1 = Path("/mnt/data_1/yds/多模态/code/mineu/1_mineu_pdf.py")
    stage2 = Path("/mnt/data_1/yds/多模态/code/mineu/2_jsonl.py")

    assetpack_jsonl = Path(args.assetpack_jsonl).resolve() if args.assetpack_jsonl else (out_root / "assetpack.jsonl")
    assetpack_manifest = Path(args.assetpack_manifest).resolve() if args.assetpack_manifest else (out_root / "assetpack_manifest.json")

    # ---------- Stage 1 ----------
    cmd1 = [
        "python", str(stage1),
        "--input_dir", str(input_dir),
        "--out_root", str(out_root),
        "--backend", args.backend,
        "--method", args.method,
        "--device", args.device,
        "--vram_gb", str(args.vram_gb),
        "--model_source", args.model_source,
    ]
    if args.mineru_bin:
        cmd1 += ["--mineru_bin", args.mineru_bin]
    if args.enable_formula:
        cmd1.append("--enable_formula")
    if args.enable_table:
        cmd1.append("--enable_table")
    if args.timeout_sec and args.timeout_sec > 0:
        cmd1 += ["--timeout_sec", str(args.timeout_sec)]
    if args.force:
        cmd1.append("--force")

    run(cmd1, env=os.environ.copy())

    # ---------- Stage 2 (atomic write) ----------
    tmp_jsonl = assetpack_jsonl.with_suffix(".jsonl.tmp")
    tmp_manifest = assetpack_manifest.with_suffix(".json.tmp")

    cmd2 = [
        "python", str(stage2),
        "--in_root", str(out_root),
        "--out_jsonl", str(tmp_jsonl),
        "--emit_manifest", str(tmp_manifest),
        "--drop_discarded",
        "--keep_unknown_raw",
    ]
    run(cmd2, env=os.environ.copy())

    # atomic replace
    tmp_jsonl.replace(assetpack_jsonl)
    tmp_manifest.replace(assetpack_manifest)

    print("\n========== PIPELINE DONE ==========")
    print("Stage1 out_root:", out_root)
    print("AssetPack JSONL :", assetpack_jsonl)
    print("Manifest        :", assetpack_manifest)

if __name__ == "__main__":
    main()
