#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export PDF provenance mappings for the cleaned assetpack JSONL.

This script resolves:
    jsonl row -> doc_out_dir -> run_cmd.json -> pdf_path

Outputs:
1. Row-level JSONL mapping for every input line.
2. Doc-level JSONL mapping with one row per unique doc_out_dir.
3. Summary JSON with counts and missing-path diagnostics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, TextIO


DEFAULT_INPUT = (
    "/mnt/data_1/yds/多模态/data_house/"
    "test_ingest_188w_20260508_bgem3/assetpack_resume552_20260508.text_only_denoised.jsonl"
)
DEFAULT_ROW_OUTPUT = (
    "/mnt/data_1/yds/多模态/data_house/"
    "test_ingest_188w_20260508_bgem3/assetpack_resume552_20260508.text_only_denoised.pdf_mapping.jsonl"
)
DEFAULT_DOC_OUTPUT = (
    "/mnt/data_1/yds/多模态/data_house/"
    "test_ingest_188w_20260508_bgem3/assetpack_resume552_20260508.text_only_denoised.doc_pdf_mapping.jsonl"
)
DEFAULT_SUMMARY_OUTPUT = (
    "/mnt/data_1/yds/多模态/data_house/"
    "test_ingest_188w_20260508_bgem3/assetpack_resume552_20260508.text_only_denoised.pdf_mapping.summary.json"
)


def extract_doc_id(doc_out_dir: str) -> str:
    name = Path(doc_out_dir).name
    if "__" in name:
        return name.split("__")[-1][:32]
    return name[:32]


def extract_doc_name(doc_out_dir: str) -> str:
    return Path(doc_out_dir).name[:256]


def iter_jsonl(path: Path) -> Iterable[tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def load_run_cmd_mapping(doc_out_dir: str) -> Dict[str, Any]:
    doc_dir = Path(doc_out_dir)
    run_cmd_path = doc_dir / "run_cmd.json"
    base: Dict[str, Any] = {
        "doc_out_dir": doc_out_dir,
        "doc_id": extract_doc_id(doc_out_dir),
        "doc_name": extract_doc_name(doc_out_dir),
        "run_cmd_path": str(run_cmd_path),
        "pdf_path": "",
        "pdf_name": "",
        "pdf_exists": False,
        "run_cmd_exists": run_cmd_path.exists(),
        "resolved_lang": "",
        "mineru_out_dir": "",
        "status": "missing_run_cmd",
    }

    if not run_cmd_path.exists():
        return base

    try:
        meta = json.loads(run_cmd_path.read_text(encoding="utf-8"))
    except Exception as exc:
        base["status"] = f"bad_run_cmd_json:{type(exc).__name__}"
        return base

    pdf_path = str(meta.get("pdf_path") or "")
    pdf_exists = bool(pdf_path) and Path(pdf_path).exists()
    base.update(
        {
            "pdf_path": pdf_path,
            "pdf_name": Path(pdf_path).name if pdf_path else "",
            "pdf_exists": pdf_exists,
            "resolved_lang": str(meta.get("resolved_lang") or ""),
            "mineru_out_dir": str(meta.get("out_dir") or ""),
            "status": "ok" if pdf_exists else "missing_pdf",
        }
    )
    return base


def write_jsonl_row(fp: TextIO, row: Dict[str, Any]) -> None:
    fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_row_mapping(
    line_no: int,
    item: Dict[str, Any],
    doc_meta: Dict[str, Any],
    input_jsonl: str,
) -> Dict[str, Any]:
    page_idx = item.get("page_idx")
    page_idx_int = int(page_idx) if page_idx is not None else -1
    return {
        "line_no": line_no,
        "input_jsonl": input_jsonl,
        "doc_id": doc_meta["doc_id"],
        "doc_name": doc_meta["doc_name"],
        "doc_out_dir": doc_meta["doc_out_dir"],
        "doc_parse_dir": str(item.get("doc_parse_dir") or ""),
        "content_list_path": str(item.get("content_list_path") or ""),
        "pdf_path": doc_meta["pdf_path"],
        "pdf_name": doc_meta["pdf_name"],
        "pdf_exists": doc_meta["pdf_exists"],
        "run_cmd_path": doc_meta["run_cmd_path"],
        "run_cmd_exists": doc_meta["run_cmd_exists"],
        "mapping_status": doc_meta["status"],
        "page_idx": page_idx_int,
        "page_num_1based": page_idx_int + 1 if page_idx_int >= 0 else None,
        "block_type": str(item.get("type") or ""),
        "mineru_type": str(item.get("mineru_type") or ""),
        "content_hash": str(item.get("content_hash") or ""),
        "text_len": len(str(item.get("text") or "")),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Export PDF provenance mapping for assetpack JSONL")
    ap.add_argument("--input-jsonl", default=DEFAULT_INPUT)
    ap.add_argument("--row-output", default=DEFAULT_ROW_OUTPUT)
    ap.add_argument("--doc-output", default=DEFAULT_DOC_OUTPUT)
    ap.add_argument("--summary-output", default=DEFAULT_SUMMARY_OUTPUT)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--progress-every", type=int, default=50000)
    args = ap.parse_args()

    input_path = Path(args.input_jsonl)
    row_output = Path(args.row_output)
    doc_output = Path(args.doc_output)
    summary_output = Path(args.summary_output)

    row_output.parent.mkdir(parents=True, exist_ok=True)
    doc_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)

    doc_cache: Dict[str, Dict[str, Any]] = {}
    stats = {
        "input_jsonl": str(input_path),
        "row_output": str(row_output),
        "doc_output": str(doc_output),
        "summary_output": str(summary_output),
        "rows_written": 0,
        "docs_written": 0,
        "unique_docs_seen": 0,
        "missing_doc_out_dir_rows": 0,
        "missing_run_cmd_docs": 0,
        "missing_pdf_docs": 0,
        "ok_docs": 0,
    }

    with input_path.open("r", encoding="utf-8") as src, \
        row_output.open("w", encoding="utf-8") as row_fp, \
        doc_output.open("w", encoding="utf-8") as doc_fp:

        for line_no, raw_line in enumerate(src, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            item = json.loads(raw_line)
            doc_out_dir = str(item.get("doc_out_dir") or "")
            if not doc_out_dir:
                stats["missing_doc_out_dir_rows"] += 1
                continue

            if doc_out_dir not in doc_cache:
                doc_meta = load_run_cmd_mapping(doc_out_dir)
                doc_cache[doc_out_dir] = doc_meta
                write_jsonl_row(doc_fp, doc_meta)
                stats["docs_written"] += 1
                if doc_meta["status"] == "ok":
                    stats["ok_docs"] += 1
                elif doc_meta["status"] == "missing_run_cmd":
                    stats["missing_run_cmd_docs"] += 1
                else:
                    stats["missing_pdf_docs"] += 1
            else:
                doc_meta = doc_cache[doc_out_dir]

            row = build_row_mapping(
                line_no=line_no,
                item=item,
                doc_meta=doc_meta,
                input_jsonl=str(input_path),
            )
            write_jsonl_row(row_fp, row)
            stats["rows_written"] += 1

            if args.progress_every > 0 and stats["rows_written"] % int(args.progress_every) == 0:
                print(
                    f"[progress] rows={stats['rows_written']} docs={stats['docs_written']} "
                    f"ok_docs={stats['ok_docs']} missing_run_cmd_docs={stats['missing_run_cmd_docs']} "
                    f"missing_pdf_docs={stats['missing_pdf_docs']}",
                    flush=True,
                )

            if args.max_rows > 0 and stats["rows_written"] >= int(args.max_rows):
                break

    stats["unique_docs_seen"] = len(doc_cache)
    summary_output.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"[done] rows={stats['rows_written']} docs={stats['docs_written']} "
        f"ok_docs={stats['ok_docs']} row_output={row_output} doc_output={doc_output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
