#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: 2_jsonl.py
Purpose:
  Convert MinerU output dirs into a stable AssetPack JSONL intermediate format.

Key FIX (critical):
  - img_path/table img_path in MinerU content_list.json is RELATIVE TO the directory
    where *_content_list.json resides (usually .../auto/).
  - Therefore we define:
        parse_run_dir = content_list_path.parent
    and resolve:
        img_abs_path = parse_run_dir / img_rel_path
  - This fixes MISSING image paths.

Policy:
  - Drop 'discarded' blocks (noise).
  - Keep other unknown/new types (type='other'), optionally keeping raw_block for later auditing.
  - One JSONL line per block (except discarded which is dropped).

Usage:
  python 2_jsonl.py --in_root <mineru_out_root> --out_jsonl assetpack.jsonl --emit_manifest manifest.json \
    --drop_discarded --keep_unknown_raw
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List


# ==============================
# Module: io_utils
# ==============================

def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def join_lines(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, list):
        return "\n".join([str(i).strip() for i in x if str(i).strip()])
    return str(x).strip()


def find_content_list_files(dir_path: Path) -> List[Path]:
    return sorted([p for p in dir_path.rglob("*_content_list.json") if p.is_file()])


def find_middle_files(dir_path: Path) -> List[Path]:
    return sorted([p for p in dir_path.rglob("*_middle.json") if p.is_file()])


def iter_doc_out_dirs(in_root: Path) -> List[Path]:
    """
    Discover per-PDF output dirs (e.g., xxx__hash).
    Each such dir should contain at least one *_content_list.json in its subtree.
    """
    # If in_root itself already contains content_list, treat it as a single doc dir
    if find_content_list_files(in_root):
        return [in_root]

    dirs = []
    for p in in_root.iterdir():
        if p.is_dir() and find_content_list_files(p):
            dirs.append(p)
    return sorted(dirs)


# ==============================
# Module: type_router
# ==============================

KNOWN_TYPES = {"text", "image", "table", "equation"}
NOISE_TYPES = {"discarded"}  # you decided to drop it


def canonicalize_type(mineru_type: str, block: Dict[str, Any]) -> str:
    """
    Lightweight canonicalization by field signatures to tolerate future type drift.
    We still record mineru_type verbatim.

    Priority by signature:
      table: has table_body OR table_caption/table_footnote
      equation: text_format == 'latex'
      image: has img_path + image_caption/image_footnote
      text: has text (non-empty)
      otherwise: other
    """
    t = (mineru_type or "").strip().lower()

    if t in NOISE_TYPES:
        return "discarded"

    if "table_body" in block or "table_caption" in block or "table_footnote" in block:
        return "table"

    if str(block.get("text_format", "")).strip().lower() == "latex":
        return "equation"

    if "img_path" in block and ("image_caption" in block or "image_footnote" in block):
        return "image"

    if isinstance(block.get("text"), str) and block.get("text", "").strip():
        return "text"

    if t in KNOWN_TYPES:
        return t

    return "other"


def is_noise_type(mineru_type: str) -> bool:
    return (mineru_type or "").strip().lower() in NOISE_TYPES


# ==============================
# Module: record_builder
# ==============================

def resolve_img_abs_path(parse_run_dir: Path, img_rel_path: str) -> Path:
    # MinerU stores img_rel_path like "images/xxx.jpg" relative to parse_run_dir (auto/)
    return (parse_run_dir / img_rel_path).resolve()


def build_assetpack_record(
    doc_out_dir: Path,
    content_list_path: Path,
    middle_meta: Dict[str, Any],
    block: Dict[str, Any],
    keep_unknown_raw: bool,
    strict_img_exist: bool,
) -> Dict[str, Any]:
    mineru_type = block.get("type", None)
    canonical_type = canonicalize_type(mineru_type, block)

    page_idx = block.get("page_idx", None)
    bbox = block.get("bbox", None)

    # Critical: parse_run_dir is where content_list.json lives (usually .../auto/)
    parse_run_dir = content_list_path.parent

    rec: Dict[str, Any] = {
        # Two levels of provenance:
        # doc_out_dir: per-PDF mineru out folder (xxx__hash)
        # doc_parse_dir: parse run folder containing content_list + images (usually .../auto/)
        "doc_out_dir": str(doc_out_dir),
        "doc_parse_dir": str(parse_run_dir),
        "content_list_path": str(content_list_path),

        # span / location
        "page_idx": page_idx,
        "bbox_norm1000": bbox,
        "coord_sys": "mineru_norm1000",

        # type fields
        "mineru_type": mineru_type,
        "type": canonical_type,
        "is_noise": False,

        # mineru provenance
        "mineru_backend": middle_meta.get("_backend"),
        "mineru_version": middle_meta.get("_version_name"),

        # payload fields
        "text": "",
        "img_rel_path": None,
        "img_abs_path": None,
        "table_html": None,

        # extra/raw
        "extra": {},
    }

    # Handle noise (should be dropped upstream, but keep safe)
    if canonical_type == "discarded":
        rec["is_noise"] = True
        rec["text"] = str(block.get("text", "") or "")

    elif canonical_type == "text":
        rec["text"] = str(block.get("text", "") or "")
        if "text_level" in block:
            rec["extra"]["text_level"] = block.get("text_level")
        if "text_format" in block:
            rec["extra"]["text_format"] = block.get("text_format")

    elif canonical_type == "image":
        img_rel = block.get("img_path")
        cap = join_lines(block.get("image_caption"))
        foot = join_lines(block.get("image_footnote"))
        rec["text"] = "\n".join([t for t in [cap, foot] if t]).strip()

        rec["img_rel_path"] = img_rel
        if img_rel:
            img_abs = resolve_img_abs_path(parse_run_dir, img_rel)
            rec["img_abs_path"] = str(img_abs)
            if strict_img_exist and (not img_abs.exists()):
                raise FileNotFoundError(f"Missing image: {img_abs}")

        rec["extra"]["image_caption_raw"] = block.get("image_caption")
        rec["extra"]["image_footnote_raw"] = block.get("image_footnote")

    elif canonical_type == "table":
        img_rel = block.get("img_path")
        cap = join_lines(block.get("table_caption"))
        foot = join_lines(block.get("table_footnote"))
        html = block.get("table_body", "")

        rec["text"] = "\n".join([t for t in [cap, foot] if t]).strip()
        rec["table_html"] = html

        rec["img_rel_path"] = img_rel
        if img_rel:
            img_abs = resolve_img_abs_path(parse_run_dir, img_rel)
            rec["img_abs_path"] = str(img_abs)
            if strict_img_exist and (not img_abs.exists()):
                raise FileNotFoundError(f"Missing table image: {img_abs}")

        rec["extra"]["table_caption_raw"] = block.get("table_caption")
        rec["extra"]["table_footnote_raw"] = block.get("table_footnote")

    elif canonical_type == "equation":
        img_rel = block.get("img_path")
        rec["text"] = str(block.get("text", "") or "")
        if "text_format" in block:
            rec["extra"]["text_format"] = block.get("text_format")

        rec["img_rel_path"] = img_rel
        if img_rel:
            img_abs = resolve_img_abs_path(parse_run_dir, img_rel)
            rec["img_abs_path"] = str(img_abs)
            if strict_img_exist and (not img_abs.exists()):
                raise FileNotFoundError(f"Missing equation image: {img_abs}")

    else:
        # other / unknown types: keep text if present + optional raw_block for later model auditing
        if isinstance(block.get("text"), str) and block.get("text", "").strip():
            rec["text"] = block.get("text", "").strip()

        if "img_path" in block:
            rec["img_rel_path"] = block.get("img_path")
            if rec["img_rel_path"]:
                img_abs = resolve_img_abs_path(parse_run_dir, rec["img_rel_path"])
                rec["img_abs_path"] = str(img_abs)
                if strict_img_exist and (not img_abs.exists()):
                    raise FileNotFoundError(f"Missing referenced image: {img_abs}")

        if "table_body" in block:
            rec["table_html"] = block.get("table_body")

        if keep_unknown_raw:
            rec["extra"]["raw_block"] = block
        else:
            for k in ["type", "text", "img_path", "bbox", "page_idx", "text_format"]:
                if k in block:
                    rec["extra"][k] = block.get(k)

    # Stable content hash for dedup/incremental
    hash_obj = {
        "mineru_type": rec.get("mineru_type"),
        "type": rec.get("type"),
        "page_idx": rec.get("page_idx"),
        "bbox_norm1000": rec.get("bbox_norm1000"),
        "text": rec.get("text"),
        "img_rel_path": rec.get("img_rel_path"),
        "table_html": rec.get("table_html"),
        "mineru_backend": rec.get("mineru_backend"),
        "mineru_version": rec.get("mineru_version"),
        "doc_parse_dir": rec.get("doc_parse_dir"),
    }
    rec["content_hash"] = sha256_text(stable_dumps(hash_obj))
    rec["canonical_view"] = (rec.get("text") or "")[:200]

    return rec


# ==============================
# Module: main
# ==============================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", required=True, help="MinerU output root (contains per-PDF dirs) or a single per-PDF dir.")
    ap.add_argument("--out_jsonl", required=True, help="Output AssetPack JSONL path.")
    ap.add_argument("--emit_manifest", default="", help="Optional manifest JSON output path.")

    ap.add_argument("--drop_discarded", action="store_true", default=True,
                    help="Drop MinerU blocks with type=discarded (default: True).")
    ap.add_argument("--drop_unknown", action="store_true", default=False,
                    help="Drop blocks with unknown/new types (default: False).")
    ap.add_argument("--keep_unknown_raw", action="store_true", default=True,
                    help="Keep full raw_block for unknown/new types (default: True).")

    ap.add_argument("--strict_img_exist", action="store_true", default=False,
                    help="Fail if any referenced image path does not exist (default: False).")

    args = ap.parse_args()

    in_root = Path(args.in_root).expanduser().resolve()
    out_jsonl = Path(args.out_jsonl).expanduser().resolve()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

        # --- FIX: batch root should enumerate FIRST-LEVEL subdirs as docs ---
    doc_out_dirs = []
    for p in in_root.iterdir():
        if p.is_dir() and find_content_list_files(p):
            doc_out_dirs.append(p)
    
    doc_out_dirs = sorted(doc_out_dirs)
    if not doc_out_dirs:
        # fallback: treat in_root as a single doc dir
        if find_content_list_files(in_root):
            doc_out_dirs = [in_root]
        else:
            raise RuntimeError(f"No MinerU output directories found under: {in_root}")
    
    if not doc_out_dirs:
        raise RuntimeError(f"No MinerU output directories found under: {in_root}")

    total_docs = 0
    total_blocks_in = 0
    total_blocks_out = 0
    total_discarded_dropped = 0
    total_unknown_dropped = 0

    doc_stats: List[Dict[str, Any]] = []

    with out_jsonl.open("w", encoding="utf-8") as fw:
        for doc_dir in doc_out_dirs:
            total_docs += 1

            cl_files = find_content_list_files(doc_dir)
            if not cl_files:
                continue

            # For now take the first content_list. If you later run multiple variants, we can iterate all.
            content_list_path = cl_files[0]

            ml_files = find_middle_files(doc_dir)
            middle_path = ml_files[0] if ml_files else None
            middle_meta: Dict[str, Any] = {}
            if middle_path is not None:
                try:
                    middle_meta = load_json(middle_path)
                except Exception:
                    middle_meta = {}

            content_list = load_json(content_list_path)
            if not isinstance(content_list, list):
                raise ValueError(f"content_list is not a list: {content_list_path}")

            counts_out = {"text": 0, "image": 0, "table": 0, "equation": 0, "other": 0, "discarded_in": 0}
            dropped = {"discarded": 0, "unknown": 0}

            for block in content_list:
                total_blocks_in += 1
                mineru_type = (block.get("type") or "").strip().lower()

                if mineru_type == "discarded":
                    counts_out["discarded_in"] += 1
                    if args.drop_discarded:
                        dropped["discarded"] += 1
                        total_discarded_dropped += 1
                        continue

                canonical_type = canonicalize_type(mineru_type, block)
                if canonical_type == "other" and args.drop_unknown:
                    dropped["unknown"] += 1
                    total_unknown_dropped += 1
                    continue

                rec = build_assetpack_record(
                    doc_out_dir=doc_dir,
                    content_list_path=content_list_path,
                    middle_meta=middle_meta,
                    block=block,
                    keep_unknown_raw=args.keep_unknown_raw,
                    strict_img_exist=args.strict_img_exist,
                )

                t = rec.get("type") or "other"
                if t in ("text", "image", "table", "equation"):
                    counts_out[t] += 1
                else:
                    counts_out["other"] += 1

                fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total_blocks_out += 1

            doc_stats.append({
                "doc_out_dir": str(doc_dir),
                "content_list_path": str(content_list_path),
                "parse_run_dir": str(content_list_path.parent),
                "middle_path": str(middle_path) if middle_path else None,
                "mineru_backend": middle_meta.get("_backend"),
                "mineru_version": middle_meta.get("_version_name"),
                "counts_out": counts_out,
                "dropped": dropped,
            })

    if args.emit_manifest:
        manifest_path = Path(args.emit_manifest).expanduser().resolve()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "in_root": str(in_root),
            "out_jsonl": str(out_jsonl),
            "total_docs": total_docs,
            "total_blocks_in": total_blocks_in,
            "total_blocks_out": total_blocks_out,
            "total_discarded_dropped": total_discarded_dropped,
            "total_unknown_dropped": total_unknown_dropped,
            "docs": doc_stats,
            "policy": {
                "drop_discarded": bool(args.drop_discarded),
                "drop_unknown": bool(args.drop_unknown),
                "keep_unknown_raw": bool(args.keep_unknown_raw),
                "strict_img_exist": bool(args.strict_img_exist),
            }
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("========== AssetPack DONE ==========")
    print(f"Input root : {in_root}")
    print(f"Docs       : {total_docs}")
    print(f"Blocks(in) : {total_blocks_in}")
    print(f"Blocks(out): {total_blocks_out}")
    print(f"Dropped discarded: {total_discarded_dropped}")
    print(f"Dropped unknown  : {total_unknown_dropped}")
    print(f"AssetPack  : {out_jsonl}")
    if args.emit_manifest:
        print(f"Manifest   : {args.emit_manifest}")


if __name__ == "__main__":
    main()
