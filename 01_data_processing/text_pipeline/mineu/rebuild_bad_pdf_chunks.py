#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List


_PAGES_RE = re.compile(r"^Pages:\s+(\d+)\s*$", re.MULTILINE)


def load_pdf_paths(args: argparse.Namespace) -> List[Path]:
    pdfs: List[Path] = []
    for raw in args.pdf:
        pdfs.append(Path(raw).expanduser().resolve())

    if args.inventory_json:
        items = json.loads(Path(args.inventory_json).expanduser().resolve().read_text(encoding="utf-8"))
        for item in items:
            pdfs.append(Path(item["pdf"]).expanduser().resolve())

    uniq: List[Path] = []
    seen = set()
    for pdf in pdfs:
        key = str(pdf)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(pdf)
    return uniq


def safe_name(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[^\w\-]+", "_", stem, flags=re.UNICODE).strip("_")
    return stem or "pdf"


def pdfinfo_pages(pdf: Path) -> int:
    proc = subprocess.run(
        ["pdfinfo", str(pdf)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"pdfinfo failed for {pdf}: {proc.stderr.strip()}")
    m = _PAGES_RE.search(proc.stdout)
    if not m:
        raise RuntimeError(f"Could not parse page count from pdfinfo output for {pdf}")
    return int(m.group(1))


def build_chunk(pdf: Path, start: int, end: int, out_pdf: Path) -> Dict:
    proc = subprocess.run(
        [
            "pdftocairo",
            "-pdf",
            "-f",
            str(start),
            "-l",
            str(end),
            str(pdf),
            str(out_pdf),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    ok = proc.returncode == 0 and out_pdf.exists() and out_pdf.stat().st_size > 0
    return {
        "chunk_pdf": str(out_pdf),
        "start_page": start,
        "end_page": end,
        "ok": ok,
        "returncode": proc.returncode,
        "stderr": proc.stderr.strip(),
        "size_mb": round(out_pdf.stat().st_size / 1024 / 1024, 2) if out_pdf.exists() else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", action="append", default=[], help="Repeatable bad PDF path.")
    ap.add_argument("--inventory_json", default="", help="Optional JSON inventory with items containing `pdf`.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--chunk_pages", type=int, default=25)
    args = ap.parse_args()

    pdfs = load_pdf_paths(args)
    if not pdfs:
        raise SystemExit("No PDFs supplied. Use --pdf and/or --inventory_json.")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: List[Dict] = []
    chunk_paths: List[str] = []

    for pdf in pdfs:
        pages = pdfinfo_pages(pdf)
        pdf_dir = out_dir / safe_name(pdf.name)
        pdf_dir.mkdir(parents=True, exist_ok=True)

        item = {
            "pdf": str(pdf),
            "pages": pages,
            "chunk_pages": args.chunk_pages,
            "chunks": [],
        }

        for start in range(1, pages + 1, args.chunk_pages):
            end = min(start + args.chunk_pages - 1, pages)
            out_pdf = pdf_dir / f"{safe_name(pdf.name)}_p{start:04d}_{end:04d}.pdf"
            chunk = build_chunk(pdf, start, end, out_pdf)
            item["chunks"].append(chunk)
            if chunk["ok"]:
                chunk_paths.append(chunk["chunk_pdf"])

        manifest.append(item)

    manifest_path = out_dir / "rebuild_manifest.json"
    input_list_path = out_dir / "rebuild_input_list.txt"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    input_list_path.write_text("\n".join(chunk_paths) + ("\n" if chunk_paths else ""), encoding="utf-8")

    total_chunks = sum(len(item["chunks"]) for item in manifest)
    ok_chunks = sum(1 for item in manifest for chunk in item["chunks"] if chunk["ok"])
    print(f"pdfs={len(manifest)} total_chunks={total_chunks} ok_chunks={ok_chunks}")
    print(f"manifest={manifest_path}")
    print(f"input_list={input_list_path}")


if __name__ == "__main__":
    main()
