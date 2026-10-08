#!/usr/bin/env python3
"""Split large JSONL datasets into <50MB parts for git storage, and merge back.

git (and many hosts) reject blobs >~50MB. This tool splits a JSONL file into
line-aligned parts each <= --max-mb (default 45MB, safely under the 50MB limit),
writing a manifest with sizes + sha256 so the file can be losslessly rebuilt.

Usage:
    # split
    python dataset_pack.py split --input train/sft_ctrl_train.jsonl \
        --outdir train/packed --max-mb 45
    # merge (verify + reconstruct)
    python dataset_pack.py merge --manifest train/packed/sft_ctrl_train.manifest.json \
        --output train/sft_ctrl_train.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split(input_path: Path, outdir: Path, max_mb: float, prefix: str = "") -> Dict:
    max_bytes = int(max_mb * 1024 * 1024)
    outdir.mkdir(parents=True, exist_ok=True)
    name = prefix or input_path.name  # e.g. sft_ctrl_train.jsonl
    stem = name[:-6] if name.endswith(".jsonl") else name

    parts: List[Dict] = []
    part_idx = 0
    num_lines = 0

    def _open_part(i: int):
        p = outdir / f"{stem}.part{i:03d}.jsonl"
        return p, p.open("wb")

    cur_path, cur = _open_part(part_idx)
    cur_size = 0
    try:
        with input_path.open("rb") as fin:
            for raw in fin:
                num_lines += 1
                if cur_size > 0 and cur_size + len(raw) > max_bytes:
                    cur.close()
                    parts.append({"name": cur_path.name, "size": cur_size, "sha256": _sha256(cur_path)})
                    part_idx += 1
                    cur_path, cur = _open_part(part_idx)
                    cur_size = 0
                cur.write(raw)
                cur_size += len(raw)
    finally:
        cur.close()
    if cur_size > 0:
        parts.append({"name": cur_path.name, "size": cur_size, "sha256": _sha256(cur_path)})
    elif cur_path.exists() and part_idx > 0:
        cur_path.unlink()  # remove empty trailing part

    manifest = {
        "orig_name": name,
        "orig_size": input_path.stat().st_size,
        "orig_sha256": _sha256(input_path),
        "num_lines": num_lines,
        "max_mb": max_mb,
        "parts": parts,
    }
    man_path = outdir / f"{stem}.manifest.json"
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(man_path), "parts": len(parts),
                      "orig_size_mb": round(manifest["orig_size"] / 1024 / 1024, 2),
                      "num_lines": num_lines}, ensure_ascii=False))
    return manifest


def merge(manifest_path: Path, output_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    outdir = manifest_path.parent
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fout:
        for part in manifest["parts"]:
            p = outdir / part["name"]
            got = _sha256(p)
            if got != part["sha256"]:
                raise SystemExit(f"sha256 mismatch for part {part['name']}: {got} != {part['sha256']}")
            with p.open("rb") as fin:
                for chunk in iter(lambda: fin.read(1 << 20), b""):
                    fout.write(chunk)
    final = _sha256(output_path)
    if final != manifest["orig_sha256"]:
        raise SystemExit(f"reconstructed sha256 mismatch: {final} != {manifest['orig_sha256']}")
    print(json.dumps({"output": str(output_path), "sha256_ok": True,
                      "size_mb": round(output_path.stat().st_size / 1024 / 1024, 2),
                      "num_lines": manifest["num_lines"]}, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("split")
    sp.add_argument("--input", required=True)
    sp.add_argument("--outdir", required=True)
    sp.add_argument("--max-mb", type=float, default=45.0)
    sp.add_argument("--prefix", default="")
    mp = sub.add_parser("merge")
    mp.add_argument("--manifest", required=True)
    mp.add_argument("--output", required=True)
    args = ap.parse_args()
    if args.cmd == "split":
        split(Path(args.input), Path(args.outdir), args.max_mb, args.prefix)
    else:
        merge(Path(args.manifest), Path(args.output))


if __name__ == "__main__":
    main()
