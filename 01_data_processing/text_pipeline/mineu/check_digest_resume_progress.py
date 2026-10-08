#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
from pathlib import Path


def safe_stem(name: str) -> str:
    s = Path(name).stem
    s = re.sub(r"[^\w\-]+", "_", s, flags=re.UNICODE).strip("_")
    return s or "pdf"


def out_dir_for(pdf: Path, out_root: Path) -> Path:
    short = hashlib.sha1(str(pdf.resolve()).encode("utf-8")).hexdigest()[:8]
    return out_root / f"{safe_stem(pdf.name)}__{short}"


def has_success(pdf: Path, out_root: Path) -> bool:
    d = out_dir_for(pdf, out_root)
    return d.exists() and any(d.rglob("*_content_list.json"))


def count_summary_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out_root", required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    out_root = Path(args.out_root).resolve()
    manifest = json.loads((run_dir / "resume_manifest.json").read_text(encoding="utf-8"))

    shard_status = []
    main_done = 0
    for shard in manifest["shards"]:
        shard_id = shard["shard_id"]
        input_list = Path(shard["input_list"])
        pdfs = [Path(x) for x in input_list.read_text(encoding="utf-8").splitlines() if x.strip()]
        done = sum(1 for pdf in pdfs if has_success(pdf, out_root))
        summary_rows = count_summary_rows(run_dir / f"shard_{shard_id}_summary.jsonl")
        failed_rows = count_summary_rows(run_dir / f"shard_{shard_id}_failed.txt")
        shard_status.append(
            {
                "shard_id": shard_id,
                "assigned": len(pdfs),
                "done": done,
                "remaining": len(pdfs) - done,
                "summary_rows": summary_rows,
                "failed_rows": failed_rows,
                "input_list": str(input_list),
                "log_path": str(run_dir / "logs" / f"shard_{shard_id}.log"),
            }
        )
        main_done += done

    snapshot = {
        "run_dir": str(run_dir),
        "out_root": str(out_root),
        "main_queue_total": manifest["main_queue_total"],
        "main_queue_done": main_done,
        "main_queue_remaining": manifest["main_queue_total"] - main_done,
        "priority_total": manifest["priority_total"],
        "known_badpdf_total": manifest["known_badpdf_total"],
        "shards": shard_status,
    }
    print(json.dumps(snapshot, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
