#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_inventory(path: Path) -> List[Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Inventory is not a list: {path}")
    return data


def score_item(item: Dict) -> float:
    pages = item.get("pages") or 0
    size_mb = item.get("size_mb") or 0
    if pages > 0:
        return float(pages)
    # Fallback for unreadable page counts
    return max(float(size_mb), 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_shards", type=int, default=4)
    args = ap.parse_args()

    inventory_path = Path(args.inventory_json).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    items = load_inventory(inventory_path)
    # Large-first greedy balancing by page count
    items = sorted(items, key=score_item, reverse=True)

    shards: List[List[Dict]] = [[] for _ in range(args.num_shards)]
    loads = [0.0 for _ in range(args.num_shards)]
    for item in items:
        idx = min(range(args.num_shards), key=lambda i: loads[i])
        shards[idx].append(item)
        loads[idx] += score_item(item)

    manifest = {
        "inventory_json": str(inventory_path),
        "num_shards": args.num_shards,
        "total_items": len(items),
        "shards": [],
    }

    for idx, shard_items in enumerate(shards):
        shard_path = out_dir / f"failed_shard_{idx}.txt"
        shard_path.write_text(
            "\n".join(item["pdf"] for item in shard_items) + ("\n" if shard_items else ""),
            encoding="utf-8",
        )
        shard_pages = sum(item.get("pages") or 0 for item in shard_items)
        shard_size_mb = round(sum(item.get("size_mb") or 0 for item in shard_items), 2)
        shard_load = round(loads[idx], 2)
        manifest["shards"].append(
            {
                "index": idx,
                "txt_path": str(shard_path),
                "count": len(shard_items),
                "pages": shard_pages,
                "size_mb": shard_size_mb,
                "load_score": shard_load,
            }
        )

    manifest_path = out_dir / "failed_shards_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Prepared {len(items)} PDFs into {args.num_shards} shards")
    for shard in manifest["shards"]:
        print(
            f"Shard {shard['index']}: count={shard['count']}, "
            f"pages={shard['pages']}, size_mb={shard['size_mb']}, load={shard['load_score']}"
        )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
