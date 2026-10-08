#!/usr/bin/env python3
"""Encode a single image into the same vector space used by the image index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--retrieval-package-root", default="/mnt/data_1/yds/多模态/retrieval")
    parser.add_argument("--image-device", default="cuda")
    parser.add_argument("--image-torch-dtype", default="auto")
    args = parser.parse_args()

    module_root = str(Path(args.retrieval_package_root) / "多模态")
    sys.path.insert(0, module_root)
    from image_dense_module import embed_images, load_qwen3vl_image_encoder

    processor, model = load_qwen3vl_image_encoder(
        device=args.image_device,
        torch_dtype=args.image_torch_dtype,
    )
    vector = embed_images(
        [args.image_path],
        processor=processor,
        model=model,
        device=args.image_device,
        pooling="mean",
    )[0]
    print(json.dumps({"image_path": args.image_path, "vector": vector}, ensure_ascii=False))


if __name__ == "__main__":
    main()
