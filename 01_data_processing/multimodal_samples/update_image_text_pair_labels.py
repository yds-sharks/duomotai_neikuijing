#!/usr/bin/env python3
"""Backfill labels for image-text pairs in multimodal_samples.db."""

import argparse
import json
import sqlite3
from typing import Dict, Iterable, List, Tuple


def normalize_tags(value) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def normalize_secondary_types(value) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def build_labels_payload(
    organ_tags: List[str],
    primary_knowledge_type: str,
    secondary_knowledge_types: List[str],
) -> Dict[str, object]:
    return {
        "level1": organ_tags,
        "level2_main": primary_knowledge_type,
        "level2_aux": secondary_knowledge_types,
    }


def build_retrieval_flags(organ_tags: List[str]) -> Dict[str, object]:
    is_general = "通用" in organ_tags
    if is_general:
        is_mixed = False
    else:
        is_mixed = len([item for item in organ_tags if item != "通用"]) >= 2
    return {
        "is_general": is_general,
        "is_mixed": is_mixed,
        "is_key_knowledge": False,
    }


def build_retrieval_meta(
    organ_tags: List[str],
    primary_knowledge_type: str,
) -> Dict[str, object]:
    if "通用" in organ_tags:
        body_site_main = "通用"
    else:
        body_site_main = organ_tags[0] if organ_tags else ""
    return {
        "body_site_main": body_site_main,
        "body_site_all": organ_tags,
        "knowledge_type_main": primary_knowledge_type,
    }


def load_db_key_map(conn: sqlite3.Connection) -> Dict[Tuple[str, str], str]:
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT sample_id,
               json_extract(images, '$[0].image_path') AS image_path,
               text
        FROM multimodal_samples
        WHERE source_type = 'image_text_pair'
        """
    )
    mapping: Dict[Tuple[str, str], str] = {}
    duplicates = 0
    for sample_id, image_path, text in cursor.fetchall():
        key = (image_path or "", text or "")
        if key in mapping:
            duplicates += 1
        mapping[key] = sample_id
    if duplicates:
        raise ValueError(f"Found {duplicates} duplicate image-text keys in multimodal_samples")
    return mapping


def build_update_rows(
    labeled_jsonl: str,
    key_to_sample_id: Dict[Tuple[str, str], str],
) -> Iterable[Tuple[str, str, str, str, str, str, str]]:
    seen_sample_ids = set()
    with open(labeled_jsonl, "r", encoding="utf-8") as file_obj:
        for line_no, line in enumerate(file_obj, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            image_path = str(record.get("image_path") or "")
            text = str(record.get("final_description") or "")
            key = (image_path, text)
            sample_id = key_to_sample_id.get(key)
            if sample_id is None:
                raise KeyError(f"No matching sample found at line {line_no}: {image_path}")
            if sample_id in seen_sample_ids:
                raise ValueError(f"Duplicate labeled mapping for sample_id={sample_id} at line {line_no}")
            seen_sample_ids.add(sample_id)

            organ_tags = normalize_tags(record.get("organ_tags"))
            primary_knowledge_type = str(record.get("primary_knowledge_type") or "").strip()
            secondary_knowledge_types = normalize_secondary_types(record.get("secondary_knowledge_types"))

            labels = build_labels_payload(
                organ_tags=organ_tags,
                primary_knowledge_type=primary_knowledge_type,
                secondary_knowledge_types=secondary_knowledge_types,
            )
            retrieval_flags = build_retrieval_flags(organ_tags)
            retrieval_meta = build_retrieval_meta(
                organ_tags=organ_tags,
                primary_knowledge_type=primary_knowledge_type,
            )

            yield (
                json.dumps(organ_tags, ensure_ascii=False),
                primary_knowledge_type,
                json.dumps(secondary_knowledge_types, ensure_ascii=False),
                json.dumps(labels, ensure_ascii=False),
                json.dumps(retrieval_flags, ensure_ascii=False),
                json.dumps(retrieval_meta, ensure_ascii=False),
                sample_id,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill image-text labels into multimodal_samples.db")
    parser.add_argument(
        "--db-path",
        default="/mnt/data_1/yds/多模态/data_house/multimodal_samples.db",
        help="Path to multimodal_samples.db",
    )
    parser.add_argument(
        "--labeled-jsonl",
        default="/mnt/data_1/yds/多模态/data_house/origin_data/output_pairs_all_min_filtered_labeled.jsonl",
        help="Path to labeled image-text JSONL",
    )
    args = parser.parse_args()

    conn = sqlite3.connect(args.db_path)
    try:
        key_to_sample_id = load_db_key_map(conn)
        update_rows = list(build_update_rows(args.labeled_jsonl, key_to_sample_id))
        if len(update_rows) != len(key_to_sample_id):
            raise ValueError(
                f"Update row count mismatch: updates={len(update_rows)} db_pairs={len(key_to_sample_id)}"
            )

        conn.execute("BEGIN")
        conn.executemany(
            """
            UPDATE multimodal_samples
            SET organ_tags = ?,
                primary_knowledge_type = ?,
                secondary_knowledge_types = ?,
                labels = ?,
                retrieval_flags = ?,
                retrieval_meta = ?
            WHERE sample_id = ?
            """,
            update_rows,
        )
        conn.commit()
        print(f"[done] updated_rows={len(update_rows)}")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
