#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path


ROOT = Path("/mnt/data_1/yds/多模态/data_house")
TEXT_DB = ROOT / "origin_data/blocks_classification/text_database_stage3_filtered_labeled.db"
IMAGE_TEXT_JSONL = ROOT / "origin_data/image_text/output_pairs_all_min_filtered.jsonl"
OUTPUT_DB = ROOT / "multimodal_samples.db"


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def parse_json_array(raw, field_name):
    if raw in (None, ""):
        return []
    if isinstance(raw, list):
        return raw
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} is not valid JSON: {raw}") from exc
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a JSON array: {raw}")
    return value


def parse_optional_json(raw, field_name):
    if raw in (None, ""):
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} is not valid JSON: {raw}") from exc
    return canonical_json(value)


def normalize_organ_tags(organ_tags):
    if not organ_tags:
        return []
    normalized = []
    seen = set()
    for tag in organ_tags:
        if tag not in seen:
            normalized.append(tag)
            seen.add(tag)
    return normalized


def build_labels(organ_tags, primary_knowledge_type, secondary_knowledge_types):
    return {
        "level1": organ_tags,
        "level2_main": primary_knowledge_type,
        "level2_aux": secondary_knowledge_types,
    }


def build_retrieval_flags(organ_tags):
    concrete = [tag for tag in organ_tags if tag != "通用"]
    return {
        "is_general": "通用" in organ_tags,
        "is_mixed": False if "通用" in organ_tags else len(concrete) >= 2,
        "is_key_knowledge": False,
    }


def build_retrieval_meta(organ_tags, primary_knowledge_type):
    return {
        "body_site_main": "通用" if "通用" in organ_tags else (organ_tags[0] if organ_tags else ""),
        "body_site_all": organ_tags,
        "knowledge_type_main": primary_knowledge_type,
    }


def extract_doc_info_from_image_path(image_path):
    path = Path(image_path)
    doc_name = path.parent.parent.parent.name if len(path.parents) >= 3 else ""
    doc_container = path.parent.parent.parent.parent.name if len(path.parents) >= 4 else ""
    match = re.search(r"__([^/_]+)$", doc_container)
    doc_id = match.group(1) if match else ""
    source_path = str(path.parent.parent.parent) if len(path.parents) >= 3 else str(path.parent)
    return doc_id, doc_name, source_path


def create_schema(conn):
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;

        CREATE TABLE multimodal_samples (
            sample_id TEXT PRIMARY KEY,
            group_id TEXT NOT NULL,
            source_type TEXT NOT NULL CHECK (source_type IN ('text_only', 'image_text_pair')),
            have_image INTEGER NOT NULL CHECK (have_image IN (0, 1)),
            doc_id TEXT,
            doc_name TEXT,
            source_path TEXT,
            page_idx INTEGER,
            block_id INTEGER,
            content_hash TEXT,
            text TEXT NOT NULL,
            text_role TEXT NOT NULL CHECK (text_role IN ('knowledge_block', 'image_description')),
            organ_tags TEXT NOT NULL,
            primary_knowledge_type TEXT NOT NULL,
            secondary_knowledge_types TEXT NOT NULL,
            labels TEXT NOT NULL,
            retrieval_flags TEXT NOT NULL,
            retrieval_meta TEXT NOT NULL,
            images TEXT NOT NULL,
            bbox TEXT,
            coord_sys TEXT,
            created_at TEXT NOT NULL,
            extra TEXT NOT NULL
        );

        CREATE INDEX idx_multimodal_group_id ON multimodal_samples(group_id);
        CREATE INDEX idx_multimodal_source_type ON multimodal_samples(source_type);
        CREATE INDEX idx_multimodal_have_image ON multimodal_samples(have_image);
        CREATE INDEX idx_multimodal_doc_id ON multimodal_samples(doc_id);
        CREATE INDEX idx_multimodal_page_idx ON multimodal_samples(page_idx);
        CREATE INDEX idx_multimodal_block_id ON multimodal_samples(block_id);
        CREATE INDEX idx_multimodal_primary_knowledge_type ON multimodal_samples(primary_knowledge_type);
        """
    )


def insert_text_samples(conn):
    source = sqlite3.connect(TEXT_DB)
    source.row_factory = sqlite3.Row
    cursor = source.execute(
        """
        SELECT
            tb.block_id,
            tb.doc_id,
            d.doc_name,
            d.source_path,
            tb.page_idx,
            tb.content_hash,
            tb.text,
            tb.bbox,
            tb.coord_sys,
            tb.mineru_version,
            tb.is_noise,
            tb.char_length,
            tb.created_at,
            tb.organ_tags,
            tb.primary_knowledge_type,
            tb.secondary_knowledge_types
        FROM text_blocks AS tb
        LEFT JOIN documents AS d ON d.doc_id = tb.doc_id
        ORDER BY tb.block_id
        """
    )

    rows = []
    count = 0
    for row in cursor:
        raw_organ_tags = parse_json_array(row["organ_tags"], "organ_tags")
        organ_tags = normalize_organ_tags(raw_organ_tags)
        secondary = parse_json_array(row["secondary_knowledge_types"], "secondary_knowledge_types")

        primary = row["primary_knowledge_type"] or ""
        labels = build_labels(organ_tags, primary, secondary)
        retrieval_flags = build_retrieval_flags(organ_tags)
        retrieval_meta = build_retrieval_meta(organ_tags, primary)
        extra = {
            "record_origin": "blocks_classification",
            "raw_organ_tags": raw_organ_tags,
            "mineru_version": row["mineru_version"],
            "is_noise": bool(row["is_noise"]) if row["is_noise"] is not None else False,
            "char_length": row["char_length"],
        }

        rows.append(
            (
                f"txt_{row['doc_id']}_{row['block_id']}",
                f"grp_txt_{row['doc_id']}_{row['block_id']}",
                "text_only",
                0,
                row["doc_id"],
                row["doc_name"] or "",
                row["source_path"] or "",
                row["page_idx"],
                row["block_id"],
                row["content_hash"],
                row["text"],
                "knowledge_block",
                canonical_json(organ_tags),
                primary,
                canonical_json(secondary),
                canonical_json(labels),
                canonical_json(retrieval_flags),
                canonical_json(retrieval_meta),
                canonical_json([]),
                parse_optional_json(row["bbox"], "bbox"),
                row["coord_sys"],
                row["created_at"] or datetime.now().isoformat(timespec="seconds"),
                canonical_json(extra),
            )
        )
        count += 1
        if len(rows) >= 2000:
            conn.executemany(
                """
                INSERT INTO multimodal_samples (
                    sample_id, group_id, source_type, have_image, doc_id, doc_name, source_path,
                    page_idx, block_id, content_hash, text, text_role, organ_tags,
                    primary_knowledge_type, secondary_knowledge_types, labels,
                    retrieval_flags, retrieval_meta, images, bbox, coord_sys, created_at, extra
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            rows.clear()

    if rows:
        conn.executemany(
            """
            INSERT INTO multimodal_samples (
                sample_id, group_id, source_type, have_image, doc_id, doc_name, source_path,
                page_idx, block_id, content_hash, text, text_role, organ_tags,
                primary_knowledge_type, secondary_knowledge_types, labels,
                retrieval_flags, retrieval_meta, images, bbox, coord_sys, created_at, extra
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    source.close()
    return count


def insert_image_text_samples(conn):
    now = datetime.now().isoformat(timespec="seconds")
    rows = []
    count = 0
    with IMAGE_TEXT_JSONL.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            record = json.loads(line)
            image_path = record["image_path"]
            text = record["final_description"]
            doc_id, doc_name, source_path = extract_doc_info_from_image_path(image_path)

            organ_tags = []
            primary = ""
            secondary = []
            labels = build_labels(organ_tags, primary, secondary)
            retrieval_flags = build_retrieval_flags(organ_tags)
            retrieval_meta = build_retrieval_meta(organ_tags, primary)
            content_hash = hashlib.sha256(f"{image_path}\n{text}".encode("utf-8")).hexdigest()
            image_id_suffix = doc_id or "unknown"
            images = [
                {
                    "image_id": f"img_{image_id_suffix}_{line_no:06d}",
                    "image_path": image_path,
                    "image_type": "unknown",
                    "is_primary": True,
                }
            ]
            extra = {
                "record_origin": "image_text",
                "pair_source": str(IMAGE_TEXT_JSONL),
                "line_no": line_no,
            }

            rows.append(
                (
                    f"imgtxt_{image_id_suffix}_{line_no:06d}",
                    f"grp_imgtxt_{image_id_suffix}_{line_no:06d}",
                    "image_text_pair",
                    1,
                    doc_id,
                    doc_name,
                    source_path,
                    None,
                    None,
                    content_hash,
                    text,
                    "image_description",
                    canonical_json(organ_tags),
                    primary,
                    canonical_json(secondary),
                    canonical_json(labels),
                    canonical_json(retrieval_flags),
                    canonical_json(retrieval_meta),
                    canonical_json(images),
                    None,
                    None,
                    now,
                    canonical_json(extra),
                )
            )
            count += 1

            if len(rows) >= 2000:
                conn.executemany(
                    """
                    INSERT INTO multimodal_samples (
                        sample_id, group_id, source_type, have_image, doc_id, doc_name, source_path,
                        page_idx, block_id, content_hash, text, text_role, organ_tags,
                        primary_knowledge_type, secondary_knowledge_types, labels,
                        retrieval_flags, retrieval_meta, images, bbox, coord_sys, created_at, extra
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                rows.clear()

    if rows:
        conn.executemany(
            """
            INSERT INTO multimodal_samples (
                sample_id, group_id, source_type, have_image, doc_id, doc_name, source_path,
                page_idx, block_id, content_hash, text, text_role, organ_tags,
                primary_knowledge_type, secondary_knowledge_types, labels,
                retrieval_flags, retrieval_meta, images, bbox, coord_sys, created_at, extra
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    return count


def main():
    parser = argparse.ArgumentParser(description="Build the unified multimodal_samples SQLite database.")
    parser.add_argument("--output-db", default=str(OUTPUT_DB))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_db = Path(args.output_db)
    if output_db.exists():
        if not args.overwrite:
            raise SystemExit(f"Output DB already exists: {output_db}. Use --overwrite to replace it.")
        output_db.unlink()

    output_db.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(output_db)
    try:
        create_schema(conn)
        text_count = insert_text_samples(conn)
        image_count = insert_image_text_samples(conn)
        conn.commit()
    finally:
        conn.close()

    print(f"Built {output_db}")
    print(f"text_only={text_count}")
    print(f"image_text_pair={image_count}")
    print(f"total={text_count + image_count}")


if __name__ == "__main__":
    main()
