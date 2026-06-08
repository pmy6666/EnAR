#!/usr/bin/env python3
"""Build a small per-topic VLMBias subset.

The source VLMBias snapshot is stored as HuggingFace-style parquet shards.
This script samples rows from a split, exports each row image, and writes one
directory per VLMBias topic under the requested output directory.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


DEFAULT_SOURCE_DIR = Path("EnAR/toy_dataset/VLMBias")
DEFAULT_OUTPUT_DIR = Path("EnAR/toy_dataset/small_dataset")
DEFAULT_SPLIT = "main"
DEFAULT_NUM_SAMPLES = 10
DEFAULT_SEED = 42

METADATA_COLUMNS = [
    "ID",
    "image_path",
    "topic",
    "sub_topic",
    "prompt",
    "ground_truth",
    "expected_bias",
    "with_title",
    "type_of_question",
    "pixel",
    "metadata",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample N rows from each VLMBias topic into a small dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="VLMBias dataset directory containing data/*.parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the sampled dataset will be written.",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help="VLMBias parquet split prefix, e.g. main, original, identification.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help="Number of rows to sample per topic.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove output-dir before writing the new small dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive.")

    source_dir = args.source_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    parquet_files = sorted((source_dir / "data").glob(f"{args.split}-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found for split '{args.split}' under {source_dir / 'data'}"
        )

    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}\n"
                "Use --overwrite to rebuild it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.concat((pd.read_parquet(path) for path in parquet_files), ignore_index=True)
    missing = {"image", *METADATA_COLUMNS} - set(df.columns)
    if missing:
        raise ValueError(f"VLMBias parquet is missing required columns: {sorted(missing)}")

    topics = sorted(str(topic) for topic in df["topic"].dropna().unique())
    all_records: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for topic in topics:
        topic_df = df[df["topic"].astype(str) == topic]
        if len(topic_df) > args.num_samples:
            sampled = topic_df.sample(n=args.num_samples, random_state=args.seed)
        else:
            sampled = topic_df.copy()
        sampled = sampled.reset_index(drop=False).rename(columns={"index": "source_row_index"})

        topic_dir = output_dir / slugify(topic)
        image_dir = topic_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        records: list[dict[str, Any]] = []
        for local_index, row in sampled.iterrows():
            raw_id = stringify(row["ID"]) or f"{slugify(topic)}_{local_index:03d}"
            sample_id = unique_sample_id(raw_id, records)
            image_path = image_dir / f"{sample_id}.png"
            export_image(row["image"], image_path, source_dir)

            record = {
                "sample_id": sample_id,
                "raw_id": raw_id,
                "split": args.split,
                "source_row_index": int(row["source_row_index"]),
                "relative_image_path": str(image_path.relative_to(output_dir)),
            }
            for column in METADATA_COLUMNS:
                record[column] = normalize_value(row[column])
            records.append(record)
            all_records.append({"topic_dir": topic_dir.name, **record})

        write_jsonl(topic_dir / "samples.jsonl", records)
        counts[topic] = len(records)

    manifest = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "split": args.split,
        "num_samples_per_topic": args.num_samples,
        "seed": args.seed,
        "topics": {topic: {"directory": slugify(topic), "num_samples": counts[topic]} for topic in topics},
        "total_samples": len(all_records),
        "parquet_files": [str(path) for path in parquet_files],
    }
    write_json(output_dir / "manifest.json", manifest)
    write_jsonl(output_dir / "sample_index.jsonl", all_records)

    print(f"Wrote {len(all_records)} samples across {len(topics)} topics to {output_dir}")
    for topic in topics:
        print(f"  {topic}: {counts[topic]} -> {output_dir / slugify(topic)}")


def export_image(image_value: Any, output_path: Path, dataset_root: Path) -> None:
    if isinstance(image_value, Image.Image):
        image_value.save(output_path)
        return

    if isinstance(image_value, (bytes, bytearray, memoryview)):
        Image.open(io.BytesIO(bytes(image_value))).save(output_path)
        return

    if isinstance(image_value, dict):
        data = image_value.get("bytes")
        if data is not None:
            Image.open(io.BytesIO(data)).save(output_path)
            return
        path_value = image_value.get("path")
        if path_value:
            copy_or_convert_image(Path(path_value), output_path, dataset_root)
            return

    raise TypeError(f"Unsupported image value: {type(image_value)!r}")


def copy_or_convert_image(source_path: Path, output_path: Path, dataset_root: Path) -> None:
    if not source_path.is_absolute():
        source_path = dataset_root / source_path
    source_path = source_path.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {source_path}")
    Image.open(source_path).save(output_path)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")


def unique_sample_id(raw_id: str, existing_records: list[dict[str, Any]]) -> str:
    base = slugify(raw_id)[:120] or "sample"
    existing_ids = {record["sample_id"] for record in existing_records}
    if base not in existing_ids:
        return base
    suffix = 1
    while f"{base}_{suffix}" in existing_ids:
        suffix += 1
    return f"{base}_{suffix}"


def slugify(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    clean = clean.strip("._")
    return clean or "unknown"


def normalize_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def stringify(value: Any) -> str:
    normalized = normalize_value(value)
    return "" if normalized is None else str(normalized)


if __name__ == "__main__":
    main()
