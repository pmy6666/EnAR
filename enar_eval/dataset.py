from __future__ import annotations

import glob
import io
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from .config import DatasetConfig


REQUIRED_COLUMNS = {
    "image",
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
}


@dataclass
class VLMBiasSample:
    sample_id: str
    raw_id: str
    subset: str
    image: Any
    image_path: str
    topic: str
    sub_topic: str
    prompt: str
    ground_truth: str
    expected_bias: str
    with_title: bool
    type_of_question: str
    pixel: int | None
    metadata: dict[str, Any]
    row_index: int

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "raw_id": self.raw_id,
            "subset": self.subset,
            "split": self.subset,
            "image_path": self.image_path,
            "topic": self.topic,
            "sub_topic": self.sub_topic,
            "prompt": self.prompt,
            "ground_truth": self.ground_truth,
            "expected_bias": self.expected_bias,
            "with_title": self.with_title,
            "type_of_question": self.type_of_question,
            "pixel": self.pixel,
            "metadata": self.metadata,
            "row_index": self.row_index,
        }


def load_vlmbias_samples(config: DatasetConfig) -> tuple[list[VLMBiasSample], dict[str, Any]]:
    files = resolve_data_files(config)
    frames = [pd.read_parquet(path) for path in files]
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"VLMBias parquet is missing required columns: {missing}")

    original_count = len(df)
    df = _apply_filters(df, config.filters)
    if config.filters.max_samples is not None:
        df = df.head(int(config.filters.max_samples))

    samples: list[VLMBiasSample] = []
    seen_ids: dict[str, int] = {}
    for row_index, row in df.reset_index(drop=True).iterrows():
        raw_id = _stringify(row.get("ID")) or f"row_{row_index:06d}"
        sample_id = sanitize_sample_id(raw_id)
        seen = seen_ids.get(sample_id, 0)
        seen_ids[sample_id] = seen + 1
        if seen:
            sample_id = f"{sample_id}_{seen}"
        metadata = parse_metadata(row.get("metadata"))
        pixel = row.get("pixel")
        pixel_value = None if pd.isna(pixel) else int(pixel)
        samples.append(
            VLMBiasSample(
                sample_id=sample_id,
                raw_id=raw_id,
                subset=config.subset,
                image=row.get("image"),
                image_path=_stringify(row.get("image_path")),
                topic=_stringify(row.get("topic")),
                sub_topic=_stringify(row.get("sub_topic")),
                prompt=_stringify(row.get("prompt")),
                ground_truth=_stringify(row.get("ground_truth")),
                expected_bias=_stringify(row.get("expected_bias")),
                with_title=bool(row.get("with_title")),
                type_of_question=_stringify(row.get("type_of_question")),
                pixel=pixel_value,
                metadata=metadata,
                row_index=int(row_index),
            )
        )

    manifest = {
        "dataset": config.name,
        "subset": config.subset,
        "split": config.subset,
        "files": [str(path) for path in files],
        "num_rows_before_filter": original_count,
        "num_rows_after_filter": len(samples),
        "categories": config.categories,
        "filters": {
            "max_samples": config.filters.max_samples,
            "sample_ids": config.filters.sample_ids,
            "topics": config.filters.topics,
            "sub_topics": config.filters.sub_topics,
            "type_of_question": config.filters.type_of_question,
            "with_title": config.filters.with_title,
            "pixel": config.filters.pixel,
        },
    }
    return samples, manifest


def resolve_data_files(config: DatasetConfig) -> list[Path]:
    pattern = config.data_files.get(config.subset)
    if not pattern:
        raise ValueError(f"Unknown VLMBias subset '{config.subset}'. Known subsets: {sorted(config.data_files)}")
    base = config.root_dir if not Path(pattern).is_absolute() else Path("/")
    matches = sorted(Path(path) for path in glob.glob(str(base / pattern)))
    if not matches:
        data_pattern = config.data_dir / f"{config.subset}-*.parquet"
        matches = sorted(Path(path) for path in glob.glob(str(data_pattern)))
    if not matches:
        raise FileNotFoundError(f"No parquet files found for subset '{config.subset}' with pattern '{pattern}'")
    return matches


def export_sample_image(sample: VLMBiasSample, output_path: str | Path, dataset_root: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = sample.image

    if isinstance(image, Image.Image):
        image.save(output_path)
        return output_path

    if isinstance(image, (bytes, bytearray, memoryview)):
        Image.open(io.BytesIO(bytes(image))).save(output_path)
        return output_path

    if isinstance(image, dict):
        data = image.get("bytes")
        if data is not None:
            Image.open(io.BytesIO(data)).save(output_path)
            return output_path
        path_value = image.get("path")
        if path_value:
            return _copy_or_convert_image(Path(path_value), output_path, dataset_root)

    if sample.image_path:
        return _copy_or_convert_image(Path(sample.image_path), output_path, dataset_root)

    raise TypeError(f"Unsupported image value for sample {sample.sample_id}: {type(image)!r}")


def parse_metadata(value: Any) -> dict[str, Any]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return {"raw": value}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    return {"value": value}


def sanitize_sample_id(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    clean = clean.strip("._")
    return clean[:120] or "sample"


def _apply_filters(df: pd.DataFrame, filters: Any) -> pd.DataFrame:
    out = df
    if filters.sample_ids:
        wanted = {str(item) for item in filters.sample_ids}
        out = out[out["ID"].astype(str).isin(wanted)]
    if filters.topics:
        wanted = {str(item).lower() for item in filters.topics}
        out = out[out["topic"].astype(str).str.lower().isin(wanted)]
    if filters.sub_topics:
        wanted = {str(item).lower() for item in filters.sub_topics}
        out = out[out["sub_topic"].astype(str).str.lower().isin(wanted)]
    if filters.type_of_question:
        wanted = {str(item).lower() for item in filters.type_of_question}
        out = out[out["type_of_question"].astype(str).str.lower().isin(wanted)]
    if filters.with_title is not None:
        out = out[out["with_title"].astype(bool) == bool(filters.with_title)]
    if filters.pixel is not None:
        out = out[out["pixel"].astype("Int64") == int(filters.pixel)]
    return out


def _copy_or_convert_image(path: Path, output_path: Path, dataset_root: str | Path) -> Path:
    if not path.is_absolute():
        path = Path(dataset_root) / path
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {path}")
    if output_path.suffix.lower() == path.suffix.lower():
        shutil.copyfile(path, output_path)
    else:
        Image.open(path).save(output_path)
    return output_path


def _stringify(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)
