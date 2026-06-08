from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


@dataclass
class UncertaintyPatchResult:
    patch_scores: np.ndarray
    patch_grid: np.ndarray
    resized_map: np.ndarray
    meta: dict[str, float | list[int]]


class UncertaintyPatchMapper:
    def __init__(self, image_size: int = 336, patch_size: int = 14) -> None:
        if image_size <= 0 or patch_size <= 0:
            raise ValueError("image_size and patch_size must be positive.")
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")
        self.image_size = image_size
        self.patch_size = patch_size

    def map_file(self, path: str | Path) -> UncertaintyPatchResult:
        return self.map_array(load_uncertainty_map(path))

    def map_file_with_geometry(
        self,
        path: str | Path,
        envision_meta: dict | None,
        llava_preprocess_meta: dict | None,
    ) -> UncertaintyPatchResult:
        array = load_uncertainty_map(path)
        if not envision_meta or not llava_preprocess_meta:
            return self.map_array(array)
        transform_meta = envision_meta.get("transform_meta", {})
        if _llava_input_is_envision_processed_image(transform_meta, llava_preprocess_meta):
            projected = _original_map_to_llava_input(array, llava_preprocess_meta)
            alignment = "envision_processed_map_to_llava_input"
        else:
            projected = project_envision_map_to_llava_input(
                array,
                transform_meta,
                llava_preprocess_meta,
            )
            alignment = "envision_map_to_original_to_llava_input"
        result = self.map_array(projected)
        result.meta["geometry_alignment"] = alignment
        result.meta["projected_shape"] = [int(projected.shape[0]), int(projected.shape[1])]
        result.meta["envision_geometry"] = transform_meta.get("geometry")
        result.meta["llava_geometry"] = llava_preprocess_meta.get("geometry")
        return result

    def map_array(self, uncertainty_map: np.ndarray) -> UncertaintyPatchResult:
        array = np.asarray(uncertainty_map, dtype=np.float32)
        if array.ndim == 3:
            array = array.mean(axis=2)
        if array.ndim != 2:
            raise ValueError(f"uncertainty_map must be 2D or image-like 3D, got {array.shape}.")

        raw_shape = [int(array.shape[0]), int(array.shape[1])]
        raw_min = float(np.nanmin(array))
        raw_max = float(np.nanmax(array))
        raw_mean = float(np.nanmean(array))
        raw_std = float(np.nanstd(array))

        normalized = _minmax_normalize_float(array)
        # Keep the map in float space while resizing. Quantizing to uint8 before
        # patch aggregation makes Eq.6 brittle when the variance range is tiny.
        image = Image.fromarray(normalized, mode="F")
        resized = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        resized_array = np.asarray(resized, dtype=np.float32)

        grid_side = self.image_size // self.patch_size
        patch_grid = resized_array.reshape(
            grid_side,
            self.patch_size,
            grid_side,
            self.patch_size,
        ).mean(axis=(1, 3))
        return UncertaintyPatchResult(
            patch_scores=patch_grid.reshape(-1).astype(np.float32),
            patch_grid=patch_grid.astype(np.float32),
            resized_map=resized_array.astype(np.float32),
            meta={
                "raw_shape": raw_shape,
                "raw_min": raw_min,
                "raw_max": raw_max,
                "raw_mean": raw_mean,
                "raw_std": raw_std,
                "normalized_min": float(normalized.min()),
                "normalized_max": float(normalized.max()),
                "patch_min": float(patch_grid.min()),
                "patch_max": float(patch_grid.max()),
                "patch_mean": float(patch_grid.mean()),
                "patch_std": float(patch_grid.std()),
            },
        )


def load_uncertainty_map(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path)
    image = ImageOps.exif_transpose(Image.open(path)).convert("L")
    return np.asarray(image, dtype=np.float32) / 255.0


def project_envision_map_to_llava_input(
    uncertainty_map: np.ndarray,
    envision_transform_meta: dict,
    llava_preprocess_meta: dict,
) -> np.ndarray:
    array = np.asarray(uncertainty_map, dtype=np.float32)
    if array.ndim == 3:
        array = array.mean(axis=2)
    if array.ndim != 2:
        raise ValueError(f"uncertainty_map must be 2D or image-like 3D, got {array.shape}.")

    original_w, original_h = _original_size(envision_transform_meta, llava_preprocess_meta)
    origin_map = _envision_map_to_original(array, envision_transform_meta, original_w, original_h)
    return _original_map_to_llava_input(origin_map, llava_preprocess_meta)


def _envision_map_to_original(
    array: np.ndarray,
    transform_meta: dict,
    original_w: int,
    original_h: int,
) -> np.ndarray:
    geometry = transform_meta.get("geometry")
    if geometry == "resize_keep_aspect_then_pad" and "content_box_target" in transform_meta:
        left, top, right, bottom = [int(round(float(v))) for v in transform_meta["content_box_target"]]
        left = max(0, min(left, array.shape[1]))
        right = max(left + 1, min(right, array.shape[1]))
        top = max(0, min(top, array.shape[0]))
        bottom = max(top + 1, min(bottom, array.shape[0]))
        content = array[top:bottom, left:right]
        return np.asarray(
            Image.fromarray(content, mode="F").resize((original_w, original_h), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )

    crop_box = transform_meta.get("crop_box_original")
    if crop_box is None:
        crop = min(original_w, original_h)
        crop_box = [
            max((original_w - crop) / 2.0, 0.0),
            max((original_h - crop) / 2.0, 0.0),
            max((original_w - crop) / 2.0, 0.0) + crop,
            max((original_h - crop) / 2.0, 0.0) + crop,
        ]
    left, top, right, bottom = [int(round(float(v))) for v in crop_box]
    crop_w = max(1, right - left)
    crop_h = max(1, bottom - top)
    crop_map = Image.fromarray(array, mode="F").resize((crop_w, crop_h), Image.Resampling.BILINEAR)
    canvas = Image.new("F", (original_w, original_h), 0.0)
    canvas.paste(crop_map, (left, top))
    return np.asarray(canvas, dtype=np.float32)


def _original_map_to_llava_input(origin_map: np.ndarray, preprocess_meta: dict) -> np.ndarray:
    input_w, input_h = _input_size(preprocess_meta)
    crop_box = preprocess_meta.get("crop_box_original")
    if crop_box is None:
        original_h, original_w = origin_map.shape
        crop = min(original_w, original_h)
        crop_box = [
            max((original_w - crop) / 2.0, 0.0),
            max((original_h - crop) / 2.0, 0.0),
            max((original_w - crop) / 2.0, 0.0) + crop,
            max((original_h - crop) / 2.0, 0.0) + crop,
        ]
    left, top, right, bottom = [int(round(float(v))) for v in crop_box]
    left = max(0, min(left, origin_map.shape[1] - 1))
    right = max(left + 1, min(right, origin_map.shape[1]))
    top = max(0, min(top, origin_map.shape[0] - 1))
    bottom = max(top + 1, min(bottom, origin_map.shape[0]))
    cropped = origin_map[top:bottom, left:right]
    return np.asarray(
        Image.fromarray(cropped, mode="F").resize((input_w, input_h), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )


def _original_size(envision_meta: dict, llava_meta: dict) -> tuple[int, int]:
    value = envision_meta.get("original_size") or llava_meta.get("original_size")
    if not value:
        raise ValueError("original_size is required for geometry-aware uncertainty mapping.")
    return int(value[0]), int(value[1])


def _input_size(preprocess_meta: dict) -> tuple[int, int]:
    value = preprocess_meta.get("vision_input_size", [336, 336])
    return int(value[0]), int(value[1])


def _llava_input_is_envision_processed_image(envision_meta: dict, llava_meta: dict) -> bool:
    target_size = envision_meta.get("target_size")
    llava_original_size = llava_meta.get("original_size")
    if not target_size or not llava_original_size:
        return False
    return [int(target_size[0]), int(target_size[1])] == [
        int(llava_original_size[0]),
        int(llava_original_size[1]),
    ]


def _minmax_normalize_float(array: np.ndarray) -> np.ndarray:
    clean = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    min_value = float(clean.min())
    max_value = float(clean.max())
    if max_value <= min_value:
        return np.zeros(clean.shape, dtype=np.float32)
    normalized = (clean - min_value) / (max_value - min_value)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)
