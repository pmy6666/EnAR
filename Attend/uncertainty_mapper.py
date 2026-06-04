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

    def map_array(self, uncertainty_map: np.ndarray) -> UncertaintyPatchResult:
        array = np.asarray(uncertainty_map, dtype=np.float32)
        if array.ndim == 3:
            array = array.mean(axis=2)
        if array.ndim != 2:
            raise ValueError(f"uncertainty_map must be 2D or image-like 3D, got {array.shape}.")

        image = Image.fromarray(_normalize_to_uint8(array), mode="L")
        resized = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        resized_array = np.asarray(resized, dtype=np.float32) / 255.0

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
        )


def load_uncertainty_map(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path)
    image = ImageOps.exif_transpose(Image.open(path)).convert("L")
    return np.asarray(image, dtype=np.float32) / 255.0


def _normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    clean = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    min_value = float(clean.min())
    max_value = float(clean.max())
    if max_value <= min_value:
        return np.zeros(clean.shape, dtype=np.uint8)
    normalized = (clean - min_value) / (max_value - min_value)
    return np.clip(np.rint(normalized * 255), 0, 255).astype(np.uint8)
