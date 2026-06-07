from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image


@dataclass
class UncertaintyOutput:
    uncertainty_map: np.ndarray
    gray_image: Image.Image
    heatmap_image: Image.Image
    meta: dict[str, float | list[int]]


class UncertaintyEstimator:
    def estimate(self, sample_images: Sequence[Image.Image]) -> UncertaintyOutput:
        if not sample_images:
            raise ValueError("sample_images must not be empty.")
        arrays = [
            np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0 for img in sample_images
        ] # [0, 1]
        stack = np.stack(arrays, axis=0)  # [B, H, W, C]
        var_map = stack.var(axis=0).mean(axis=2) # [H, W]
        normalized = self._normalize(var_map) # [0, 1]
        gray = Image.fromarray((normalized * 255.0).round().astype(np.uint8), mode="L") # [0, 255]
        heatmap = self._to_heatmap(normalized) 
        return UncertaintyOutput(
            var_map.astype(np.float32),
            gray,
            heatmap,
            {
                "sample_count": int(stack.shape[0]),
                "height": int(stack.shape[1]),
                "width": int(stack.shape[2]),
                "raw_min": float(var_map.min()),
                "raw_max": float(var_map.max()),
                "raw_mean": float(var_map.mean()),
                "raw_std": float(var_map.std()),
                "normalized_min": float(normalized.min()),
                "normalized_max": float(normalized.max()),
                "normalized_mean": float(normalized.mean()),
                "normalized_std": float(normalized.std()),
            },
        )

    @staticmethod
    def _normalize(array: np.ndarray) -> np.ndarray:
        min_value = float(array.min())
        max_value = float(array.max())
        if max_value - min_value < 1e-12:
            return np.zeros_like(array, dtype=np.float32)
        return ((array - min_value) / (max_value - min_value)).astype(np.float32)

    @staticmethod
    def _to_heatmap(normalized: np.ndarray) -> Image.Image:
        x = np.clip(normalized, 0.0, 1.0)
        r = np.clip(4.0 * x - 1.0, 0.0, 1.0)
        g = np.clip(1.0 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
        b = np.clip(1.0 - 4.0 * x, 0.0, 1.0)
        heatmap = np.stack([r, g, b], axis=2)
        return Image.fromarray((heatmap * 255.0).round().astype(np.uint8), mode="RGB")
