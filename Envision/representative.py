from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image


@dataclass
class RepresentativeOutput:
    image: Image.Image
    index: int
    diff_scores: list[float]


class RepresentativeSelector:
    def select(self, original_image: Image.Image, sample_images: Sequence[Image.Image]) -> RepresentativeOutput:
        if not sample_images:
            raise ValueError("sample_images must not be empty.")
        original = np.asarray(original_image.convert("RGB"), dtype=np.float32) / 255.0
        scores = []
        for image in sample_images:
            arr = np.asarray(
                image.convert("RGB").resize(original_image.size), dtype=np.float32
            ) / 255.0
            scores.append(float(np.linalg.norm(arr - original))) # sqrt L2 distance
        index = int(np.argmax(scores))
        return RepresentativeOutput(sample_images[index], index, scores)

    def difference_image(self, original_image: Image.Image, representative_image: Image.Image) -> Image.Image:
        original = np.asarray(original_image.convert("RGB"), dtype=np.float32)
        representative = np.asarray(representative_image.convert("RGB").resize(original_image.size), dtype=np.float32)
        diff = np.abs(representative - original)
        if diff.max() > 0:
            diff = diff / diff.max() * 255.0
        return Image.fromarray(diff.round().astype(np.uint8), mode="RGB")
