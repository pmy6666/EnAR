from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


@dataclass
class LlavaPreprocessResult:
    pixel_values_original: Any
    pixel_values_impression: Any
    original_image: Image.Image
    impression_image: Image.Image
    preprocess_meta: dict


class LlavaImagePreprocessor:
    def __init__(self, processor: Any, image_size: int = 336) -> None:
        self.processor = processor
        self.image_size = image_size

    def run(self, original_image: str | Path, impression_image: str | Path) -> LlavaPreprocessResult:
        original = ImageOps.exif_transpose(Image.open(original_image)).convert("RGB")
        impression = ImageOps.exif_transpose(Image.open(impression_image)).convert("RGB")
        original_inputs = self._process_image(original)
        impression_inputs = self._process_image(impression)
        pixel_values_original = original_inputs["pixel_values"]
        pixel_values_impression = impression_inputs["pixel_values"]
        meta = build_center_crop_preprocess_meta(
            original.size,
            self._infer_input_size(pixel_values_original),
            image_processor=getattr(self.processor, "image_processor", None),
        )
        return LlavaPreprocessResult(
            pixel_values_original=pixel_values_original,
            pixel_values_impression=pixel_values_impression,
            original_image=original,
            impression_image=impression,
            preprocess_meta=meta,
        )

    def _process_image(self, image: Image.Image):
        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is not None:
            return image_processor(images=image, return_tensors="pt")
        return self.processor(images=image, return_tensors="pt")

    def _infer_input_size(self, pixel_values: Any) -> tuple[int, int]:
        shape = tuple(pixel_values.shape)
        if len(shape) >= 4:
            return int(shape[-1]), int(shape[-2])
        return self.image_size, self.image_size


def build_center_crop_preprocess_meta(
    original_size: tuple[int, int],
    input_size: tuple[int, int] = (336, 336),
    image_processor: Any = None,
) -> dict:
    original_w, original_h = original_size
    input_w, input_h = input_size
    crop_size = min(original_w, original_h) 
    crop_left = max((original_w - crop_size) / 2.0, 0.0)
    crop_top = max((original_h - crop_size) / 2.0, 0.0)
    return {
        "original_size": [int(original_w), int(original_h)],
        "vision_input_size": [int(input_w), int(input_h)],
        "geometry": "center_crop_then_resize",
        "crop_box_original": [
            float(crop_left),
            float(crop_top),
            float(crop_left + crop_size),
            float(crop_top + crop_size),
        ],
        "mapping_assumption": (
            "Approximate inverse of LLaVA/CLIP shortest-edge resize plus center crop. "
            "Use explicit Envision metadata later if exact processor geometry is needed."
        ),
        "processor_size": getattr(image_processor, "size", None),
        "processor_crop_size": getattr(image_processor, "crop_size", None),
    }
