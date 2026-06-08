from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import torch
from PIL import Image, ImageOps
from torchvision import transforms


@dataclass
class PreprocessOutput:
    image_pil: Image.Image
    image_tensor: torch.Tensor
    transform_meta: dict


class ImagePreprocessor:
    def __init__(
        self,
        image_size: int = 512,
        preprocess_mode: str = "pad",
        pad_color: Sequence[int] = (127, 127, 127),
    ) -> None:
        self.image_size = image_size
        self.preprocess_mode = preprocess_mode
        self.pad_color = tuple(int(value) for value in pad_color)

    def run(self, input_image: str | Path) -> PreprocessOutput:
        image = Image.open(input_image)
        image = ImageOps.exif_transpose(image).convert("RGB")
        original_size: Tuple[int, int] = image.size

        if self.preprocess_mode == "center_crop":
            image_pil, meta = self._center_crop_resize(image, original_size)
        elif self.preprocess_mode == "pad":
            image_pil, meta = self._resize_keep_aspect_pad(image, original_size)
        else:
            raise ValueError(f"Unsupported preprocess_mode: {self.preprocess_mode}")

        tensor = transforms.ToTensor()(image_pil).unsqueeze(0)
        tensor = tensor * 2.0 - 1.0
        meta["tensor_range"] = [-1.0, 1.0]
        return PreprocessOutput(image_pil, tensor, meta)

    def _center_crop_resize(self, image: Image.Image, original_size: Tuple[int, int]) -> tuple[Image.Image, dict]:
        original_w, original_h = original_size
        crop_size = min(original_w, original_h)
        crop_left = max((original_w - crop_size) / 2.0, 0.0)
        crop_top = max((original_h - crop_size) / 2.0, 0.0)
        image_pil = ImageOps.fit(
            image,
            (self.image_size, self.image_size),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        return image_pil, {
            "original_size": [original_w, original_h],
            "target_size": [self.image_size, self.image_size],
            "geometry": "center_crop_then_resize",
            "method": "PIL.ImageOps.fit",
            "crop": "center",
            "crop_box_original": [
                float(crop_left),
                float(crop_top),
                float(crop_left + crop_size),
                float(crop_top + crop_size),
            ],
            "content_box_target": [0, 0, self.image_size, self.image_size],
            "scale": self.image_size / float(crop_size),
        }

    def _resize_keep_aspect_pad(self, image: Image.Image, original_size: Tuple[int, int]) -> tuple[Image.Image, dict]:
        original_w, original_h = original_size
        scale = self.image_size / float(max(original_w, original_h))
        resized_w = max(1, int(round(original_w * scale)))
        resized_h = max(1, int(round(original_h * scale)))
        resized = image.resize((resized_w, resized_h), Image.Resampling.LANCZOS)

        pad_left = (self.image_size - resized_w) // 2
        pad_top = (self.image_size - resized_h) // 2
        pad_right = self.image_size - resized_w - pad_left
        pad_bottom = self.image_size - resized_h - pad_top
        canvas = Image.new("RGB", (self.image_size, self.image_size), self.pad_color)
        canvas.paste(resized, (pad_left, pad_top))

        return canvas, {
            "original_size": [original_w, original_h],
            "target_size": [self.image_size, self.image_size],
            "geometry": "resize_keep_aspect_then_pad",
            "method": "PIL.resize + centered RGB padding",
            "scale": scale,
            "resized_size": [resized_w, resized_h],
            "pad": {
                "left": pad_left,
                "top": pad_top,
                "right": pad_right,
                "bottom": pad_bottom,
            },
            "pad_color": list(self.pad_color),
            "content_box_target": [
                pad_left,
                pad_top,
                pad_left + resized_w,
                pad_top + resized_h,
            ],
        }
