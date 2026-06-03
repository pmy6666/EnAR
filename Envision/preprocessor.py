from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image, ImageOps
from torchvision import transforms


@dataclass
class PreprocessOutput:
    image_pil: Image.Image
    image_tensor: torch.Tensor
    transform_meta: dict


class ImagePreprocessor:
    def __init__(self, image_size: int = 512) -> None:
        self.image_size = image_size

    def run(self, input_image: str | Path) -> PreprocessOutput:
        image = Image.open(input_image)
        image = ImageOps.exif_transpose(image).convert("RGB")
        original_size: Tuple[int, int] = image.size

        resized = ImageOps.fit(
            image,
            (self.image_size, self.image_size),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        tensor = transforms.ToTensor()(resized).unsqueeze(0) #[B, C, H, W], [0, 1]
        tensor = tensor * 2.0 - 1.0 # [B, C, H, W], [-1, 1]

        meta = {
            "original_size": list(original_size),
            "target_size": [self.image_size, self.image_size],
            "method": "PIL.ImageOps.fit",
            "crop": "center",
            "tensor_range": [-1.0, 1.0],
        }
        return PreprocessOutput(resized, tensor, meta)
