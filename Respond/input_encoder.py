from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass
class EncodedMultimodalInput:
    input_ids: Any
    attention_mask: Any
    pixel_values: Any
    prompt_len: int
    image_token_positions: list[int]
    meta: dict[str, Any]


class MultimodalInputEncoder:
    def __init__(self, processor: Any, device: Any) -> None:
        self.processor = processor
        self.device = device

    def encode(self, image_path: str | Path, prompt: str, image_token_index: int | None = None) -> EncodedMultimodalInput:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        pixel_values = inputs["pixel_values"].to(self.device)
        positions = find_image_token_positions(input_ids, image_token_index)
        return EncodedMultimodalInput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            prompt_len=int(input_ids.shape[-1]),
            image_token_positions=positions,
            meta={
                "image_path": str(image_path),
                "prompt_len": int(input_ids.shape[-1]),
                "image_token_positions": positions,
                "input_ids_shape": [int(dim) for dim in input_ids.shape],
                "attention_mask_shape": [int(dim) for dim in attention_mask.shape] if attention_mask is not None else None,
                "pixel_values_shape": [int(dim) for dim in pixel_values.shape],
            },
        )


def find_image_token_positions(input_ids, image_token_index: int | None) -> list[int]:
    if image_token_index is None:
        return []
    matches = (input_ids[0] == image_token_index).nonzero(as_tuple=False)
    return [int(item.item()) for item in matches.flatten()]
