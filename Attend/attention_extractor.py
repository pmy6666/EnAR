from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class VisionAttentionResult:
    attention_scores: np.ndarray
    raw_attention: Any
    token_layout_meta: dict


class VisionAttentionExtractor:
    def __init__(self, vision_tower: Any, device: Any = None) -> None:
        self.vision_tower = vision_tower
        self.device = device

    def extract(self, pixel_values: Any, vision_layer_number: int) -> VisionAttentionResult:
        if vision_layer_number < 1:
            raise ValueError("vision_layer_number uses 1-based indexing and must be >= 1.")
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("torch is required for attention extraction.") from exc

        if self.device is not None:
            pixel_values = pixel_values.to(self.device)
        layer_index = vision_layer_number - 1
        with torch.inference_mode():
            outputs = self.vision_tower(
                pixel_values=pixel_values,
                output_attentions=True,
                return_dict=True,
            )
        attentions = outputs.attentions # [batch, heads, tokens, tokens]
        if layer_index >= len(attentions):
            raise ValueError(
                f"vision_layer_number={vision_layer_number} exceeds available layers={len(attentions)}."
            )
        raw_attention = attentions[layer_index]
        scores, meta = attention_tensor_to_patch_scores(raw_attention)
        meta["vision_layer_number"] = vision_layer_number
        meta["vision_layer_index"] = layer_index
        return VisionAttentionResult(scores, raw_attention, meta)


def attention_tensor_to_patch_scores(raw_attention: Any) -> tuple[np.ndarray, dict]:
    try:
        import torch
    except Exception:
        torch = None

    attn = raw_attention
    if torch is not None and isinstance(attn, torch.Tensor):
        attn_np = attn.detach().float().cpu().numpy() # [batch, heads, tokens, tokens]
    else:
        attn_np = np.asarray(attn, dtype=np.float32)
    if attn_np.ndim != 4:
        raise ValueError(f"raw attention must have shape [batch, heads, tokens, tokens], got {attn_np.shape}.")
    if attn_np.shape[0] != 1:
        attn_np = attn_np[:1] # [batch, heads, tokens, tokens] -> [1, heads, tokens, tokens]

    mean_attention = attn_np[0].mean(axis=0) # [heads, tokens, tokens] -> [tokens, tokens]
    token_count = mean_attention.shape[-1]
    has_cls_token = token_count == 577 or _is_square(token_count - 1)
    if has_cls_token:
        scores = mean_attention[0, 1:] # [tokens - 1]
    elif _is_square(token_count):
        # Fallback when a caller provides patch-only attention.
        scores = mean_attention.mean(axis=0)
    else:
        raise ValueError(f"Cannot infer patch layout from token_count={token_count}.")

    grid_side = int(round(scores.size ** 0.5))
    meta = {
        "token_count": int(token_count),
        "has_cls_token": bool(has_cls_token),
        "num_patches": int(scores.size),
        "patch_grid": [grid_side, grid_side],
    }
    return scores.astype(np.float32), meta


def _is_square(value: int) -> bool:
    if value <= 0:
        return False
    side = int(round(value ** 0.5))
    return side * side == value
