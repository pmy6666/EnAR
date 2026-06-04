from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .array_utils import infer_square_grid, minmax_normalize


@dataclass
class TokenSelectionResult:
    h_attn: list[int]
    h_unc: list[int]
    h_union_raw: list[int]
    h_final: list[int]
    vision_token_indices: list[int]
    union_patch_mask_grid: np.ndarray
    score: np.ndarray


class CounterfactualTokenSelector:
    def select(
        self,
        delta_attention_scores: np.ndarray,
        uncertainty_patch_scores: np.ndarray,
        attention_top_ratio: float = 0.10,
        uncertainty_top_ratio: float = 0.05,
        padding_ratio_limit: float = 0.10,
        uncertainty_weight: float = 1.0,
        has_cls_token: bool = True,
    ) -> TokenSelectionResult:
        delta = np.asarray(delta_attention_scores, dtype=np.float32).reshape(-1)
        uncertainty = np.asarray(uncertainty_patch_scores, dtype=np.float32).reshape(-1)
        if delta.shape != uncertainty.shape:
            raise ValueError(f"score shapes must match: {delta.shape} vs {uncertainty.shape}")
        if delta.size == 0:
            raise ValueError("score arrays cannot be empty.")
        for name, ratio in (
            ("attention_top_ratio", attention_top_ratio),
            ("uncertainty_top_ratio", uncertainty_top_ratio),
            ("padding_ratio_limit", padding_ratio_limit),
        ):
            if not 0 < ratio <= 1:
                raise ValueError(f"{name} must be in (0, 1].")

        num_patches = delta.size
        attn_k = max(1, math.ceil(num_patches * attention_top_ratio))
        unc_k = max(1, math.ceil(num_patches * uncertainty_top_ratio))
        padding_limit = max(1, math.ceil(num_patches * padding_ratio_limit))

        h_attn = _topk_indices(delta, attn_k)
        h_unc = _topk_indices(uncertainty, unc_k)
        union_raw = sorted(set(h_attn) | set(h_unc))

        combined_score = minmax_normalize(delta) + uncertainty_weight * minmax_normalize(uncertainty)
        if len(union_raw) > padding_limit:
            union_scores = np.asarray([combined_score[idx] for idx in union_raw], dtype=np.float32)
            keep_order = np.argsort(-union_scores, kind="stable")[:padding_limit]
            h_final = sorted(union_raw[int(i)] for i in keep_order)
        else:
            h_final = union_raw

        grid_h, grid_w = infer_square_grid(num_patches)
        mask = np.zeros(num_patches, dtype=bool)
        mask[h_final] = True
        offset = 1 if has_cls_token else 0
        return TokenSelectionResult(
            h_attn=sorted(h_attn),
            h_unc=sorted(h_unc),
            h_union_raw=union_raw,
            h_final=h_final,
            vision_token_indices=[idx + offset for idx in h_final],
            union_patch_mask_grid=mask.reshape(grid_h, grid_w),
            score=combined_score,
        )


def _topk_indices(values: np.ndarray, k: int) -> list[int]:
    order = np.argsort(-values, kind="stable")
    return [int(idx) for idx in order[:k]]
