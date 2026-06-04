from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .array_utils import infer_square_grid, minmax_normalize


@dataclass
class ContrastiveAttentionResult:
    delta_scores: np.ndarray
    delta_grid: np.ndarray
    normalized_grid: np.ndarray


class ContrastiveAttentionComputer:
    def compute(
        self,
        attention_scores_original: np.ndarray,
        attention_scores_impression: np.ndarray,
    ) -> ContrastiveAttentionResult:
        original = np.asarray(attention_scores_original, dtype=np.float32).reshape(-1)
        impression = np.asarray(attention_scores_impression, dtype=np.float32).reshape(-1)
        if original.shape != impression.shape:
            raise ValueError(
                "attention score shapes must match: "
                f"{original.shape} vs {impression.shape}"
            )
        delta = np.abs(original - impression)
        grid_h, grid_w = infer_square_grid(delta.size)
        delta_grid = delta.reshape(grid_h, grid_w)
        return ContrastiveAttentionResult(
            delta_scores=delta,
            delta_grid=delta_grid,
            normalized_grid=minmax_normalize(delta_grid),
        )
