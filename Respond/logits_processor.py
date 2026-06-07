from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ContrastiveLogitsProcessor:
    def __init__(self, alpha: float = 1.0) -> None:
        if alpha < 0:
            raise ValueError("alpha must be non-negative.")
        self.alpha = float(alpha)

    def __call__(self, logits_original: Any, logits_contrastive_input: Any):
        if tuple(logits_original.shape) != tuple(logits_contrastive_input.shape):
            raise ValueError(f"logit shapes must match: {logits_original.shape} vs {logits_contrastive_input.shape}")
        return (1.0 + self.alpha) * logits_original.float() - self.alpha * logits_contrastive_input.float()


class AdaptivePlausibilityConstraint:
    def __init__(self, beta: float = 0.1) -> None:
        if not 0 <= beta <= 1:
            raise ValueError("beta must be in [0, 1].")
        self.beta = float(beta)

    def __call__(self, logits_orig, logits_contrastive):
        return self.apply(logits_orig, logits_contrastive).logits

    def apply(self, logits_orig, logits_contrastive):
        import torch

        logits_orig_float = logits_orig.float()
        if self.beta == 0:
            cutoff = torch.full_like(logits_orig_float.max(dim=-1, keepdim=True).values, -torch.inf)
        else:
            cutoff = torch.log(torch.tensor(self.beta, device=logits_orig.device, dtype=logits_orig_float.dtype))
            cutoff = cutoff + logits_orig_float.max(dim=-1, keepdim=True).values
        candidate_mask = logits_orig_float >= cutoff
        fallback_to_top = False
        if not bool(candidate_mask.any(dim=-1).all()):
            top_idx = logits_orig_float.argmax(dim=-1, keepdim=True)
            candidate_mask = candidate_mask.scatter(dim=-1, index=top_idx, value=True)
            fallback_to_top = True
        neg_inf = torch.full_like(logits_contrastive, -torch.inf)
        kept_per_batch = candidate_mask.sum(dim=-1)
        filtered_per_batch = (~candidate_mask).sum(dim=-1)
        return AdaptivePlausibilityResult(
            logits=torch.where(candidate_mask, logits_contrastive, neg_inf),
            kept_count=int(kept_per_batch.min().detach().cpu().item()),
            filtered_count=int(filtered_per_batch.max().detach().cpu().item()),
            fallback_to_top=bool(fallback_to_top),
            beta=self.beta,
            cutoff_mode="vcd_logit_cutoff",
        )


@dataclass
class AdaptivePlausibilityResult:
    logits: Any
    kept_count: int
    filtered_count: int
    fallback_to_top: bool
    beta: float
    cutoff_mode: str
