from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from .gradient_estimator import TweedieGradientEstimator


@dataclass
class LangevinOutput:
    latent: torch.Tensor
    debug_steps: List[dict]


class LangevinPerturber:
    def __init__(self, gradient_estimator: TweedieGradientEstimator) -> None:
        self.gradient_estimator = gradient_estimator

    @torch.no_grad()
    def perturb(
        self,
        zT: torch.Tensor,
        timestep: torch.Tensor | int,
        text_embeddings: torch.Tensor,
        negative_text_embeddings: torch.Tensor | None,
        steps: int,
        eta_start: float,
        eta_end: float,
        temperature_tau: float,
        seed: int,
    ) -> LangevinOutput:
        latent = zT.detach().clone()
        generator = torch.Generator(device=latent.device).manual_seed(seed)
        debug_steps = []

        for idx in range(steps):
            eta = self._annealed_eta(idx, steps, eta_start, eta_end)
            estimate = self.gradient_estimator.estimate(
                latent, timestep, text_embeddings, negative_text_embeddings
            )
            noise = torch.randn(
                latent.shape, generator=generator, device=latent.device, dtype=latent.dtype
            )
            delta = eta * estimate.gradient + (eta * temperature_tau) ** 0.5 * noise
            latent = latent + delta
            debug_steps.append(
                {
                    "step": idx,
                    "eta": eta,
                    "gradient_norm": estimate.gradient_norm,
                    "noise_norm": float(noise.float().norm().detach().cpu()),
                    "latent_delta_norm": float(delta.float().norm().detach().cpu()),
                }
            )

        return LangevinOutput(latent=latent, debug_steps=debug_steps)

    @staticmethod
    def _annealed_eta(idx: int, steps: int, eta_start: float, eta_end: float) -> float:
        if steps <= 1:
            return eta_end
        ratio = idx / float(steps - 1)
        return eta_start + (eta_end - eta_start) * ratio
