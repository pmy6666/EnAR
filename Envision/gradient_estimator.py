from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GradientEstimate:
    gradient: torch.Tensor
    noise_pred: torch.Tensor
    gradient_norm: float


class TweedieGradientEstimator:
    def __init__(self, unet, scheduler, guidance_scale: float = 1.0, clip_norm: float | None = None) -> None:
        self.unet = unet
        self.scheduler = scheduler
        self.guidance_scale = guidance_scale
        self.clip_norm = clip_norm

    @torch.no_grad()
    def estimate(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor | int,
        text_embeddings: torch.Tensor,
        negative_text_embeddings: torch.Tensor | None = None,
    ) -> GradientEstimate:
        timestep_tensor = timestep if torch.is_tensor(timestep) else torch.tensor(timestep, device=latent.device, dtype=torch.long)
        noise_pred = self._predict_noise(latent, timestep_tensor, text_embeddings, negative_text_embeddings)
        timestep_index = int(timestep_tensor.detach().cpu().item())
        alpha = self.scheduler.alphas_cumprod[timestep_index].to(latent.device, latent.dtype)  # alpha = alpha1 * alpha2 * ... * alpha_t
        sigma = (1.0 - alpha).sqrt().clamp_min(1e-6)  # sigma = sqrt(1 - alpha)
        gradient = -noise_pred / sigma  
        norm = float(gradient.float().norm().detach().cpu())
        if self.clip_norm is not None and norm > self.clip_norm:
            gradient = gradient * (self.clip_norm / (norm + 1e-12))
            norm = float(gradient.float().norm().detach().cpu())
        return GradientEstimate(gradient=gradient, noise_pred=noise_pred, gradient_norm=norm)

    def _predict_noise(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        text_embeddings: torch.Tensor,
        negative_text_embeddings: torch.Tensor | None,
    ) -> torch.Tensor:
        if negative_text_embeddings is None or self.guidance_scale == 1.0:
            return self.unet(latent, timestep, encoder_hidden_states=text_embeddings).sample
        latents = torch.cat([latent, latent], dim=0)
        embeddings = torch.cat([negative_text_embeddings, text_embeddings], dim=0)
        noise_uncond, noise_text = self.unet(latents, timestep, encoder_hidden_states=embeddings).sample.chunk(2)
        return noise_uncond + self.guidance_scale * (noise_text - noise_uncond)
