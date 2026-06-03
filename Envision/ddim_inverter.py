from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


@dataclass
class DDIMInversionOutput:
    zT: torch.Tensor
    timestep_T: int
    step_index_T: int
    trajectory: List[torch.Tensor] | None = None


class DDIMInverter:
    def __init__(self, unet, scheduler, guidance_scale: float = 1.0) -> None:
        self.unet = unet
        self.scheduler = scheduler
        self.guidance_scale = guidance_scale

    @torch.no_grad()
    def invert(
        self,
        z0: torch.Tensor,
        text_embeddings: torch.Tensor,
        num_ddim_steps: int,
        inversion_step_T: int,
        negative_text_embeddings: torch.Tensor | None = None,
        return_trajectory: bool = False,
    ) -> DDIMInversionOutput:
        self.scheduler.set_timesteps(num_ddim_steps, device=z0.device)
        timesteps = list(self.scheduler.timesteps)
        reverse_timesteps = list(reversed(timesteps))
        if inversion_step_T == 0:
            return DDIMInversionOutput(z0, int(timesteps[-1].item()), 0, [z0] if return_trajectory else None)

        latent = z0
        trajectory = [z0] if return_trajectory else None
        completed_steps = 0
        for source_t, target_t in zip(reverse_timesteps[:-1], reverse_timesteps[1:]):
            noise_pred = self._predict_noise(latent, source_t, text_embeddings, negative_text_embeddings)
            latent = self._forward_step(latent, noise_pred, source_t, target_t)
            completed_steps += 1
            if trajectory is not None:
                trajectory.append(latent.detach().clone())
            if completed_steps >= inversion_step_T:
                break

        step_index_T = min(inversion_step_T, len(timesteps) - 1)
        timestep_T = int(timesteps[-1 - step_index_T].item())
        return DDIMInversionOutput(latent, timestep_T, step_index_T, trajectory)

    def _predict_noise(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        text_embeddings: torch.Tensor,
        negative_text_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if negative_text_embeddings is None or self.guidance_scale == 1.0:
            return self.unet(latent, timestep, encoder_hidden_states=text_embeddings).sample
        latents = torch.cat([latent, latent], dim=0)
        embeddings = torch.cat([negative_text_embeddings, text_embeddings], dim=0)
        noise_uncond, noise_text = self.unet(latents, timestep, encoder_hidden_states=embeddings).sample.chunk(2)
        return noise_uncond + self.guidance_scale * (noise_text - noise_uncond)

    def _forward_step(self, latent: torch.Tensor, noise_pred: torch.Tensor, source_t: torch.Tensor, target_t: torch.Tensor) -> torch.Tensor:
        source_index = int(source_t.detach().cpu().item()) if torch.is_tensor(source_t) else int(source_t)
        target_index = int(target_t.detach().cpu().item()) if torch.is_tensor(target_t) else int(target_t)
        alpha_source = self.scheduler.alphas_cumprod[source_index].to(latent.device, latent.dtype)
        alpha_target = self.scheduler.alphas_cumprod[target_index].to(latent.device, latent.dtype)
        pred_x0 = (latent - (1.0 - alpha_source).sqrt() * noise_pred) / alpha_source.sqrt()
        direction = (1.0 - alpha_target).sqrt() * noise_pred
        return alpha_target.sqrt() * pred_x0 + direction
