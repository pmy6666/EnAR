from __future__ import annotations

from dataclasses import dataclass

import torch

from .ddim_inverter import DDIMInverter
from .latent_codec import LatentCodec


@dataclass
class DDIMReconstructionOutput:
    latent: torch.Tensor
    image: object


class DDIMReconstructor:
    def __init__(self, unet, scheduler, latent_codec: LatentCodec, guidance_scale: float = 1.0) -> None:
        self.unet = unet
        self.scheduler = scheduler
        self.latent_codec = latent_codec
        self.guidance_scale = guidance_scale
        self._noise_helper = DDIMInverter(unet, scheduler, guidance_scale)

    @torch.no_grad()
    def reconstruct(
        self,
        zT: torch.Tensor,
        step_index_T: int,
        text_embeddings: torch.Tensor,
        negative_text_embeddings: torch.Tensor | None = None,
    ) -> DDIMReconstructionOutput:
        timesteps = list(self.scheduler.timesteps)
        start_index = len(timesteps) - 1 - step_index_T
        selected = timesteps[start_index:-1]

        latent = zT
        for timestep in selected:
            noise_pred = self._noise_helper._predict_noise(latent, timestep, text_embeddings, negative_text_embeddings)
            latent = self.scheduler.step(noise_pred, timestep, latent, eta=0.0).prev_sample

        image = self.latent_codec.decode_to_pil(latent)[0]
        return DDIMReconstructionOutput(latent=latent, image=image)
