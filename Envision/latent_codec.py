from __future__ import annotations

from typing import List

import numpy as np
import torch
from PIL import Image


class LatentCodec:
    def __init__(self, vae, device: torch.device, dtype: torch.dtype) -> None:
        self.vae = vae
        self.device = device
        self.dtype = dtype
        self.scaling_factor = float(getattr(getattr(vae, "config", object()), "scaling_factor", 0.18215))

    @torch.no_grad()
    def encode(self, image_tensor: torch.Tensor) -> torch.Tensor:
        image_tensor = image_tensor.to(device=self.device, dtype=self.dtype)
        posterior = self.vae.encode(image_tensor).latent_dist
        if hasattr(posterior, "mode"):
            latent = posterior.mode()
        elif hasattr(posterior, "mean"):
            latent = posterior.mean
        else:
            raise AttributeError("VAE posterior must expose mode() or mean.")
        return latent * self.scaling_factor

    @torch.no_grad()
    def decode_tensor(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.to(device=self.device, dtype=self.dtype) / self.scaling_factor
        images = self.vae.decode(latents).sample  # [B, C, H, W], [-1, 1]
        return (images / 2.0 + 0.5).clamp(0.0, 1.0) # [B, C, H, W], [0, 1]

    @torch.no_grad()
    def decode_to_pil(self, latents: torch.Tensor) -> List[Image.Image]:
        images = self.decode_tensor(latents) # [B, C, H, W], [0, 1]
        images = images.detach().cpu().permute(0, 2, 3, 1).float().numpy() # [B, H, W, C], [0, 1]
        pil_images = []
        for image in images:
            array = (image * 255.0).round().astype(np.uint8) # [H, W, C], uint8
            pil_images.append(Image.fromarray(array)) 
        return pil_images
