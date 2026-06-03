from __future__ import annotations

from types import SimpleNamespace

import torch


class FakeLatentDist:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.mean = tensor

    def mode(self) -> torch.Tensor:
        return self.mean


class FakeVAE:
    def __init__(self) -> None:
        self.config = SimpleNamespace(scaling_factor=0.5)

    def encode(self, image_tensor: torch.Tensor):
        latent = torch.nn.functional.avg_pool2d(image_tensor, kernel_size=8)
        return SimpleNamespace(latent_dist=FakeLatentDist(latent))

    def decode(self, latents: torch.Tensor):
        image = torch.nn.functional.interpolate(latents, scale_factor=8, mode="nearest")
        return SimpleNamespace(sample=image.clamp(-1.0, 1.0))


class FakeUNet:
    def __call__(self, latent: torch.Tensor, timestep, encoder_hidden_states=None):
        return SimpleNamespace(sample=torch.zeros_like(latent) + 0.1)


class FakeScheduler:
    def __init__(self) -> None:
        self.alphas_cumprod = torch.linspace(1.0, 0.01, 1000)
        self.timesteps = torch.tensor([])

    def set_timesteps(self, steps: int, device=None) -> None:
        self.timesteps = torch.linspace(999, 0, steps, dtype=torch.long, device=device)

    def step(self, noise_pred: torch.Tensor, timestep, sample: torch.Tensor, eta: float = 0.0):
        t = int(timestep.detach().cpu().item()) if torch.is_tensor(timestep) else int(timestep)
        prev = sample - noise_pred * (1.0 / max(t + 1, 1))
        return SimpleNamespace(prev_sample=prev)


class FakeTokenizer:
    model_max_length = 4

    def __call__(self, texts, padding, max_length, truncation, return_tensors):
        return SimpleNamespace(input_ids=torch.ones((len(texts), max_length), dtype=torch.long))


class FakeTextEncoder:
    def __call__(self, input_ids: torch.Tensor):
        return (torch.ones((input_ids.shape[0], input_ids.shape[1], 8), device=input_ids.device),)
