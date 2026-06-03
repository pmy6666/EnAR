from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline


@dataclass
class StableDiffusionComponents:
    vae: object
    unet: object
    tokenizer: object
    text_encoder: object
    scheduler: DDIMScheduler
    device: torch.device
    dtype: torch.dtype


class StableDiffusionLoader:
    def __init__(
        self,
        model_dir: str | Path,
        dtype: str = "float16",
        device: Optional[str] = None,
        disable_safety_checker: bool = True,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.dtype = self._resolve_dtype(dtype)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cpu" and self.dtype == torch.float16:
            self.dtype = torch.float32
        self.disable_safety_checker = disable_safety_checker

    @staticmethod
    def _resolve_dtype(dtype: str) -> torch.dtype:
        return {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }[dtype]

    def load(self) -> StableDiffusionComponents:
        kwargs = {
            "torch_dtype": self.dtype,
            "local_files_only": True,
            "use_safetensors": True,
        }
        if self.disable_safety_checker:
            kwargs["safety_checker"] = None
            kwargs["requires_safety_checker"] = False

        pipe = StableDiffusionPipeline.from_pretrained(str(self.model_dir), **kwargs)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)

        for module in (pipe.vae, pipe.unet, pipe.text_encoder):
            module.eval()
            module.requires_grad_(False)

        return StableDiffusionComponents(
            vae=pipe.vae,
            unet=pipe.unet,
            tokenizer=pipe.tokenizer,
            text_encoder=pipe.text_encoder,
            scheduler=pipe.scheduler,
            device=self.device,
            dtype=self.dtype,
        )
