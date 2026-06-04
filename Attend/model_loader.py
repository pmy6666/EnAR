from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class LlavaVisionComponents:
    processor: Any
    model: Any
    vision_tower: Any
    vision_config: Any
    device: Any
    dtype: Any
    image_size: int
    patch_size: int
    patch_grid: tuple[int, int]


class LlavaVisionLoader:
    def __init__(
        self,
        llava_model_dir: str | Path,
        vision_feature_select_strategy: str = "full",
        num_additional_image_tokens: int = 1,
        device: str = "auto",
        dtype: str = "float16",
    ) -> None:
        self.llava_model_dir = Path(llava_model_dir)
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.num_additional_image_tokens = num_additional_image_tokens
        self.device_name = device
        self.dtype_name = dtype

    def load(self) -> LlavaVisionComponents:
        try:
            import torch
            from transformers import AutoProcessor, LlavaForConditionalGeneration
        except Exception as exc:
            raise RuntimeError(
                "Missing LLaVA runtime dependencies. Install transformers, torch, "
                "accelerate, safetensors, sentencepiece, protobuf and pillow."
            ) from exc

        device = self._resolve_device(torch)
        dtype = self._resolve_dtype(torch, device)
        processor = AutoProcessor.from_pretrained(str(self.llava_model_dir), local_files_only=True)
        patch_processor_from_config(
            processor,
            self.llava_model_dir,
            self.vision_feature_select_strategy,
            self.num_additional_image_tokens,
        )
        model = LlavaForConditionalGeneration.from_pretrained(
            str(self.llava_model_dir),
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            local_files_only=True,
            attn_implementation="eager",
        )
        model.to(device)
        model.eval()

        vision_tower = _get_llava_vision_tower(model)
        vision_config = getattr(vision_tower, "config", None) or getattr(model.config, "vision_config")
        image_size = int(getattr(vision_config, "image_size", 336))
        patch_size = int(getattr(vision_config, "patch_size", 14))
        patch_side = image_size // patch_size
        return LlavaVisionComponents(
            processor=processor,
            model=model,
            vision_tower=vision_tower,
            vision_config=vision_config,
            device=device,
            dtype=dtype,
            image_size=image_size,
            patch_size=patch_size,
            patch_grid=(patch_side, patch_side),
        )

    def _resolve_device(self, torch):
        if self.device_name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device_name)

    def _resolve_dtype(self, torch, device):
        if self.dtype_name == "float32" or str(device) == "cpu":
            return torch.float32
        if self.dtype_name == "bfloat16":
            return torch.bfloat16
        return torch.float16


def patch_processor_from_config(
    processor,
    model_dir: str | Path,
    vision_feature_select_strategy: str,
    num_additional_image_tokens: int,
) -> None:
    config_path = Path(model_dir) / "config.json"
    if not config_path.is_file():
        return
    config = json.loads(config_path.read_text(encoding="utf-8"))
    vision_config = config.get("vision_config", {})
    patch_size = vision_config.get("patch_size")
    image_token_index = config.get("image_token_index")

    if getattr(processor, "patch_size", None) is None and patch_size is not None:
        processor.patch_size = patch_size
    processor.num_additional_image_tokens = num_additional_image_tokens
    if getattr(processor, "vision_feature_select_strategy", None) is None and vision_feature_select_strategy:
        processor.vision_feature_select_strategy = vision_feature_select_strategy
    if getattr(processor, "image_token", None) is None and image_token_index is not None:
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None:
            image_token = tokenizer.convert_ids_to_tokens(image_token_index)
            if image_token:
                processor.image_token = image_token


def _get_llava_vision_tower(model):
    direct_vision_tower = getattr(model, "vision_tower", None)
    if direct_vision_tower is not None:
        return direct_vision_tower

    base_model = getattr(model, "model", None)
    nested_vision_tower = getattr(base_model, "vision_tower", None)
    if nested_vision_tower is not None:
        return nested_vision_tower

    get_encoder = getattr(model, "get_encoder", None)
    if get_encoder is not None:
        encoder = get_encoder()
        encoder_vision_tower = getattr(encoder, "vision_tower", None)
        if encoder_vision_tower is not None:
            return encoder_vision_tower

    raise AttributeError(
        "Could not find the LLaVA vision tower. Expected one of "
        "model.vision_tower, model.model.vision_tower, or "
        "model.get_encoder().vision_tower."
    )


def read_vision_config(model_dir: str | Path) -> dict[str, Any]:
    config_path = Path(model_dir) / "config.json"
    if not config_path.is_file():
        return {"image_size": 336, "patch_size": 14, "num_hidden_layers": 24}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    vision_config = config.get("vision_config", {})
    return {
        "image_size": int(vision_config.get("image_size", 336)),
        "patch_size": int(vision_config.get("patch_size", 14)),
        "num_hidden_layers": int(vision_config.get("num_hidden_layers", 24)),
        "pad_token_id": config.get("pad_token_id"),
    }
