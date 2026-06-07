from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class LlavaGenerationComponents:
    processor: Any
    model: Any
    tokenizer: Any
    device: Any
    dtype: Any
    generation_meta: dict[str, Any]


class LlavaGenerationLoader:
    def __init__(
        self,
        llava_model_dir: str | Path,
        device: str = "auto",
        dtype: str = "float16",
        vision_feature_select_strategy: str = "default",
        num_additional_image_tokens: int = 0,
    ) -> None:
        self.llava_model_dir = Path(llava_model_dir)
        self.device_name = device
        self.dtype_name = dtype
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.num_additional_image_tokens = num_additional_image_tokens

    def load(self) -> LlavaGenerationComponents:
        try:
            import torch
            from transformers import AutoProcessor, LlavaForConditionalGeneration
        except Exception as exc:
            raise RuntimeError(
                "Missing LLaVA runtime dependencies. Install torch, transformers, accelerate, "
                "safetensors, sentencepiece, protobuf and pillow."
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
        tokenizer = getattr(processor, "tokenizer", None)
        model = LlavaForConditionalGeneration.from_pretrained(
            str(self.llava_model_dir),
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            local_files_only=True,
            attn_implementation="eager",
        )
        model.to(device)
        model.eval()
        meta = {
            "device": str(device),
            "dtype": str(dtype),
            "vision_feature_select_strategy": self.vision_feature_select_strategy,
            "vision_feature_layer": getattr(model.config, "vision_feature_layer", -2),
            "num_additional_image_tokens": self.num_additional_image_tokens,
            "image_token_index": getattr(model.config, "image_token_index", None),
            "pad_token_id": getattr(model.config, "pad_token_id", None),
            "eos_token_id": getattr(model.config, "eos_token_id", None),
        }
        return LlavaGenerationComponents(processor, model, tokenizer, device, dtype, meta)

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
