from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PromptEmbeddings:
    text_embeddings: torch.Tensor
    negative_text_embeddings: torch.Tensor | None = None


class PromptConditioner:
    def __init__(self, tokenizer, text_encoder, device: torch.device, dtype: torch.dtype) -> None:
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = device
        self.dtype = dtype

    def encode(self, prompt: str = "", negative_prompt: str = "") -> PromptEmbeddings:
        text_embeddings = self._encode_text(prompt)
        negative_embeddings = self._encode_text(negative_prompt) if negative_prompt else None
        return PromptEmbeddings(text_embeddings, negative_embeddings)

    def _encode_text(self, text: str) -> torch.Tensor:
        max_length = getattr(self.tokenizer, "model_max_length", 77)
        tokens = self.tokenizer(
            [text],
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(self.device)
        with torch.no_grad():
            embeddings = self.text_encoder(input_ids)[0]
        return embeddings.to(device=self.device, dtype=self.dtype)
