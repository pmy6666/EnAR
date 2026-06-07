from __future__ import annotations

from typing import Any


class RegularGenerationRunner:
    def __init__(self, model: Any, processor: Any, tokenizer: Any) -> None:
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer

    def run(
        self,
        encoded_input: Any,
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: int | None = None,
    ) -> str:
        import torch

        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
        kwargs = {
            "input_ids": encoded_input.input_ids,
            "attention_mask": encoded_input.attention_mask,
            "pixel_values": encoded_input.pixel_values,
            "max_new_tokens": int(max_new_tokens),
            "do_sample": bool(do_sample),
        }
        if do_sample:
            kwargs.update({"temperature": float(temperature), "top_p": float(top_p)})
        with torch.inference_mode():
            output_ids = self.model.generate(**kwargs)
        new_ids = output_ids[:, encoded_input.prompt_len:]
        return self.tokenizer.batch_decode(
            new_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
