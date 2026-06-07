from __future__ import annotations

from typing import Any


def build_inputs_embeds(
        model: Any, input_ids: Any, visual_embeddings: Any, image_token_index: int
    ) -> Any:
    text_embeddings = get_input_embeddings(model)(input_ids)
    image_features = visual_embeddings.to(device=text_embeddings.device, dtype=text_embeddings.dtype)
    mask = (input_ids == image_token_index).unsqueeze(-1).expand_as(text_embeddings) # [batch, seq_len] -> [batch, seq_len, hidden_dim]
    expected = int(mask.sum().item() // text_embeddings.shape[-1])
    actual = int(image_features.shape[0] * image_features.shape[1])
    if expected != actual:
        raise ValueError(
            f"Image placeholder count ({expected}) does not match visual embedding count ({actual}). "
            "Check processor image token patch settings and vision_feature_select_strategy."
        )
    return text_embeddings.masked_scatter(
        mask, image_features.reshape(-1).to(text_embeddings.dtype)
    )


def get_input_embeddings(model: Any) -> Any:
    getter = getattr(model, "get_input_embeddings", None)
    if getter is not None:
        return getter()
    base = getattr(model, "model", None)
    getter = getattr(base, "get_input_embeddings", None)
    if getter is not None:
        return getter()
    raise AttributeError("Could not find input embeddings on LLaVA model.")
