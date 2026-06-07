from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .embedding_merge import build_inputs_embeds


@dataclass
class DualBranchForwardResult:
    logits_orig: Any
    logits_pad: Any


class DualBranchForwarder:
    def __init__(self, model: Any, image_token_index: int) -> None:
        self.model = model
        self.image_token_index = int(image_token_index)

    def forward(
        self,
        input_ids: Any,
        attention_mask: Any,
        visual_embeddings_orig: Any,
        visual_embeddings_padded: Any,
    ) -> DualBranchForwardResult:
        inputs_embeds_orig = build_inputs_embeds(
            self.model,
            input_ids,
            visual_embeddings_orig,
            self.image_token_index,
        )
        inputs_embeds_pad = build_inputs_embeds(
            self.model,
            input_ids,
            visual_embeddings_padded,
            self.image_token_index,
        )
        out_orig = forward_language_model(self.model, inputs_embeds_orig, attention_mask) # [batch, seq_len, hidden_dim]
        out_pad = forward_language_model(self.model, inputs_embeds_pad, attention_mask) # [batch, seq_len, hidden_dim]
        return DualBranchForwardResult(
            logits_orig=out_orig.logits[:, -1, :], # [batch, hidden_dim]
            logits_pad=out_pad.logits[:, -1, :], # [batch, hidden_dim]
        )


def forward_language_model(model: Any, inputs_embeds: Any, attention_mask: Any) -> Any:
    base = getattr(model, "model", None)
    language_model = getattr(base, "language_model", None)
    lm_head = getattr(model, "lm_head", None)
    if language_model is None or lm_head is None:
        return model(input_ids=None, inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    outputs = language_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    hidden_states = outputs[0]
    logits = lm_head(hidden_states)
    return _LogitsOutput(logits=logits)


@dataclass
class _LogitsOutput:
    logits: Any
