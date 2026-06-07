from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PaddedVisualInput:
    visual_embeddings_padded: Any
    padding_mask: Any
    padding_meta: dict[str, Any]


class PaddedVisualInputBuilder:
    def __init__(self, model: Any, tokenizer: Any | None = None, strategy: str = "pad_token_embedding") -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.strategy = strategy

    def build(self, visual_embeddings_orig: Any, selected_vision_token_indices: list[int]) -> PaddedVisualInput:
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("torch is required to build padded visual embeddings.") from exc

        padded = visual_embeddings_orig.clone()
        token_count = int(padded.shape[-2])
        requested_indices = [int(idx) for idx in selected_vision_token_indices]
        indices = sorted({idx for idx in requested_indices if 0 <= idx < token_count})
        ignored_indices = [idx for idx in requested_indices if idx < 0 or idx >= token_count]
        mask = torch.zeros(token_count, dtype=torch.bool, device=padded.device)
        if indices:
            mask[torch.tensor(indices, dtype=torch.long, device=padded.device)] = True

        requested_strategy = self.strategy
        actual_strategy = requested_strategy
        fallback_reason = None
        pad_token_meta = None
        replacement = None
        if indices:
            if requested_strategy == "pad_token_embedding":
                pad_lookup = self._pad_token_embedding(padded)
                replacement = pad_lookup["embedding"]
                pad_token_meta = pad_lookup["meta"]
                if replacement is None:
                    actual_strategy = "zero_embedding"
                    fallback_reason = pad_token_meta.get("fallback_reason", "pad token embedding unavailable")
            if actual_strategy == "zero_embedding":
                replacement = torch.zeros(padded.shape[-1], dtype=padded.dtype, device=padded.device)
            elif actual_strategy in {"mean_visual_embedding", "matched_mean_visual_embedding"}:
                keep = ~mask
                replacement = padded[:, keep, :].mean(dim=(0, 1)) if bool(keep.any()) else padded.mean(dim=(0, 1))
                if actual_strategy == "matched_mean_visual_embedding":
                    target_norm = padded[:, mask, :].norm(dim=-1).mean() if bool(mask.any()) else padded.norm(dim=-1).mean()
                    repl_norm = replacement.norm().clamp_min(1e-12)
                    replacement = replacement * (target_norm / repl_norm)
            padded[:, mask, :] = replacement.to(dtype=padded.dtype, device=padded.device)

        before_norm = visual_embeddings_orig[:, mask, :].norm(dim=-1).detach().float().cpu().tolist() if indices else []
        after_norm = padded[:, mask, :].norm(dim=-1).detach().float().cpu().tolist() if indices else []
        global_norm = visual_embeddings_orig.norm(dim=-1).detach().float()
        replacement_stats = _tensor_stats("replacement", replacement) if replacement is not None else None
        return PaddedVisualInput(
            visual_embeddings_padded=padded,
            padding_mask=mask,
            padding_meta={
                "requested_strategy": requested_strategy,
                "actual_strategy": actual_strategy,
                "fallback_reason": fallback_reason,
                "pad_token": pad_token_meta,
                "requested_vision_token_indices": requested_indices,
                "selected_vision_token_indices": indices,
                "ignored_vision_token_indices": ignored_indices,
                "replaced_count": len(indices),
                "token_count": token_count,
                "before_norm": before_norm,
                "after_norm": after_norm,
                "visual_norm_stats": {
                    "min": float(global_norm.min().cpu()),
                    "max": float(global_norm.max().cpu()),
                    "mean": float(global_norm.mean().cpu()),
                    "std": float(global_norm.std(unbiased=False).cpu()),
                },
                "selected_before_norm_mean": _mean_nested(before_norm),
                "selected_after_norm_mean": _mean_nested(after_norm),
                "replacement_stats": replacement_stats,
            },
        )

    def _pad_token_embedding(self, like_tensor: Any) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "pad_token_id": None,
            "pad_token_id_source": None,
            "fallback_reason": None,
        }
        pad_token_id = getattr(getattr(self.model, "config", None), "pad_token_id", None)
        if pad_token_id is not None:
            meta["pad_token_id_source"] = "model.config.pad_token_id"
        if pad_token_id is None and self.tokenizer is not None:
            pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
            if pad_token_id is not None:
                meta["pad_token_id_source"] = "tokenizer.pad_token_id"
        if pad_token_id is None:
            meta["fallback_reason"] = "pad_token_id is missing from model config and tokenizer"
            return {"embedding": None, "meta": meta}
        meta["pad_token_id"] = int(pad_token_id)
        embeddings = _get_input_embeddings(self.model)
        if embeddings is None:
            meta["fallback_reason"] = "input embedding table is unavailable"
            return {"embedding": None, "meta": meta}
        weight = embeddings.weight
        if int(pad_token_id) < 0 or int(pad_token_id) >= int(weight.shape[0]):
            meta["fallback_reason"] = f"pad_token_id {int(pad_token_id)} is outside embedding table size {int(weight.shape[0])}"
            return {"embedding": None, "meta": meta}
        pad_embedding = weight[int(pad_token_id)].detach()
        if int(pad_embedding.shape[-1]) != int(like_tensor.shape[-1]):
            meta["fallback_reason"] = (
                f"pad embedding hidden size {int(pad_embedding.shape[-1])} "
                f"does not match visual hidden size {int(like_tensor.shape[-1])}"
            )
            return {"embedding": None, "meta": meta}
        meta["embedding_hidden_size"] = int(pad_embedding.shape[-1])
        meta["visual_hidden_size"] = int(like_tensor.shape[-1])
        return {"embedding": pad_embedding, "meta": meta}


def _get_input_embeddings(model: Any) -> Any | None:
    getter = getattr(model, "get_input_embeddings", None)
    if getter is not None:
        return getter()
    base = getattr(model, "model", None)
    getter = getattr(base, "get_input_embeddings", None)
    if getter is not None:
        return getter()
    return None


def _mean_nested(values: list) -> float | None:
    flat = []
    for item in values:
        if isinstance(item, list):
            flat.extend(float(x) for x in item)
        else:
            flat.append(float(item))
    if not flat:
        return None
    return sum(flat) / len(flat)


def _tensor_stats(name: str, tensor: Any) -> dict[str, Any]:
    values = tensor.detach().float()
    if values.ndim == 1:
        norm = values.norm().reshape(1)
    else:
        norm = values.norm(dim=-1).reshape(-1)
    return {
        "name": name,
        "shape": [int(dim) for dim in values.shape],
        "mean": float(values.mean().cpu()),
        "std": float(values.std(unbiased=False).cpu()),
        "norm_mean": float(norm.mean().cpu()),
        "norm_min": float(norm.min().cpu()),
        "norm_max": float(norm.max().cpu()),
    }
