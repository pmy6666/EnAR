from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .dual_branch_forwarder import DualBranchForwarder
from .logits_processor import AdaptivePlausibilityConstraint, ContrastiveLogitsProcessor
from .token_selector import NextTokenSelector
from .token_probability_debug import DEFAULT_CANDIDATES, candidate_probability_table


@dataclass
class ContrastiveGenerationResult:
    generated_ids: list[int]
    decoded_text: str
    decode_trace: list[dict[str, Any]]
    token_logits_trace: list[dict[str, Any]]


class ContrastiveGenerationLoop:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        image_token_index: int,
        alpha: float = 1.0,
        max_new_tokens: int = 64,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        use_apc: bool = False,
        apc_beta: float = 0.1,
        trace_top_k: int = 5,
        seed: int | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = int(max_new_tokens)
        self.forwarder = DualBranchForwarder(model, image_token_index)
        self.logits_processor = ContrastiveLogitsProcessor(alpha)
        self.selector = NextTokenSelector(do_sample, temperature, top_p, seed=seed)
        self.apc = AdaptivePlausibilityConstraint(apc_beta) if use_apc else None
        self.trace_top_k = int(trace_top_k)

    def run(self, input_ids, attention_mask, visual_embeddings_orig, visual_embeddings_padded) -> ContrastiveGenerationResult:
        import torch

        current_ids = input_ids
        current_mask = attention_mask
        generated: list[int] = []
        trace: list[dict[str, Any]] = []
        token_logits_trace: list[dict[str, Any]] = []
        eos_ids = _as_id_set(getattr(self.tokenizer, "eos_token_id", None), getattr(self.model.config, "eos_token_id", None))

        with torch.inference_mode():
            for step in range(self.max_new_tokens):
                out = self.forwarder.forward(
                    current_ids,
                    current_mask,
                    visual_embeddings_orig,
                    visual_embeddings_padded,
                )
                logits_vcd = self.logits_processor(out.logits_orig, out.logits_pad)
                logits_final, apc_meta = self._apply_apc(out.logits_orig, logits_vcd)
                selection = self.selector.select(logits_final)
                generated.append(selection.token_id)
                token_logits_trace.append(
                    self._build_token_logits_step(
                        step,
                        selection.token_id,
                        out.logits_orig,
                        out.logits_pad,
                    )
                )
                trace.append(
                    self._build_trace_step(
                        step,
                        selection,
                        out.logits_orig,
                        out.logits_pad,
                        logits_vcd,
                        logits_final,
                        apc_meta,
                    )
                )
                if selection.token_id in eos_ids:
                    break
                next_id = torch.tensor([[selection.token_id]], dtype=current_ids.dtype, device=current_ids.device)
                current_ids = torch.cat([current_ids, next_id], dim=-1)
                if current_mask is not None:
                    next_mask = torch.ones((current_mask.shape[0], 1), dtype=current_mask.dtype, device=current_mask.device)
                    current_mask = torch.cat([current_mask, next_mask], dim=-1)

        decoded = self.tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        return ContrastiveGenerationResult(generated, decoded, trace, token_logits_trace)

    def next_token_debug(self, input_ids, attention_mask, visual_embeddings_orig, visual_embeddings_padded):
        import torch

        with torch.inference_mode():
            out = self.forwarder.forward(
                input_ids,
                attention_mask,
                visual_embeddings_orig,
                visual_embeddings_padded,
            )
            logits_vcd = self.logits_processor(out.logits_orig, out.logits_pad)
            logits_final, apc_meta = self._apply_apc(out.logits_orig, logits_vcd)
            selection = self.selector.select(logits_final)
            return self._build_trace_step(
                None,
                selection,
                out.logits_orig,
                out.logits_pad,
                logits_vcd,
                logits_final,
                apc_meta,
            )

    def _decode_token(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def _apply_apc(self, logits_original, logits_vcd):
        apc_meta = {
            "enabled": self.apc is not None,
            "beta": getattr(self.apc, "beta", None),
            "kept_count": None,
            "filtered_count": None,
            "cutoff_mode": None,
            "fallback_to_top": False,
        }
        if self.apc is None:
            return logits_vcd, apc_meta
        apc_result = self.apc.apply(logits_original, logits_vcd)
        return apc_result.logits, {
            "enabled": True,
            "beta": apc_result.beta,
            "kept_count": apc_result.kept_count,
            "filtered_count": apc_result.filtered_count,
            "cutoff_mode": apc_result.cutoff_mode,
            "fallback_to_top": apc_result.fallback_to_top,
        }

    def _build_trace_step(
        self,
        step: int | None,
        selection,
        logits_original,
        logits_padded,
        logits_vcd,
        logits_final,
        apc_meta: dict[str, Any],
    ) -> dict[str, Any]:
        logits_by_name = {
            "original": logits_original,
            "padded": logits_padded,
            "vcd": logits_vcd,
            "final": logits_final,
        }
        trace = {
            "selected_token_id": selection.token_id,
            "selected_token": self._decode_token(selection.token_id),
            "selected_token_logprob": selection.logprob,
            "decode_mode": "vcd_sampling" if self.selector.do_sample else "greedy_debug",
            "alpha": self.logits_processor.alpha,
            "temperature": self.selector.temperature,
            "top_p": self.selector.top_p,
            "do_sample": self.selector.do_sample,
            "seed": self.selector.seed,
            "apc": apc_meta,
            "candidate_token_probabilities": candidate_probability_table(
                self.tokenizer,
                logits_by_name,
                DEFAULT_CANDIDATES,
            ),
            "logit_delta": logit_delta_stats(logits_original, logits_padded),
            "top_original": top_k_tokens(logits_original, self.tokenizer, self.trace_top_k),
            "top_padded": top_k_tokens(logits_padded, self.tokenizer, self.trace_top_k),
            "top_vcd_before_apc": top_k_tokens(logits_vcd, self.tokenizer, self.trace_top_k),
            "top_final_after_apc": top_k_tokens(logits_final, self.tokenizer, self.trace_top_k),
            "selected_token_logits": selected_token_values(selection.token_id, logits_by_name),
            "selected_token_probs": selected_token_probs(selection.token_id, logits_by_name),
            "logits_original": "logit_theta(y | x, v, y_<t)",
            "logits_contrastive_input": "logit_theta(y | x, v_pad, y_<t)",
            "logits_vcd": "(1 + alpha) * logits_original - alpha * logits_contrastive_input",
            "top_orig": top_k_tokens(logits_original, self.tokenizer, self.trace_top_k),
            "top_pad": top_k_tokens(logits_padded, self.tokenizer, self.trace_top_k),
            "top_contrastive": top_k_tokens(logits_vcd, self.tokenizer, self.trace_top_k),
            "top_final": top_k_tokens(logits_final, self.tokenizer, self.trace_top_k),
        }
        if step is not None:
            trace["step"] = step
        return trace

    def _build_token_logits_step(
        self,
        step: int,
        selected_token_id: int,
        logits_original,
        logits_padded,
    ) -> dict[str, Any]:
        return {
            "step": int(step),
            "selected_token_id": int(selected_token_id),
            "selected_token": self._decode_token(selected_token_id),
            "top_k": 20,
            "origin": top_k_logits(logits_original, self.tokenizer, 20),
            "pad": top_k_logits(logits_padded, self.tokenizer, 20),
        }


def top_k_tokens(logits, tokenizer, k: int) -> list[dict[str, Any]]:
    import torch

    k = min(int(k), int(logits.shape[-1]))
    values, indices = torch.topk(logits.float(), k=k, dim=-1)
    probs = torch.softmax(logits.float(), dim=-1)
    items = []
    for value, idx in zip(values[0], indices[0]):
        token_id = int(idx.item())
        items.append(
            {
                "token_id": token_id,
                "token": tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False),
                "logit": float(value.item()),
                "prob": float(probs[0, token_id].item()),
            }
        )
    return items


def top_k_logits(logits, tokenizer, k: int) -> list[dict[str, Any]]:
    import torch

    k = min(int(k), int(logits.shape[-1]))
    values, indices = torch.topk(logits.float(), k=k, dim=-1)
    items = []
    for rank, (value, idx) in enumerate(zip(values[0], indices[0]), start=1):
        token_id = int(idx.item())
        items.append(
            {
                "rank": rank,
                "token_id": token_id,
                "token": tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False),
                "logit": float(value.item()),
            }
        )
    return items


def logit_delta_stats(logits_orig, logits_pad) -> dict[str, float]:
    import torch

    delta = (logits_orig.float() - logits_pad.float()).reshape(-1)
    orig_top = int(torch.argmax(logits_orig.float(), dim=-1).item())
    pad_top = int(torch.argmax(logits_pad.float(), dim=-1).item())
    return {
        "l2": float(delta.norm().detach().cpu()),
        "mean_abs": float(delta.abs().mean().detach().cpu()),
        "max_abs": float(delta.abs().max().detach().cpu()),
        "orig_top_minus_pad_top_logit": float(delta[orig_top].detach().cpu()),
        "pad_top_id": pad_top,
        "orig_top_id": orig_top,
    }


def selected_token_values(token_id: int, logits_by_name: dict[str, Any]) -> dict[str, float]:
    return {
        name: float(logits.float()[0, token_id].detach().cpu())
        for name, logits in logits_by_name.items()
    }


def selected_token_probs(token_id: int, logits_by_name: dict[str, Any]) -> dict[str, float]:
    import torch

    return {
        name: float(torch.softmax(logits.float(), dim=-1)[0, token_id].detach().cpu())
        for name, logits in logits_by_name.items()
    }


def _as_id_set(*ids) -> set[int]:
    result = set()
    for item in ids:
        if item is None:
            continue
        if isinstance(item, (list, tuple, set)):
            result.update(int(x) for x in item if x is not None)
        else:
            result.add(int(item))
    return result
