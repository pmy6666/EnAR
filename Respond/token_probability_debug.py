from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEFAULT_CANDIDATES = ("four", "five", "4", "5")


@dataclass
class CandidateTokenization:
    candidate: str
    token_ids: list[int]
    tokens: list[str]
    scoring: str


def build_candidate_tokenizations(
    tokenizer: Any,
    candidates: tuple[str, ...] = DEFAULT_CANDIDATES,
) -> list[CandidateTokenization]:
    """Tokenize answer candidates for next-token diagnostics.

    The generation loop can only compare the immediate next token. If a
    tokenizer splits a candidate into multiple IDs, this records the split and
    uses the first-token probability as an approximation.
    """
    result: list[CandidateTokenization] = []
    for candidate in candidates:
        token_ids = _encode_without_special_tokens(tokenizer, candidate)
        if not token_ids:
            prefixed = _encode_without_special_tokens(tokenizer, " " + candidate)
            token_ids = prefixed or token_ids
        result.append(
            CandidateTokenization(
                candidate=candidate,
                token_ids=token_ids,
                tokens=[
                    tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    for token_id in token_ids
                ],
                scoring="first_token_probability" if len(token_ids) != 1 else "single_token_probability",
            )
        )
    return result


def candidate_probability_table(
    tokenizer: Any,
    logits_by_name: dict[str, Any],
    candidates: tuple[str, ...] = DEFAULT_CANDIDATES,
) -> list[dict[str, Any]]:
    import torch

    tokenizations = build_candidate_tokenizations(tokenizer, candidates)
    probs_by_name = {
        name: torch.softmax(logits.float(), dim=-1)
        for name, logits in logits_by_name.items()
        if logits is not None
    }
    table: list[dict[str, Any]] = []
    for tokenization in tokenizations:
        row: dict[str, Any] = {
            "candidate": tokenization.candidate,
            "token_ids": tokenization.token_ids,
            "tokens": tokenization.tokens,
            "scoring": tokenization.scoring,
        }
        token_id = tokenization.token_ids[0] if tokenization.token_ids else None
        for name, logits in logits_by_name.items():
            if token_id is None or logits is None:
                row[f"{name}_logit"] = None
                row[f"{name}_prob"] = None
                continue
            row[f"{name}_logit"] = float(logits.float()[0, token_id].detach().cpu())
            row[f"{name}_prob"] = float(probs_by_name[name][0, token_id].detach().cpu())
        table.append(row)
    return table


def probability_lookup(table: list[dict[str, Any]], prob_column: str) -> dict[str, float | None]:
    return {
        str(row["candidate"]): row.get(prob_column)
        for row in table
    }


def _encode_without_special_tokens(tokenizer: Any, text: str) -> list[int]:
    encode = getattr(tokenizer, "encode", None)
    if encode is not None:
        return [int(token_id) for token_id in encode(text, add_special_tokens=False)]
    encoded = tokenizer(text, add_special_tokens=False)
    if hasattr(encoded, "get"):
        input_ids = encoded.get("input_ids")
    else:
        input_ids = encoded
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return [int(token_id) for token_id in input_ids]
