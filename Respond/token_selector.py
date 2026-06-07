from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NextTokenSelection:
    token_id: int
    logprob: float


class NextTokenSelector:
    def __init__(self, do_sample: bool = False, temperature: float = 1.0, top_p: float = 1.0, seed: int | None = None) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1].")
        self.do_sample = do_sample
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.seed = int(seed) if seed is not None else None
        self._generator = None

    def select(self, logits) -> NextTokenSelection:
        import torch

        logits = logits.float()
        if not self.do_sample:
            token = int(torch.argmax(logits, dim=-1).item())
            logprob = float(torch.log_softmax(logits, dim=-1)[0, token].item())
            return NextTokenSelection(token, logprob)
        filtered = top_p_filter(logits / self.temperature, self.top_p)
        probs = torch.softmax(filtered, dim=-1)
        if not torch.isfinite(probs).all() or float(probs.sum(dim=-1).min().item()) <= 0.0:
            filtered = logits / self.temperature
            probs = torch.softmax(filtered, dim=-1)
        token_tensor = torch.multinomial(probs, num_samples=1, generator=self._get_generator(probs.device))
        token = int(token_tensor.item())
        logprob = float(torch.log(probs[0, token].clamp_min(1e-20)).item())
        return NextTokenSelection(token, logprob)

    def _get_generator(self, device):
        import torch

        if self.seed is None:
            return None
        if self._generator is None:
            self._generator = torch.Generator(device=device)
            self._generator.manual_seed(self.seed)
        return self._generator


def top_p_filter(logits, top_p: float):
    import torch

    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    filtered = logits.clone()
    filtered.scatter_(dim=-1, index=sorted_indices, src=torch.where(remove, torch.full_like(sorted_logits, -torch.inf), sorted_logits))
    return filtered
