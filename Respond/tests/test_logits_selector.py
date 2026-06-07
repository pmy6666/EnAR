import torch

from Respond.logits_processor import AdaptivePlausibilityConstraint, ContrastiveLogitsProcessor
from Respond.token_selector import NextTokenSelector, top_p_filter


def test_contrastive_logits_processor_applies_eq7():
    orig = torch.tensor([[1.0, 2.0]])
    pad = torch.tensor([[0.5, 4.0]])
    out = ContrastiveLogitsProcessor(alpha=2.0)(orig, pad)
    torch.testing.assert_close(out, torch.tensor([[2.0, -2.0]]))


def test_vcd_formula_matches_reference():
    orig = torch.tensor([[1.0, 2.0, -1.0]])
    contrastive = torch.tensor([[0.5, 4.0, 3.0]])
    alpha = 1.0
    out = ContrastiveLogitsProcessor(alpha=alpha)(orig, contrastive)
    reference = (1 + alpha) * orig - alpha * contrastive
    torch.testing.assert_close(out, reference)


def test_alpha_zero_matches_regular_branch():
    orig = torch.tensor([[1.0, 2.0]])
    contrastive = torch.tensor([[100.0, -100.0]])
    out = ContrastiveLogitsProcessor(alpha=0.0)(orig, contrastive)
    torch.testing.assert_close(out, orig)


def test_alpha_one_is_two_orig_minus_contrastive():
    orig = torch.tensor([[1.0, 2.0]])
    contrastive = torch.tensor([[0.5, 4.0]])
    out = ContrastiveLogitsProcessor(alpha=1.0)(orig, contrastive)
    torch.testing.assert_close(out, 2 * orig - contrastive)


def test_contrastive_logits_processor_rejects_shape_mismatch():
    orig = torch.tensor([[1.0, 2.0]])
    contrastive = torch.tensor([[1.0]])
    try:
        ContrastiveLogitsProcessor(alpha=1.0)(orig, contrastive)
    except ValueError as exc:
        assert "logit shapes must match" in str(exc)
    else:
        raise AssertionError("shape mismatch should raise ValueError")


def test_apc_filters_low_original_probability_tokens():
    orig = torch.tensor([[10.0, 1.0, 0.0]])
    contrastive = torch.tensor([[0.0, 100.0, 50.0]])
    result = AdaptivePlausibilityConstraint(beta=0.5).apply(orig, contrastive)
    out = result.logits
    assert torch.isfinite(out[0, 0])
    assert torch.isneginf(out[0, 1])
    assert torch.isneginf(out[0, 2])
    assert result.cutoff_mode == "vcd_logit_cutoff"


def test_apc_matches_vcd_cutoff():
    orig = torch.tensor([[2.0, 1.0, -4.0]])
    contrastive = torch.tensor([[10.0, 20.0, 30.0]])
    beta = 0.2
    result = AdaptivePlausibilityConstraint(beta=beta).apply(orig, contrastive)
    cutoff = torch.log(torch.tensor(beta)) + orig.max(dim=-1, keepdim=True).values
    expected_mask = orig >= cutoff
    assert torch.equal(torch.isfinite(result.logits), expected_mask)


def test_apc_beta_zero_keeps_all_tokens():
    orig = torch.tensor([[2.0, 1.0, -4.0]])
    contrastive = torch.tensor([[10.0, 20.0, 30.0]])
    out = AdaptivePlausibilityConstraint(beta=0.0)(orig, contrastive)
    torch.testing.assert_close(out, contrastive)


def test_next_token_selector_greedy_returns_argmax_and_logprob():
    selection = NextTokenSelector(do_sample=False).select(torch.tensor([[0.0, 2.0, 1.0]]))
    assert selection.token_id == 1
    assert selection.logprob < 0


def test_sampling_seed_reproducible():
    logits = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    selector_a = NextTokenSelector(do_sample=True, seed=123)
    selector_b = NextTokenSelector(do_sample=True, seed=123)
    first = [selector_a.select(logits).token_id for _ in range(3)]
    second = [selector_b.select(logits).token_id for _ in range(3)]
    assert first == second


def test_top_p_filter_keeps_at_least_one_token():
    logits = torch.tensor([[10.0, 9.0, 1.0]])
    filtered = top_p_filter(logits, top_p=0.2)
    assert torch.isfinite(filtered[0, 0])
    assert torch.isneginf(filtered[0, 1])
