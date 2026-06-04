import numpy as np

from Attend.attention_extractor import attention_tensor_to_patch_scores
from Attend.contrastive import ContrastiveAttentionComputer


def test_attention_tensor_to_patch_scores_uses_cls_to_patch_attention():
    raw = np.zeros((1, 2, 5, 5), dtype=np.float32)
    raw[0, 0, 0, 1:] = [1, 2, 3, 4]
    raw[0, 1, 0, 1:] = [3, 4, 5, 6]
    scores, meta = attention_tensor_to_patch_scores(raw)
    assert scores.tolist() == [2, 3, 4, 5]
    assert meta["has_cls_token"] is True
    assert meta["patch_grid"] == [2, 2]


def test_contrastive_attention_computes_abs_delta_grid():
    result = ContrastiveAttentionComputer().compute(
        np.array([0.1, 0.2, 0.4, 0.8]),
        np.array([0.3, 0.1, 0.1, 0.7]),
    )
    assert result.delta_grid.shape == (2, 2)
    np.testing.assert_allclose(result.delta_scores, [0.2, 0.1, 0.3, 0.1], atol=1e-6)
    assert result.normalized_grid.min() >= 0
    assert result.normalized_grid.max() <= 1
