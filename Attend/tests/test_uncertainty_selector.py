import numpy as np

from Attend.token_selector import CounterfactualTokenSelector
from Attend.uncertainty_mapper import UncertaintyPatchMapper


def test_uncertainty_mapper_pools_to_patch_grid():
    arr = np.zeros((8, 8), dtype=np.float32)
    arr[:4, :4] = 1.0
    result = UncertaintyPatchMapper(image_size=8, patch_size=4).map_array(arr)
    assert result.patch_grid.shape == (2, 2)
    np.testing.assert_allclose(result.patch_grid, [[1, 0], [0, 0]], atol=1 / 255)


def test_token_selector_merges_and_respects_padding_limit():
    delta = np.zeros(16, dtype=np.float32)
    unc = np.zeros(16, dtype=np.float32)
    delta[[1, 2, 3, 4]] = [9, 8, 7, 6]
    unc[[4, 5, 6, 7]] = [9, 8, 7, 6]
    result = CounterfactualTokenSelector().select(
        delta,
        unc,
        attention_top_ratio=0.25,
        uncertainty_top_ratio=0.25,
        padding_ratio_limit=0.25,
        has_cls_token=True,
    )
    assert len(result.h_final) == 4
    assert result.union_patch_mask_grid.shape == (4, 4)
    assert all(v == p + 1 for p, v in zip(result.h_final, result.vision_token_indices))
