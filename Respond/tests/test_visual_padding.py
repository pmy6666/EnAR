import torch

from Respond.padded_visual_builder import PaddedVisualInputBuilder
from Respond.visual_embeddings import build_visual_token_layout


def test_visual_token_layout_uses_attend_vision_indices():
    embeddings = torch.zeros((1, 5, 3))
    layout = build_visual_token_layout(
        embeddings,
        {
            "selected_patch_indices": [0, 2, 99],
            "selected_vision_token_indices": [1, 3, 99],
            "has_cls_token": True,
            "patch_grid": [2, 2],
            "patch_size": 14,
        },
    )
    assert layout["token_count"] == 5
    assert layout["selected_vision_token_indices"] == [1, 3]


def test_visual_token_layout_falls_back_to_cls_offset():
    embeddings = torch.zeros((1, 5, 3))
    layout = build_visual_token_layout(
        embeddings,
        {"selected_patch_indices": [0, 2], "has_cls_token": True},
    )
    assert layout["selected_vision_token_indices"] == [1, 3]


def test_padded_visual_builder_zero_strategy_replaces_only_selected_tokens():
    embeddings = torch.arange(15, dtype=torch.float32).reshape(1, 5, 3)
    result = PaddedVisualInputBuilder(model=object(), strategy="zero_embedding").build(embeddings, [1, 3, 99])
    assert result.padding_meta["replaced_count"] == 2
    assert result.padding_meta["actual_strategy"] == "zero_embedding"
    assert result.padding_meta["requested_vision_token_indices"] == [1, 3, 99]
    assert result.padding_meta["ignored_vision_token_indices"] == [99]
    assert torch.equal(result.visual_embeddings_padded[:, 0, :], embeddings[:, 0, :])
    assert torch.equal(result.visual_embeddings_padded[:, 1, :], torch.zeros(1, 3))
    assert torch.equal(result.visual_embeddings_padded[:, 3, :], torch.zeros(1, 3))
    assert torch.equal(embeddings[:, 1, :], torch.tensor([[3.0, 4.0, 5.0]]))


def test_padded_visual_builder_mean_strategy():
    embeddings = torch.tensor([[[1.0], [3.0], [5.0], [7.0]]])
    result = PaddedVisualInputBuilder(model=object(), strategy="mean_visual_embedding").build(embeddings, [1])
    assert torch.equal(result.visual_embeddings_padded[:, 1, :], torch.tensor([[13.0 / 3.0]]))


def test_padded_visual_builder_pad_strategy_records_fallback_reason():
    embeddings = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
    result = PaddedVisualInputBuilder(model=object(), strategy="pad_token_embedding").build(embeddings, [0])
    assert result.padding_meta["requested_strategy"] == "pad_token_embedding"
    assert result.padding_meta["actual_strategy"] == "zero_embedding"
    assert result.padding_meta["fallback_reason"]
    assert result.padding_meta["pad_token"]["fallback_reason"]


def test_padding_branch_changes_only_selected_tokens():
    embeddings = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    result = PaddedVisualInputBuilder(model=object(), strategy="zero_embedding").build(embeddings, [1, 2])
    assert torch.equal(result.visual_embeddings_padded[:, 0, :], embeddings[:, 0, :])
    assert torch.equal(result.visual_embeddings_padded[:, 3, :], embeddings[:, 3, :])
    assert not torch.equal(result.visual_embeddings_padded[:, 1, :], embeddings[:, 1, :])
    assert not torch.equal(result.visual_embeddings_padded[:, 2, :], embeddings[:, 2, :])
