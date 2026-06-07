import pytest
import torch

from Respond.embedding_merge import build_inputs_embeds


class TinyModel:
    def __init__(self):
        self.embedding = torch.nn.Embedding(10, 2)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.arange(20, dtype=torch.float32).reshape(10, 2))

    def get_input_embeddings(self):
        return self.embedding


def test_build_inputs_embeds_replaces_image_placeholders():
    model = TinyModel()
    input_ids = torch.tensor([[1, 9, 9, 2]])
    visual = torch.tensor([[[100.0, 101.0], [200.0, 201.0]]])
    out = build_inputs_embeds(model, input_ids, visual, image_token_index=9)
    torch.testing.assert_close(out[0, 0], model.embedding.weight[1])
    torch.testing.assert_close(out[0, 1], torch.tensor([100.0, 101.0]))
    torch.testing.assert_close(out[0, 2], torch.tensor([200.0, 201.0]))
    torch.testing.assert_close(out[0, 3], model.embedding.weight[2])


def test_build_inputs_embeds_checks_placeholder_count():
    model = TinyModel()
    input_ids = torch.tensor([[1, 9, 2]])
    visual = torch.zeros((1, 2, 2))
    with pytest.raises(ValueError):
        build_inputs_embeds(model, input_ids, visual, image_token_index=9)
