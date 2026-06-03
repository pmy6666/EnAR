import torch

from Envision.prompt_conditioner import PromptConditioner
from Envision.tests.fakes import FakeTextEncoder, FakeTokenizer


def test_prompt_conditioner_encodes_empty_prompt():
    conditioner = PromptConditioner(FakeTokenizer(), FakeTextEncoder(), torch.device("cpu"), torch.float32)
    embeddings = conditioner.encode("")
    assert tuple(embeddings.text_embeddings.shape) == (1, 4, 8)
    assert embeddings.negative_text_embeddings is None
