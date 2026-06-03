import torch

from Envision.latent_codec import LatentCodec
from Envision.tests.fakes import FakeVAE


def test_latent_codec_encode_decode():
    codec = LatentCodec(FakeVAE(), torch.device("cpu"), torch.float32)
    image = torch.zeros((1, 3, 32, 32))
    latent = codec.encode(image)
    assert tuple(latent.shape) == (1, 3, 4, 4)
    pil = codec.decode_to_pil(latent)[0]
    assert pil.size == (32, 32)
