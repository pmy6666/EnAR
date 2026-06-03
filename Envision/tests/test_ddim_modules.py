import torch

from Envision.ddim_inverter import DDIMInverter
from Envision.ddim_reconstructor import DDIMReconstructor
from Envision.latent_codec import LatentCodec
from Envision.tests.fakes import FakeScheduler, FakeUNet, FakeVAE


def test_ddim_inverter_returns_requested_step():
    scheduler = FakeScheduler()
    inverter = DDIMInverter(FakeUNet(), scheduler)
    z0 = torch.zeros((1, 3, 4, 4))
    text = torch.zeros((1, 4, 8))
    output = inverter.invert(z0, text, num_ddim_steps=10, inversion_step_T=3)
    assert output.zT.shape == z0.shape
    assert output.step_index_T == 3


def test_ddim_reconstructor_decodes_image():
    scheduler = FakeScheduler()
    scheduler.set_timesteps(10)
    codec = LatentCodec(FakeVAE(), torch.device("cpu"), torch.float32)
    reconstructor = DDIMReconstructor(FakeUNet(), scheduler, codec)
    output = reconstructor.reconstruct(torch.zeros((1, 3, 4, 4)), 3, torch.zeros((1, 4, 8)))
    assert output.image.size == (32, 32)
