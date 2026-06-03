import torch

from Envision.gradient_estimator import TweedieGradientEstimator
from Envision.langevin import LangevinPerturber
from Envision.tests.fakes import FakeScheduler, FakeUNet


def test_gradient_estimator_shape_and_norm():
    scheduler = FakeScheduler()
    estimator = TweedieGradientEstimator(FakeUNet(), scheduler)
    latent = torch.zeros((1, 3, 4, 4))
    estimate = estimator.estimate(latent, 10, torch.zeros((1, 4, 8)))
    assert estimate.gradient.shape == latent.shape
    assert estimate.gradient_norm > 0


def test_langevin_perturber_changes_latent():
    scheduler = FakeScheduler()
    estimator = TweedieGradientEstimator(FakeUNet(), scheduler)
    perturber = LangevinPerturber(estimator)
    latent = torch.zeros((1, 3, 4, 4))
    output = perturber.perturb(latent, 10, torch.zeros((1, 4, 8)), None, 2, 1e-2, 1e-4, 0.1, 42)
    assert output.latent.shape == latent.shape
    assert len(output.debug_steps) == 2
    assert not torch.allclose(output.latent, latent)
