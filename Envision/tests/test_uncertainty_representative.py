from PIL import Image

from Envision.representative import RepresentativeSelector
from Envision.uncertainty import UncertaintyEstimator


def test_uncertainty_outputs_images():
    samples = [
        Image.new("RGB", (8, 8), (0, 0, 0)),
        Image.new("RGB", (8, 8), (255, 0, 0)),
    ]
    output = UncertaintyEstimator().estimate(samples)
    assert output.uncertainty_map.shape == (8, 8)
    assert output.gray_image.mode == "L"
    assert output.heatmap_image.mode == "RGB"


def test_representative_selects_largest_l2():
    original = Image.new("RGB", (4, 4), (0, 0, 0))
    samples = [
        Image.new("RGB", (4, 4), (10, 10, 10)),
        Image.new("RGB", (4, 4), (100, 100, 100)),
    ]
    selected = RepresentativeSelector().select(original, samples)
    assert selected.index == 1
    assert selected.diff_scores[1] > selected.diff_scores[0]
