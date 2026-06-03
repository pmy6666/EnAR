from pathlib import Path

from PIL import Image

from Envision.preprocessor import ImagePreprocessor


def test_preprocessor_outputs_tensor_range(tmp_path: Path):
    path = tmp_path / "image.png"
    Image.new("RGB", (32, 48), (128, 64, 32)).save(path)
    output = ImagePreprocessor(16).run(path)
    assert output.image_pil.size == (16, 16)
    assert tuple(output.image_tensor.shape) == (1, 3, 16, 16)
    assert output.image_tensor.min() >= -1.0
    assert output.image_tensor.max() <= 1.0
    assert output.transform_meta["original_size"] == [32, 48]
