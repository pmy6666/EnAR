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
    assert output.transform_meta["geometry"] == "resize_keep_aspect_then_pad"
    assert output.transform_meta["resized_size"] == [11, 16]
    assert output.transform_meta["pad"] == {"left": 2, "top": 0, "right": 3, "bottom": 0}
    assert output.transform_meta["content_box_target"] == [2, 0, 13, 16]


def test_preprocessor_keeps_center_crop_mode(tmp_path: Path):
    path = tmp_path / "image.png"
    Image.new("RGB", (32, 48), (128, 64, 32)).save(path)
    output = ImagePreprocessor(16, preprocess_mode="center_crop").run(path)
    assert output.image_pil.size == (16, 16)
    assert output.transform_meta["geometry"] == "center_crop_then_resize"
    assert output.transform_meta["crop_box_original"] == [0.0, 8.0, 32.0, 40.0]
