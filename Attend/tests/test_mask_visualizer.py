from pathlib import Path

import numpy as np
from PIL import Image

from Attend.mask_mapper import MaskOriginMapper
from Attend.visualizer import AttendVisualizer


def test_mask_origin_mapper_outputs_original_size(tmp_path: Path):
    image_path = tmp_path / "original.png"
    Image.new("RGB", (80, 40), "white").save(image_path)
    patch_mask = np.zeros((2, 2), dtype=bool)
    patch_mask[0, 0] = True
    result = MaskOriginMapper(patch_size=4, vision_input_size=(8, 8)).map_and_save(
        patch_mask,
        image_path,
        {
            "vision_input_size": [8, 8],
            "crop_box_original": [20, 0, 60, 40],
            "mapping_assumption": "test",
        },
        tmp_path,
    )
    assert result.mask_origin.shape == (40, 80)
    assert Path(result.mask_origin_path).is_file()
    assert Path(result.mask_origin_overlay_path).is_file()


def test_mask_origin_mapper_supports_pad_content_box(tmp_path: Path):
    image_path = tmp_path / "original.png"
    Image.new("RGB", (8, 4), "white").save(image_path)
    patch_mask = np.ones((2, 2), dtype=bool)
    result = MaskOriginMapper(patch_size=4, vision_input_size=(8, 8)).map_and_save(
        patch_mask,
        image_path,
        {
            "vision_input_size": [8, 8],
            "geometry": "resize_keep_aspect_then_pad",
            "content_box_target": [0, 2, 8, 6],
        },
        tmp_path,
    )
    assert result.mask_origin.shape == (4, 8)
    assert result.meta["crop_box_original"] == [0.0, 0.0, 8.0, 4.0]
    assert result.meta["content_box_target"] == [0.0, 2.0, 8.0, 6.0]
    assert result.mask_origin.min() == 255


def test_mask_origin_mapper_outputs_label_masks(tmp_path: Path):
    image_path = tmp_path / "original.png"
    Image.new("RGB", (80, 40), "white").save(image_path)
    label_grid = np.array([[1, 2], [3, 0]], dtype=np.uint8)
    result = MaskOriginMapper(patch_size=4, vision_input_size=(8, 8)).map_label_and_save(
        label_grid,
        image_path,
        {
            "vision_input_size": [8, 8],
            "crop_box_original": [20, 0, 60, 40],
            "mapping_assumption": "test",
        },
        tmp_path,
    )
    assert result.label_mask_origin.shape == (40, 80)
    assert set(np.unique(result.label_mask_origin).tolist()) <= {0, 1, 2, 3}
    assert {1, 2, 3} <= set(np.unique(result.label_mask_origin).tolist())
    assert Path(result.label_mask_origin_path).is_file()
    assert Path(result.label_mask_origin_color_path).is_file()
    assert Path(result.label_mask_origin_overlay_path).is_file()


def test_visualizer_writes_patch_images(tmp_path: Path):
    image_path = tmp_path / "original.png"
    Image.new("RGB", (32, 32), "white").save(image_path)
    patch_mask = np.zeros((2, 2), dtype=bool)
    patch_mask[1, 1] = True
    visualizer = AttendVisualizer(tmp_path, image_size=8, patch_size=4)
    assert Path(visualizer.save_selected_patch_mask(patch_mask)).is_file()
    assert Path(visualizer.save_patch_overlay(image_path, patch_mask)).is_file()
    assert Path(visualizer.save_heatmap(np.array([[0, 1], [2, 3]], dtype=np.float32), "heat.png")).is_file()


def test_visualizer_writes_source_label_images(tmp_path: Path):
    image_path = tmp_path / "original.png"
    Image.new("RGB", (32, 32), "white").save(image_path)
    label_grid = np.array([[1, 2], [3, 0]], dtype=np.uint8)
    visualizer = AttendVisualizer(tmp_path, image_size=8, patch_size=4)
    mask_path = Path(visualizer.save_source_label_patch_mask(label_grid))
    overlay_path = Path(visualizer.save_source_label_patch_overlay(image_path, label_grid))
    assert mask_path.is_file()
    assert overlay_path.is_file()

    colors = np.asarray(Image.open(mask_path).convert("RGB")).reshape(-1, 3)
    unique_colors = {tuple(color) for color in colors.tolist()}
    assert (255, 48, 48) in unique_colors
    assert (47, 128, 255) in unique_colors
    assert (255, 210, 63) in unique_colors
