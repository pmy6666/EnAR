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


def test_visualizer_writes_patch_images(tmp_path: Path):
    image_path = tmp_path / "original.png"
    Image.new("RGB", (32, 32), "white").save(image_path)
    patch_mask = np.zeros((2, 2), dtype=bool)
    patch_mask[1, 1] = True
    visualizer = AttendVisualizer(tmp_path, image_size=8, patch_size=4)
    assert Path(visualizer.save_selected_patch_mask(patch_mask)).is_file()
    assert Path(visualizer.save_patch_overlay(image_path, patch_mask)).is_file()
    assert Path(visualizer.save_heatmap(np.array([[0, 1], [2, 3]], dtype=np.float32), "heat.png")).is_file()
