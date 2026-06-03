from pathlib import Path

import numpy as np
from PIL import Image

from Envision.config import EnvisionConfig
from Envision.output_writer import EnvisionOutputWriter


def test_output_writer_saves_expected_files(tmp_path: Path):
    image = Image.new("RGB", (4, 4), (1, 2, 3))
    writer = EnvisionOutputWriter(tmp_path)
    paths = writer.save_images(image, image, [image], image, image, image.convert("L"), image, np.zeros((4, 4), dtype=np.float32))
    metadata = writer.save_metadata(EnvisionConfig(input_image="x.png", output_dir=tmp_path), {"outputs": paths})
    assert Path(paths["impression_image"]).is_file()
    assert Path(paths["sample_images"][0]).is_file()
    assert metadata.is_file()
