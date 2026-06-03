from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from .config import EnvisionConfig


class EnvisionOutputWriter:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.samples_dir = self.output_dir / "samples"

    def prepare(self) -> None:
        self.samples_dir.mkdir(parents=True, exist_ok=True)

    def save_images(
        self,
        original_image: Image.Image,
        preprocessed_image: Image.Image,
        sample_images: Sequence[Image.Image],
        representative_image: Image.Image,
        difference_image: Image.Image,
        uncertainty_gray: Image.Image,
        uncertainty_heatmap: Image.Image,
        uncertainty_map: np.ndarray,
    ) -> dict:
        self.prepare()
        paths = {
            "original_image": self.output_dir / "original.png",
            "preprocessed_image": self.output_dir / "preprocessed.png",
            "impression_image": self.output_dir / "impression.png",
            "difference_image": self.output_dir / "difference.png",
            "uncertainty_gray": self.output_dir / "uncertainty_gray.png",
            "uncertainty_heatmap": self.output_dir / "uncertainty_heatmap.png",
            "uncertainty_map": self.output_dir / "uncertainty_map.npy",
        }
        original_image.save(paths["original_image"])
        preprocessed_image.save(paths["preprocessed_image"])
        representative_image.save(paths["impression_image"])
        difference_image.save(paths["difference_image"])
        uncertainty_gray.save(paths["uncertainty_gray"])
        uncertainty_heatmap.save(paths["uncertainty_heatmap"])
        np.save(paths["uncertainty_map"], uncertainty_map)

        sample_paths = []
        for idx, image in enumerate(sample_images):
            path = self.samples_dir / f"sample_{idx:03d}.png"
            image.save(path)
            sample_paths.append(path)

        result = {key: str(value) for key, value in paths.items()}
        result["sample_images"] = [str(path) for path in sample_paths]
        return result

    def save_metadata(self, config: EnvisionConfig, metadata: dict) -> Path:
        path = self.output_dir / "metadata.json"
        payload = {"config": config.to_dict(), **metadata}
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path
