from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .array_utils import minmax_normalize
from .mask_mapper import SOURCE_LABEL_RGB_COLORS, colorize_source_labels


class AttendVisualizer:
    def __init__(self, output_dir: str | Path, image_size: int = 336, patch_size: int = 14) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_size = image_size
        self.patch_size = patch_size

    def save_heatmap(self, grid: np.ndarray, filename: str) -> str:
        heat = colorize_heatmap(grid)
        path = self.output_dir / filename
        heat.save(path)
        return str(path)

    def save_selected_patch_mask(self, patch_mask_grid: np.ndarray, filename: str = "selected_patch_mask.png") -> str:
        mask = np.kron(
            np.asarray(patch_mask_grid, dtype=np.uint8),
            np.ones((self.patch_size, self.patch_size), dtype=np.uint8),
        ) * 255
        image = Image.fromarray(mask, mode="L").resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        path = self.output_dir / filename
        image.save(path)
        return str(path)

    def save_patch_overlay(
        self,
        original_image_path: str | Path,
        patch_mask_grid: np.ndarray,
        filename: str = "patch_overlay.png",
    ) -> str:
        original = ImageOps.exif_transpose(Image.open(original_image_path)).convert("RGB")
        canvas = ImageOps.fit(original, (self.image_size, self.image_size), method=Image.Resampling.BICUBIC)
        canvas = canvas.convert("RGBA")
        selected = np.asarray(patch_mask_grid, dtype=bool)
        draw = ImageDraw.Draw(canvas)
        fill = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        fill_draw = ImageDraw.Draw(fill)
        for row in range(selected.shape[0]):
            for col in range(selected.shape[1]):
                if not selected[row, col]:
                    continue
                x0 = col * self.patch_size
                y0 = row * self.patch_size
                x1 = x0 + self.patch_size - 1
                y1 = y0 + self.patch_size - 1
                fill_draw.rectangle([x0, y0, x1, y1], fill=(255, 32, 32, 72))
                draw.rectangle([x0, y0, x1, y1], outline=(255, 24, 24, 255), width=1)
        overlay = Image.alpha_composite(canvas, fill).convert("RGB")
        path = self.output_dir / filename
        overlay.save(path)
        return str(path)

    def save_source_label_patch_mask(
        self,
        source_label_grid: np.ndarray,
        filename: str = "selected_patch_source_mask.png",
    ) -> str:
        labels = np.kron(
            np.asarray(source_label_grid, dtype=np.uint8),
            np.ones((self.patch_size, self.patch_size), dtype=np.uint8),
        )
        image = colorize_source_labels(labels).resize(
            (self.image_size, self.image_size),
            Image.Resampling.NEAREST,
        )
        path = self.output_dir / filename
        image.save(path)
        return str(path)

    def save_source_label_patch_overlay(
        self,
        original_image_path: str | Path,
        source_label_grid: np.ndarray,
        filename: str = "patch_source_overlay.png",
    ) -> str:
        original = ImageOps.exif_transpose(Image.open(original_image_path)).convert("RGB")
        canvas = ImageOps.fit(original, (self.image_size, self.image_size), method=Image.Resampling.BICUBIC)
        canvas = canvas.convert("RGBA")
        labels = np.asarray(source_label_grid, dtype=np.uint8)
        draw = ImageDraw.Draw(canvas)
        fill = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        fill_draw = ImageDraw.Draw(fill)
        for row in range(labels.shape[0]):
            for col in range(labels.shape[1]):
                label = int(labels[row, col])
                if label == 0:
                    continue
                color = SOURCE_LABEL_RGB_COLORS[label]
                x0 = col * self.patch_size
                y0 = row * self.patch_size
                x1 = x0 + self.patch_size - 1
                y1 = y0 + self.patch_size - 1
                fill_draw.rectangle([x0, y0, x1, y1], fill=(*color, 72))
                draw.rectangle([x0, y0, x1, y1], outline=(*color, 255), width=1)
        overlay = Image.alpha_composite(canvas, fill).convert("RGB")
        path = self.output_dir / filename
        overlay.save(path)
        return str(path)


def colorize_heatmap(grid: np.ndarray) -> Image.Image:
    values = minmax_normalize(np.asarray(grid, dtype=np.float32))
    red = np.clip(255 * values, 0, 255)
    green = np.clip(255 * (1.0 - np.abs(values - 0.5) * 2.0), 0, 255)
    blue = np.clip(255 * (1.0 - values), 0, 255)
    rgb = np.stack([red, green, blue], axis=-1).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB").resize((336, 336), Image.Resampling.NEAREST)
