from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


@dataclass
class MaskOriginResult:
    mask_origin: np.ndarray
    mask_origin_path: str
    mask_origin_overlay_path: str | None
    meta: dict


class MaskOriginMapper:
    def __init__(
        self,
        patch_size: int = 14,
        vision_input_size: tuple[int, int] = (336, 336),
        alpha: float = 0.45,
    ) -> None:
        self.patch_size = patch_size
        self.vision_input_size = vision_input_size
        self.alpha = alpha

    def map_and_save(
        self,
        union_patch_mask_grid: np.ndarray,
        original_image_path: str | Path,
        preprocess_meta: dict,
        output_dir: str | Path,
        save_overlay: bool = True,
    ) -> MaskOriginResult:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        original = ImageOps.exif_transpose(Image.open(original_image_path)).convert("RGB")
        mask_origin, meta = self.map_to_origin(union_patch_mask_grid, original.size, preprocess_meta)
        mask_path = output_dir / "mask_origin.png"
        Image.fromarray(mask_origin, mode="L").save(mask_path)

        overlay_path = None
        if save_overlay:
            overlay_path_obj = output_dir / "mask_origin_overlay.png"
            make_overlay(original, mask_origin, self.alpha).save(overlay_path_obj)
            overlay_path = str(overlay_path_obj)

        return MaskOriginResult(
            mask_origin=mask_origin,
            mask_origin_path=str(mask_path),
            mask_origin_overlay_path=overlay_path,
            meta=meta,
        )

    def map_to_origin(
        self,
        union_patch_mask_grid: np.ndarray,
        original_size: tuple[int, int],
        preprocess_meta: dict,
    ) -> tuple[np.ndarray, dict]:
        patch_mask = np.asarray(union_patch_mask_grid, dtype=np.uint8)
        vision_mask = np.kron(patch_mask, np.ones((self.patch_size, self.patch_size), dtype=np.uint8)) * 255
        input_w, input_h = self._meta_input_size(preprocess_meta)
        vision_image = Image.fromarray(vision_mask, mode="L").resize((input_w, input_h), Image.Resampling.NEAREST)

        original_w, original_h = original_size
        crop_box = preprocess_meta.get("crop_box_original")
        if crop_box is None:
            crop = min(original_w, original_h)
            crop_box = [
                max((original_w - crop) / 2.0, 0.0),
                max((original_h - crop) / 2.0, 0.0),
                max((original_w - crop) / 2.0, 0.0) + crop,
                max((original_h - crop) / 2.0, 0.0) + crop,
            ]

        left, top, right, bottom = crop_box
        crop_w = max(1, int(round(right - left)))
        crop_h = max(1, int(round(bottom - top)))
        crop_mask = vision_image.resize((crop_w, crop_h), Image.Resampling.NEAREST)
        canvas = Image.new("L", (original_w, original_h), 0)
        canvas.paste(crop_mask, (int(round(left)), int(round(top))))
        mask = np.asarray(canvas, dtype=np.uint8)
        meta = {
            "original_size": [int(original_w), int(original_h)],
            "vision_input_size": [int(input_w), int(input_h)],
            "patch_grid": list(patch_mask.shape),
            "patch_size": int(self.patch_size),
            "crop_box_original": [float(v) for v in crop_box],
            "mapping_assumption": preprocess_meta.get("mapping_assumption", "center_crop_then_resize"),
        }
        return mask, meta

    def _meta_input_size(self, preprocess_meta: dict) -> tuple[int, int]:
        value = preprocess_meta.get("vision_input_size", self.vision_input_size)
        return int(value[0]), int(value[1])


def make_overlay(image: Image.Image, mask: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base = image.convert("RGBA")
    mask_image = Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255, mode="L")
    color = Image.new("RGBA", base.size, (255, 48, 48, int(round(255 * alpha))))
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    overlay.paste(color, (0, 0), mask_image)
    return Image.alpha_composite(base, overlay).convert("RGB")
