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


@dataclass
class LabelMaskOriginResult:
    label_mask_origin: np.ndarray
    label_mask_origin_path: str
    label_mask_origin_color_path: str
    label_mask_origin_overlay_path: str | None
    meta: dict


SOURCE_LABEL_ENCODING = {
    0: "background",
    1: "attention_only",
    2: "uncertainty_only",
    3: "attention_and_uncertainty",
}

SOURCE_LABEL_RGB_COLORS = {
    0: (0, 0, 0),
    1: (255, 48, 48),
    2: (47, 128, 255),
    3: (255, 210, 63),
}


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

    def map_label_and_save(
        self,
        source_label_grid: np.ndarray,
        original_image_path: str | Path,
        preprocess_meta: dict,
        output_dir: str | Path,
        save_overlay: bool = True,
    ) -> LabelMaskOriginResult:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        original = ImageOps.exif_transpose(Image.open(original_image_path)).convert("RGB")
        label_mask_origin, meta = self.map_label_to_origin(source_label_grid, original.size, preprocess_meta)

        label_path = output_dir / "mask_origin_label.png"
        Image.fromarray(label_mask_origin, mode="L").save(label_path)

        color_path = output_dir / "mask_origin_color.png"
        colorize_source_labels(label_mask_origin).save(color_path)

        overlay_path = None
        if save_overlay:
            overlay_path_obj = output_dir / "mask_origin_three_color_overlay.png"
            make_label_overlay(original, label_mask_origin, self.alpha).save(overlay_path_obj)
            overlay_path = str(overlay_path_obj)

        return LabelMaskOriginResult(
            label_mask_origin=label_mask_origin,
            label_mask_origin_path=str(label_path),
            label_mask_origin_color_path=str(color_path),
            label_mask_origin_overlay_path=overlay_path,
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
        content_box = preprocess_meta.get("content_box_target")
        if content_box is not None and preprocess_meta.get("geometry") == "resize_keep_aspect_then_pad":
            left, top, right, bottom = [int(round(float(v))) for v in content_box]
            content = vision_image.crop((left, top, right, bottom))
            crop_mask = content.resize((original_w, original_h), Image.Resampling.NEAREST)
            crop_box = [0.0, 0.0, float(original_w), float(original_h)]
        else:
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
        canvas.paste(crop_mask, (int(round(crop_box[0])), int(round(crop_box[1]))))
        mask = np.asarray(canvas, dtype=np.uint8)
        meta = {
            "original_size": [int(original_w), int(original_h)],
            "vision_input_size": [int(input_w), int(input_h)],
            "patch_grid": list(patch_mask.shape),
            "patch_size": int(self.patch_size),
            "crop_box_original": [float(v) for v in crop_box],
            "content_box_target": [float(v) for v in content_box] if content_box is not None else None,
            "mapping_assumption": preprocess_meta.get("geometry", preprocess_meta.get("mapping_assumption", "center_crop_then_resize")),
        }
        return mask, meta

    def map_label_to_origin(
        self,
        source_label_grid: np.ndarray,
        original_size: tuple[int, int],
        preprocess_meta: dict,
    ) -> tuple[np.ndarray, dict]:
        label_grid = np.asarray(source_label_grid, dtype=np.uint8)
        _validate_source_labels(label_grid)
        vision_labels = np.kron(
            label_grid,
            np.ones((self.patch_size, self.patch_size), dtype=np.uint8),
        )
        input_w, input_h = self._meta_input_size(preprocess_meta)
        vision_image = Image.fromarray(vision_labels, mode="L").resize(
            (input_w, input_h),
            Image.Resampling.NEAREST,
        )

        original_w, original_h = original_size
        content_box = preprocess_meta.get("content_box_target")
        if content_box is not None and preprocess_meta.get("geometry") == "resize_keep_aspect_then_pad":
            left, top, right, bottom = [int(round(float(v))) for v in content_box]
            content = vision_image.crop((left, top, right, bottom))
            crop_mask = content.resize((original_w, original_h), Image.Resampling.NEAREST)
            crop_box = [0.0, 0.0, float(original_w), float(original_h)]
        else:
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
        canvas.paste(crop_mask, (int(round(crop_box[0])), int(round(crop_box[1]))))
        label_mask = np.asarray(canvas, dtype=np.uint8)
        _validate_source_labels(label_mask)
        meta = {
            "original_size": [int(original_w), int(original_h)],
            "vision_input_size": [int(input_w), int(input_h)],
            "patch_grid": list(label_grid.shape),
            "patch_size": int(self.patch_size),
            "crop_box_original": [float(v) for v in crop_box],
            "content_box_target": [float(v) for v in content_box] if content_box is not None else None,
            "mapping_assumption": preprocess_meta.get("geometry", preprocess_meta.get("mapping_assumption", "center_crop_then_resize")),
            "source_label_encoding": {str(k): v for k, v in SOURCE_LABEL_ENCODING.items()},
        }
        return label_mask, meta

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


def colorize_source_labels(label_mask: np.ndarray) -> Image.Image:
    labels = np.asarray(label_mask, dtype=np.uint8)
    _validate_source_labels(labels)
    rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for label, color in SOURCE_LABEL_RGB_COLORS.items():
        rgb[labels == label] = color
    return Image.fromarray(rgb, mode="RGB")


def make_label_overlay(image: Image.Image, label_mask: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base = image.convert("RGBA")
    labels = np.asarray(label_mask, dtype=np.uint8)
    _validate_source_labels(labels)
    rgba = np.zeros((*labels.shape, 4), dtype=np.uint8)
    alpha_value = int(round(255 * alpha))
    for label, color in SOURCE_LABEL_RGB_COLORS.items():
        if label == 0:
            continue
        rgba[labels == label] = (*color, alpha_value)
    overlay = Image.fromarray(rgba, mode="RGBA")
    return Image.alpha_composite(base, overlay).convert("RGB")


def _validate_source_labels(labels: np.ndarray) -> None:
    unique = set(int(value) for value in np.unique(labels))
    allowed = set(SOURCE_LABEL_ENCODING)
    invalid = sorted(unique - allowed)
    if invalid:
        raise ValueError(f"source label mask contains invalid labels: {invalid}")
