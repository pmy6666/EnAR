from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_IMAGE = Path("EnAR/outputs/attend/run_001/mask_origin_three_color_overlay.png")
DEFAULT_ATTEND_RESULT = Path("EnAR/outputs/attend/run_001/attend_result.json")
DEFAULT_OUTPUT = Path("EnAR/outputs/attend/run_001/mask_origin_three_color_patch_indices.png")


def main() -> None:
    args = parse_args()
    attend_result = load_json(args.attend_result) if args.attend_result.is_file() else {}
    grid_h, grid_w = resolve_grid(args, attend_result)

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")

    mapping = resolve_mapping(args, attend_result, image.shape[1], image.shape[0])
    guide = draw_patch_index_guide(
        image=image,
        grid_h=grid_h,
        grid_w=grid_w,
        crop_box=mapping["crop_box_original"],
        vision_input_size=mapping["vision_input_size"],
        alpha=args.alpha,
        font_scale=args.font_scale,
        line_width=args.line_width,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), guide):
        raise RuntimeError(f"Failed to write output image: {args.output}")

    print(f"image: {args.image}")
    print(f"size: {image.shape[1]}x{image.shape[0]}")
    print(f"vision_input_size: {mapping['vision_input_size'][0]}x{mapping['vision_input_size'][1]}")
    print(f"crop_box_original: {mapping['crop_box_original']}")
    print(f"patch_grid: {grid_h}x{grid_w}")
    print(f"index_range: 0..{grid_h * grid_w - 1}")
    print(f"output: {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw 0..N-1 patch indices on an image for manual mask debugging."
    )
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="Input overlay image.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output guide image.")
    parser.add_argument(
        "--attend-result",
        type=Path,
        default=DEFAULT_ATTEND_RESULT,
        help="JSON file used to auto-read patch_grid when --patch/--grid is omitted.",
    )
    parser.add_argument("--patch", type=int, help="Square grid side, e.g. 24 means 24x24 patches.")
    parser.add_argument(
        "--grid",
        type=str,
        help="Non-square grid in HxW form, e.g. 24x24. Overrides --attend-result.",
    )
    parser.add_argument("--alpha", type=float, default=0.22, help="White label background alpha.")
    parser.add_argument("--font-scale", type=float, default=0.36, help="OpenCV font scale for labels.")
    parser.add_argument("--line-width", type=int, default=1, help="Grid line width in pixels.")
    parser.add_argument(
        "--crop-box-original",
        type=str,
        help="Original-image crop box as left,top,right,bottom. Defaults to attend preprocess_meta.",
    )
    parser.add_argument(
        "--vision-input-size",
        type=str,
        help="Processed vision input size as WxH. Defaults to attend preprocess_meta, usually 336x336.",
    )
    return parser.parse_args()


def resolve_grid(args: argparse.Namespace, attend_result: dict[str, Any]) -> tuple[int, int]:
    if args.grid:
        return parse_grid(args.grid)
    if args.patch:
        if args.patch <= 0:
            raise ValueError("--patch must be positive.")
        return args.patch, args.patch
    patch_grid = attend_result.get("patch_grid")
    if isinstance(patch_grid, list) and len(patch_grid) == 2:
        return int(patch_grid[0]), int(patch_grid[1])
    raise ValueError("Cannot infer patch grid. Pass --patch 24 or --grid 24x24.")


def parse_grid(value: str) -> tuple[int, int]:
    normalized = value.lower().replace(",", "x")
    parts = normalized.split("x")
    if len(parts) != 2:
        raise ValueError("--grid must look like HxW, for example 24x24.")
    grid_h, grid_w = int(parts[0]), int(parts[1])
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError("--grid values must be positive.")
    return grid_h, grid_w


def resolve_mapping(
    args: argparse.Namespace,
    attend_result: dict[str, Any],
    image_w: int,
    image_h: int,
) -> dict[str, list[float] | list[int]]:
    preprocess_meta = attend_result.get("preprocess_meta") or attend_result.get("mask_origin_mapping_meta") or {}
    if args.crop_box_original:
        crop_box = parse_crop_box(args.crop_box_original)
    else:
        crop_box = preprocess_meta.get("crop_box_original")
        if not isinstance(crop_box, list) or len(crop_box) != 4:
            crop_size = min(image_w, image_h)
            left = max((image_w - crop_size) / 2.0, 0.0)
            top = max((image_h - crop_size) / 2.0, 0.0)
            crop_box = [left, top, left + crop_size, top + crop_size]
        crop_box = [float(value) for value in crop_box]

    if args.vision_input_size:
        vision_input_size = parse_size(args.vision_input_size)
    else:
        vision_input_size = preprocess_meta.get("vision_input_size", [336, 336])
        if not isinstance(vision_input_size, list) or len(vision_input_size) != 2:
            vision_input_size = [336, 336]
        vision_input_size = [int(vision_input_size[0]), int(vision_input_size[1])]

    return {
        "crop_box_original": crop_box,
        "vision_input_size": vision_input_size,
    }


def parse_crop_box(value: str) -> list[float]:
    parts = value.replace("x", ",").split(",")
    if len(parts) != 4:
        raise ValueError("--crop-box-original must look like left,top,right,bottom.")
    crop_box = [float(part) for part in parts]
    if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
        raise ValueError("--crop-box-original must satisfy right > left and bottom > top.")
    return crop_box


def parse_size(value: str) -> list[int]:
    parts = value.lower().replace(",", "x").split("x")
    if len(parts) != 2:
        raise ValueError("--vision-input-size must look like WxH, for example 336x336.")
    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError("--vision-input-size values must be positive.")
    return [width, height]


def draw_patch_index_guide(
    image: np.ndarray,
    grid_h: int,
    grid_w: int,
    crop_box: list[float],
    vision_input_size: list[int],
    alpha: float,
    font_scale: float,
    line_width: int,
) -> np.ndarray:
    height, width = image.shape[:2]
    input_w, input_h = vision_input_size
    left, top, right, bottom = crop_box
    x_edges = [vision_x_to_origin(x, left, right, input_w) for x in rounded_edges(input_w, grid_w)]
    y_edges = [vision_x_to_origin(y, top, bottom, input_h) for y in rounded_edges(input_h, grid_h)]
    x_edges = [clip_int(x, 0, width - 1) for x in x_edges]
    y_edges = [clip_int(y, 0, height - 1) for y in y_edges]
    result = image.copy()

    overlay = result.copy()
    for y in y_edges:
        cv2.line(overlay, (x_edges[0], y), (x_edges[-1], y), (0, 0, 0), line_width)
    for x in x_edges:
        cv2.line(overlay, (x, y_edges[0]), (x, y_edges[-1]), (0, 0, 0), line_width)
    result = cv2.addWeighted(overlay, 0.55, result, 0.45, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    total = grid_h * grid_w
    label_scale = choose_font_scale(
        max(1, x_edges[-1] - x_edges[0] + 1),
        max(1, y_edges[-1] - y_edges[0] + 1),
        grid_h,
        grid_w,
        total,
        font_scale,
    )
    text_thickness = 1

    for row in range(grid_h):
        for col in range(grid_w):
            idx = row * grid_w + col
            x0, x1 = x_edges[col], x_edges[col + 1]
            y0, y1 = y_edges[row], y_edges[row + 1]
            draw_centered_label(
                result,
                str(idx),
                (x0, y0, x1, y1),
                font=font,
                font_scale=label_scale,
                thickness=text_thickness,
                alpha=alpha,
            )
    return result


def vision_x_to_origin(value: int, start: float, end: float, input_length: int) -> int:
    if input_length <= 1:
        return int(round(start))
    ratio = value / float(input_length - 1)
    return int(round(start + ratio * (end - start)))


def clip_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def rounded_edges(length: int, parts: int) -> list[int]:
    edges = [round(i * length / parts) for i in range(parts + 1)]
    edges[0] = 0
    edges[-1] = length - 1
    return [int(edge) for edge in edges]


def choose_font_scale(
    width: int,
    height: int,
    grid_h: int,
    grid_w: int,
    total: int,
    requested_scale: float,
) -> float:
    cell_w = width / grid_w
    cell_h = height / grid_h
    longest = str(total - 1)
    scale = requested_scale
    for _ in range(40):
        (text_w, text_h), _ = cv2.getTextSize(longest, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        if text_w <= cell_w * 0.82 and text_h <= cell_h * 0.62:
            return scale
        scale *= 0.92
    return max(scale, 0.18)


def draw_centered_label(
    image: np.ndarray,
    text: str,
    cell: tuple[int, int, int, int],
    font: int,
    font_scale: float,
    thickness: int,
    alpha: float,
) -> None:
    x0, y0, x1, y1 = cell
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    text_x = int(cx - text_w / 2)
    text_y = int(cy + text_h / 2)
    pad_x = max(2, math.ceil(text_w * 0.18))
    pad_y = max(2, math.ceil(text_h * 0.18))

    bg_x0 = max(x0 + 1, text_x - pad_x)
    bg_y0 = max(y0 + 1, text_y - text_h - pad_y)
    bg_x1 = min(x1 - 1, text_x + text_w + pad_x)
    bg_y1 = min(y1 - 1, text_y + baseline + pad_y)

    label_layer = image.copy()
    cv2.rectangle(label_layer, (bg_x0, bg_y0), (bg_x1, bg_y1), (255, 255, 255), -1)
    cv2.addWeighted(label_layer, alpha, image, 1.0 - alpha, 0, dst=image)
    cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    main()
