from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


def main() -> None:
    args = parse_args()
    envision_dir = args.envision_dir
    attend_dir = args.attend_dir
    respond_dir = args.respond_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    envision_meta = load_json(envision_dir / "metadata.json")
    attend = load_json(attend_dir / "attend_result.json")
    respond = load_json(respond_dir / "respond_result.json")

    artifacts = copy_artifacts(envision_dir, attend_dir, respond_dir, output_dir, envision_meta, attend, respond)
    summary = build_summary(envision_dir, attend_dir, respond_dir, envision_meta, attend, respond, artifacts)
    (output_dir / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect and sanity-check one EnAR pipeline run.")
    parser.add_argument("--envision-dir", type=Path, default=Path("EnAR/outputs/envision/run_001"))
    parser.add_argument("--attend-dir", type=Path, default=Path("EnAR/outputs/attend/run_001"))
    parser.add_argument("--respond-dir", type=Path, default=Path("EnAR/outputs/respond/run_001"))
    parser.add_argument("--output-dir", type=Path, default=Path("EnAR/outputs/validation/run_001"))
    return parser.parse_args()


def copy_artifacts(
    envision_dir: Path,
    attend_dir: Path,
    respond_dir: Path,
    output_dir: Path,
    envision_meta: dict[str, Any],
    attend: dict[str, Any],
    respond: dict[str, Any],
) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    fixed = {
        "original_image": envision_dir / "original.png",
        "processed_image": envision_dir / "preprocessed.png",
        "representative_impression": envision_dir / "impression.png",
        "uncertainty_heatmap": envision_dir / "uncertainty_heatmap.png",
        "difference_image": envision_dir / "difference.png",
        "attention_heatmap": _path_from(attend, ["image_paths", "contrastive_attention_heatmap"]),
        "selected_mask": _path_from(attend, ["image_paths", "selected_patch_mask"]),
        "selected_overlay": _path_from(attend, ["image_paths", "patch_overlay"]),
        "selected_origin_overlay": _path_from(attend, ["image_paths", "mask_origin_overlay"]),
        "answer_regular": respond_dir / "answer_regular.txt",
        "answer_enar": respond_dir / "answer_enar.txt",
    }
    for name, path in fixed.items():
        copied = copy_if_exists(path, output_dir / f"{name}{Path(path).suffix if path else ''}")
        if copied:
            artifacts[name] = str(copied)

    samples_dir = output_dir / "visual_impressions"
    samples_dir.mkdir(exist_ok=True)
    for sample_path in envision_meta.get("outputs", {}).get("sample_images", []):
        path = Path(sample_path)
        copied = copy_if_exists(path, samples_dir / path.name)
        if copied:
            artifacts.setdefault("visual_impressions", []).append(str(copied))  # type: ignore[union-attr]
    return artifacts


def build_summary(
    envision_dir: Path,
    attend_dir: Path,
    respond_dir: Path,
    envision_meta: dict[str, Any],
    attend: dict[str, Any],
    respond: dict[str, Any],
    artifacts: dict[str, Any],
) -> dict[str, Any]:
    uncertainty = np.load(envision_dir / "uncertainty_map.npy")
    selected = attend.get("selected_patch_indices", [])
    patch_grid = attend.get("patch_grid") or respond.get("visual_token_layout", {}).get("patch_grid")
    selected_bbox = patch_bbox(selected, patch_grid)
    decode_trace_path = respond.get("decode_trace_path")
    first_step = None
    if decode_trace_path and Path(decode_trace_path).is_file():
        trace = load_json(Path(decode_trace_path)).get("steps", [])
        first_step = trace[0] if trace else None
    return {
        "envision_dir": str(envision_dir),
        "attend_dir": str(attend_dir),
        "respond_dir": str(respond_dir),
        "artifacts": artifacts,
        "key_parameters": {
            "ddim": {
                "num_ddim_steps": envision_meta.get("config", {}).get("num_ddim_steps"),
                "inversion_step_T": envision_meta.get("config", {}).get("inversion_step_T"),
                "timestep_T": envision_meta.get("timestep_T"),
                "langevin_steps_M": envision_meta.get("config", {}).get("langevin_steps_M"),
                "sample_count_K": envision_meta.get("config", {}).get("sample_count_K"),
                "eta_start": envision_meta.get("config", {}).get("eta_start"),
                "eta_end": envision_meta.get("config", {}).get("eta_end"),
                "temperature_tau": envision_meta.get("config", {}).get("temperature_tau"),
            },
            "attend": {
                "vision_layer_number": attend.get("vision_layer_number"),
                "attention_top_ratio": attend.get("attention_top_ratio"),
                "uncertainty_top_ratio": attend.get("uncertainty_top_ratio"),
                "padding_ratio_limit": attend.get("padding_ratio_limit"),
            },
            "respond": {
                "alpha": respond.get("alpha"),
                "padding_strategy": respond.get("padding_strategy"),
                "use_apc": respond.get("use_apc"),
                "apc_beta": respond.get("apc_beta"),
            },
        },
        "tensor_shapes_and_layout": {
            "uncertainty_map_shape": list(uncertainty.shape),
            "patch_grid": patch_grid,
            "has_cls_token_attend": attend.get("has_cls_token"),
            "visual_token_layout": respond.get("visual_token_layout"),
            "input_meta": respond.get("input_meta"),
        },
        "diagnostics": {
            "uncertainty_stats": stats(uncertainty),
            "envision_uncertainty_stats": envision_meta.get("uncertainty_stats"),
            "sample_diff_stats": envision_meta.get("sample_diff_stats"),
            "attend_score_stats": attend.get("score_stats"),
            "source_counts": attend.get("source_counts"),
            "selected_patch_count": len(selected),
            "selected_patch_bbox_grid_xyxy": selected_bbox,
            "padding_meta": respond.get("padding_meta"),
            "first_decode_step": first_step,
        },
        "answers": {
            "regular_answer": respond.get("regular_answer"),
            "enar_answer": respond.get("enar_answer"),
        },
    }


def patch_bbox(indices: list[int], patch_grid: Any) -> list[int] | None:
    if not indices or not isinstance(patch_grid, list) or len(patch_grid) != 2:
        return None
    h, w = int(patch_grid[0]), int(patch_grid[1])
    ys = [int(idx) // w for idx in indices]
    xs = [int(idx) % w for idx in indices]
    return [min(xs), min(ys), max(xs), max(ys)]


def stats(array: np.ndarray) -> dict[str, float]:
    arr = np.asarray(array, dtype=np.float32)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def _path_from(data: dict[str, Any], keys: list[str]) -> Path | None:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return Path(value) if value else None


def copy_if_exists(src: Path | None, dst: Path) -> Path | None:
    if src is None or not src.is_file():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    main()
