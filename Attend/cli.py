from __future__ import annotations

import argparse
from pathlib import Path

from .config import AttendConfig
from .pipeline import AttendPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run EnAR Attend stage.")
    parser.add_argument("--config", type=Path, help="YAML config file.")
    parser.add_argument("--original_image", type=Path)
    parser.add_argument("--impression_image", type=Path)
    parser.add_argument("--uncertainty_map", type=Path)
    parser.add_argument("--envision_metadata", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--llava_model_dir", type=Path)
    parser.add_argument("--vision_layer_number", type=int)
    parser.add_argument("--attention_top_ratio", type=float)
    parser.add_argument("--uncertainty_top_ratio", type=float)
    parser.add_argument("--padding_ratio_limit", type=float)
    parser.add_argument("--uncertainty_weight", type=float)
    parser.add_argument("--device", type=str)
    parser.add_argument("--dtype", choices=["float16", "float32", "bfloat16"])
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = AttendConfig.from_yaml(args.config) if args.config else AttendConfig()
    overrides = {
        key: value
        for key, value in vars(args).items()
        if key != "config" and value is not None
    }
    for key, value in overrides.items():
        setattr(config, key, value)
    config.__post_init__()

    result = AttendPipeline(config).run()
    print(f"selected patches: {len(result.selected_patch_indices)}")
    print(f"mask_origin: {result.mask_origin_path}")
    print(f"patch_overlay: {result.patch_overlay_path}")
    print(f"result_json: {result.attend_result_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
