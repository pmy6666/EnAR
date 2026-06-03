from __future__ import annotations

import argparse
from pathlib import Path

from .config import EnvisionConfig
from .pipeline import EnvisionPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run EnAR Envision stage on one image.")
    parser.add_argument("--config", type=Path, help="YAML config file.")
    parser.add_argument("--input_image", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--sd_model_dir", type=Path)
    parser.add_argument("--image_size", type=int)
    parser.add_argument("--num_ddim_steps", type=int)
    parser.add_argument("--inversion_step_T", type=int)
    parser.add_argument("--langevin_steps_M", type=int)
    parser.add_argument("--sample_count_K", type=int)
    parser.add_argument("--eta_start", type=float)
    parser.add_argument("--eta_end", type=float)
    parser.add_argument("--temperature_tau", type=float)
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--negative_prompt", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dtype", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--device", type=str)
    parser.add_argument("--guidance_scale", type=float)
    parser.add_argument("--debug", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = EnvisionConfig.from_yaml(args.config) if args.config else EnvisionConfig()
    overrides = {
        key: value
        for key, value in vars(args).items()
        if key != "config" and value is not None
    }
    for key, value in overrides.items():
        setattr(config, key, value)
    config.__post_init__()

    result = EnvisionPipeline(config).run()
    print(f"impression: {result.impression_image_path}")
    print(f"uncertainty_heatmap: {result.uncertainty_heatmap_path}")
    print(f"metadata: {result.metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
