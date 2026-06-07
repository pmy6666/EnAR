from __future__ import annotations

import argparse
from pathlib import Path

from .config import RespondConfig
from .pipeline import RespondPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run EnAR Respond stage.")
    parser.add_argument("--config", type=Path, help="YAML config file.")
    parser.add_argument("--llava_model_dir", type=Path)
    parser.add_argument("--image_path", type=Path)
    parser.add_argument("--question", type=str)
    parser.add_argument("--attend_result_json", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--max_new_tokens", type=int)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top_p", type=float)
    parser.add_argument("--use_apc", action="store_true")
    parser.add_argument("--apc_beta", type=float)
    parser.add_argument(
        "--padding_strategy",
        choices=[
            "pad_token_embedding",
            "zero_embedding",
            "mean_visual_embedding",
            "matched_mean_visual_embedding",
        ],
    )
    parser.add_argument("--device", type=str)
    parser.add_argument("--dtype", choices=["float16", "float32", "bfloat16"])
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = RespondConfig.from_yaml(args.config) if args.config else RespondConfig()
    overrides = {
        key: value
        for key, value in vars(args).items()
        if key != "config" and value is not None
    }
    for key, value in overrides.items():
        setattr(config, key, value)
    config.__post_init__()

    result = RespondPipeline(config).run()
    print(f"regular_answer: {result.regular_answer}")
    print(f"enar_answer: {result.enar_answer}")
    print(f"result_json: {result.respond_result_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
