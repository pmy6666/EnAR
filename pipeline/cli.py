from __future__ import annotations

import argparse
from pathlib import Path

from .config import EnARPipelineConfig
from .runner import EnARPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full EnAR Envision-Attend-Respond pipeline.")
    parser.add_argument("--config", type=Path, required=True, help="Pipeline YAML config file.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = EnARPipelineConfig.from_yaml(args.config)
    result = EnARPipeline(config).run()
    print(f"regular_answer: {result.respond.regular_answer}")
    print(f"enar_answer: {result.respond.enar_answer}")
    print(f"pipeline_result: {result.metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
