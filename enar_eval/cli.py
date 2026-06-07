from __future__ import annotations

import argparse
from pathlib import Path

from .config import EvalConfig
from .runner import VLMBiasEvalRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run EnAR evaluation on local VLMBias parquet files.")
    parser.add_argument("--config", type=Path, required=True, help="Evaluation YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Use deterministic fake predictions to validate IO and metrics.")
    parser.add_argument("--max-samples", type=int, default=None, help="Override dataset.filters.max_samples.")
    parser.add_argument("--split", type=str, default=None, help="Override dataset split.")
    parser.add_argument("--run-name", type=str, default=None, help="Override experiment.run_name.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute samples even when result.json exists.")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume cache lookup.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = EvalConfig.from_yaml(args.config)
    if args.dry_run:
        config.experiment.dry_run = True
    if args.max_samples is not None:
        config.dataset.filters.max_samples = args.max_samples
    if args.split:
        config.dataset.split = args.split
    if args.run_name:
        config.experiment.run_name = args.run_name
    if args.overwrite:
        config.experiment.overwrite = True
    if args.no_resume:
        config.experiment.resume = False

    metrics = VLMBiasEvalRunner(config).run()
    print(f"run_dir: {config.run_dir}")
    print(f"num_evaluated: {metrics['num_evaluated']}")
    print(f"regular_accuracy: {metrics['regular']['overall_accuracy']:.4f}")
    print(f"enar_accuracy: {metrics['enar']['overall_accuracy']:.4f}")
    print(f"delta_accuracy: {metrics['delta']['overall_accuracy']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
