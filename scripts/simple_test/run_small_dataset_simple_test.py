#!/usr/bin/env python3
"""Run EnAR on the exported small counterfactual dataset.

This runner is intentionally independent from the parquet-based evaluator:
it reads EnAR/toy_dataset/small_dataset/*.jsonl records directly, keeps every
per-sample pipeline artifact, and writes compact distribution reports.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_PARENT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from enar_eval.evaluator import AnswerEvaluator
from enar_eval.reports import build_metrics, write_jsonl, write_reports
from pipeline.runner import EnARPipeline


DEFAULT_CONFIG = Path(__file__).with_name("simple_test_config.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EnAR on EnAR/toy_dataset/small_dataset and keep all stage artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Simple-test YAML config.")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Override dataset.root_dir.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override experiment.output_dir.")
    parser.add_argument("--run-name", default=None, help="Override experiment.run_name.")
    parser.add_argument(
        "--category",
        "--categories",
        dest="categories",
        action="append",
        default=None,
        help="Run only selected categories. Accepts names like Animals, Logos, Game_Boards, or Game Boards.",
    )
    parser.add_argument("--sample-id", action="append", default=None, help="Run only selected sample IDs.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit total samples after filtering.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute samples even when result.json exists.")
    parser.add_argument("--no-resume", action="store_true", help="Disable result.json resume lookup.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop immediately on the first failed sample.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_yaml(config_path)
    apply_cli_overrides(config, args)

    project_root = infer_enar_root(config_path)
    dataset_root = resolve_path(nested_get(config, ["dataset", "root_dir"], "toy_dataset/small_dataset"), project_root)
    run_dir = resolve_path(nested_get(config, ["experiment", "output_dir"], "outputs/simple_test"), project_root)
    run_dir = run_dir / str(nested_get(config, ["experiment", "run_name"], "run_small_dataset"))
    run_dir.mkdir(parents=True, exist_ok=True)

    samples = filter_samples(load_samples(dataset_root), config)
    write_json(run_dir / "resolved_simple_test_config.json", config)
    write_jsonl(run_dir / "sample_index.jsonl", [sample_public_dict(sample) for sample in samples])

    evaluator = AnswerEvaluator(dict(config.get("evaluation") or {}))
    records: list[dict[str, Any]] = []
    started = time.time()
    total = len(samples)
    log_interval = max(1, int(nested_get(config, ["runtime", "log_interval"], 1)))
    fail_fast = bool(nested_get(config, ["experiment", "fail_fast"], False))

    for index, sample in enumerate(samples, start=1):
        if index == 1 or index % log_interval == 0:
            print(f"[simple_test] {index}/{total} {sample['sample_id']}")
        try:
            record = run_sample(sample, config, dataset_root, run_dir, evaluator, project_root)
        except Exception as exc:
            if fail_fast:
                raise
            record = failed_record(sample, exc)
            sample_dir(sample, run_dir).mkdir(parents=True, exist_ok=True)
            write_json(sample_dir(sample, run_dir) / "result.json", record)
        records.append(record)
        write_jsonl(run_dir / "predictions.jsonl", records)

    metrics = build_metrics(records, dataset="small_dataset", split="jsonl")
    metrics["elapsed_seconds"] = round(time.time() - started, 4)
    metrics["answer_distribution"] = answer_distribution(records)
    metrics["outcome_distribution"] = outcome_distribution(records)
    metrics["artifact_policy"] = {
        "save_intermediate": True,
        "kept_per_sample": [
            "input.png",
            "sample.json",
            "pipeline_config.yaml",
            "pipeline/envision",
            "pipeline/attend",
            "pipeline/respond",
            "pipeline/pipeline_result.json",
        ],
    }
    write_reports(run_dir, records, metrics)
    write_distribution_csvs(run_dir, records)
    write_json(run_dir / "metrics.json", metrics)

    print(f"run_dir: {run_dir}")
    print(f"num_evaluated: {metrics['num_evaluated']}")
    print(f"regular_accuracy: {metrics['regular']['overall_accuracy']:.4f}")
    print(f"enar_accuracy: {metrics['enar']['overall_accuracy']:.4f}")
    print(f"regular_expected_bias_rate: {metrics['regular']['expected_bias_rate']:.4f}")
    print(f"enar_expected_bias_rate: {metrics['enar']['expected_bias_rate']:.4f}")
    return 0


def run_sample(
    sample: dict[str, Any],
    config: dict[str, Any],
    dataset_root: Path,
    run_dir: Path,
    evaluator: AnswerEvaluator,
    project_root: Path,
) -> dict[str, Any]:
    out_dir = sample_dir(sample, run_dir)
    result_path = out_dir / "result.json"
    resume = bool(nested_get(config, ["experiment", "resume"], True))
    overwrite = bool(nested_get(config, ["experiment", "overwrite"], False))
    if resume and not overwrite and result_path.is_file():
        cached = load_json(result_path)
        if cached.get("status") == "ok":
            return cached

    if overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "sample.json", sample_public_dict(sample))

    input_image = out_dir / "input.png"
    source_image = dataset_root / str(sample["relative_image_path"])
    if not source_image.is_file():
        raise FileNotFoundError(f"Sample image does not exist: {source_image}")
    shutil.copyfile(source_image, input_image)

    pipeline_yaml = out_dir / "pipeline_config.yaml"
    pipeline_config = build_pipeline_config(sample, input_image, out_dir / "pipeline", config, project_root)
    write_yaml(pipeline_yaml, pipeline_config)

    result = EnARPipeline.from_yaml(pipeline_yaml).run()
    pipeline_meta = load_json(Path(result.metadata_path))
    respond = pipeline_meta.get("respond", {})
    regular_answer = str(respond.get("regular_answer", ""))
    enar_answer = str(respond.get("enar_answer", ""))
    metadata = parse_metadata(sample.get("metadata"))

    regular_eval = evaluator.evaluate(
        regular_answer,
        str(sample.get("ground_truth", "")),
        str(sample.get("expected_bias", "")),
        type_of_question=str(sample.get("type_of_question", "")),
        metadata=metadata,
    )
    enar_eval = evaluator.evaluate(
        enar_answer,
        str(sample.get("ground_truth", "")),
        str(sample.get("expected_bias", "")),
        type_of_question=str(sample.get("type_of_question", "")),
        metadata=metadata,
    )
    paths = {
        "input_image": str(input_image),
        "sample_json": str(out_dir / "sample.json"),
        "pipeline_config": str(pipeline_yaml),
        "pipeline_result": str(result.metadata_path),
        "envision": str(out_dir / "pipeline" / "envision"),
        "attend": str(out_dir / "pipeline" / "attend"),
        "respond": str(out_dir / "pipeline" / "respond"),
    }
    record = {
        **sample_public_dict(sample),
        "regular": regular_eval.to_dict(),
        "enar": enar_eval.to_dict(),
        "paths": paths,
        "stage_summary": stage_summary(pipeline_meta),
        "status": "ok",
        "error": None,
    }
    write_json(result_path, record)
    return record


def build_pipeline_config(
    sample: dict[str, Any],
    input_image: Path,
    output_dir: Path,
    config: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    pipeline = dict(config.get("pipeline") or {})
    prompt_config = dict(pipeline.get("prompt") or {})
    stages = dict(pipeline.get("stages") or {})
    stages = force_keep_intermediates(stages)
    runtime = dict(config.get("runtime") or {})
    models = dict(config.get("models") or {})
    return {
        "paths": {
            "input_image": str(input_image),
            "output_dir": str(output_dir),
            "sd_model_dir": str(resolve_path(models.get("sd_model_dir", "pre_model/DDIM/stable-diffusion-v1-5"), project_root)),
            "llava_model_dir": str(resolve_path(models.get("llava_model_dir", "pre_model/LLM/llava-1.5-7b-hf"), project_root)),
        },
        "prompt": {
            "question": str(sample.get("prompt", "")),
            "envision_prompt": str(prompt_config.get("envision_prompt", "")),
            "negative_prompt": str(prompt_config.get("negative_prompt", "")),
        },
        "runtime": {
            "run_name": None,
            "device": runtime.get("device", "auto"),
            "dtype": runtime.get("dtype", "float16"),
            "seed": runtime.get("seed", 42),
        },
        "stages": stages,
    }


def force_keep_intermediates(stages: dict[str, Any]) -> dict[str, Any]:
    data = json.loads(json.dumps(stages))
    attend = data.setdefault("attend", {})
    visualization = attend.setdefault("visualization", {})
    visualization.update(
        {
            "save_raw_arrays": True,
            "save_heatmaps": True,
            "save_patch_overlay": True,
            "save_mask_origin": True,
            "save_source_masks": True,
        }
    )
    respond = data.setdefault("respond", {})
    contrastive = respond.setdefault("contrastive", {})
    contrastive["save_decode_trace"] = True
    return data


def load_samples(dataset_root: Path) -> list[dict[str, Any]]:
    index_path = dataset_root / "sample_index.jsonl"
    if index_path.is_file():
        rows = read_jsonl(index_path)
    else:
        rows = []
        for samples_path in sorted(dataset_root.glob("*/samples.jsonl")):
            topic_dir = samples_path.parent.name
            rows.extend({"topic_dir": topic_dir, **row} for row in read_jsonl(samples_path))
    if not rows:
        raise FileNotFoundError(f"No samples found under {dataset_root}")
    return rows


def filter_samples(samples: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    dataset = dict(config.get("dataset") or {})
    out = list(samples)
    categories = {normalize_category(value) for value in dataset.get("categories") or []}
    if categories:
        out = [
            sample
            for sample in out
            if normalize_category(sample.get("topic")) in categories
            or normalize_category(sample.get("topic_dir")) in categories
        ]
    sample_ids = {str(value) for value in dataset.get("sample_ids") or []}
    if sample_ids:
        out = [sample for sample in out if str(sample.get("sample_id")) in sample_ids or str(sample.get("raw_id")) in sample_ids]
    max_samples = dataset.get("max_samples")
    if max_samples is not None:
        out = out[: int(max_samples)]
    if not out:
        raise ValueError("No samples remain after filtering.")
    return out


def answer_distribution(records: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for method in ("regular", "enar"):
        overall = Counter(record.get(method, {}).get("normalized_answer", "") for record in records if record.get("status") == "ok")
        by_topic: dict[str, Counter[str]] = defaultdict(Counter)
        for record in records:
            if record.get("status") != "ok":
                continue
            by_topic[str(record.get("topic", ""))][record.get(method, {}).get("normalized_answer", "")] += 1
        output[method] = {
            "overall": dict(overall.most_common()),
            "by_topic": {topic: dict(counts.most_common()) for topic, counts in sorted(by_topic.items())},
        }
    return output


def outcome_distribution(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        if record.get("status") != "ok":
            counts["failed"] += 1
            continue
        regular = record.get("regular", {})
        enar = record.get("enar", {})
        regular_correct = bool(regular.get("correct"))
        enar_correct = bool(enar.get("correct"))
        regular_bias = bool(regular.get("hits_expected_bias"))
        enar_bias = bool(enar.get("hits_expected_bias"))
        if regular_correct and enar_correct:
            counts["both_correct"] += 1
        elif regular_correct and not enar_correct:
            counts["regular_only_correct"] += 1
        elif not regular_correct and enar_correct:
            counts["enar_only_correct"] += 1
        else:
            counts["both_wrong"] += 1
        if regular_bias and enar_correct:
            counts["regular_bias_to_enar_correct"] += 1
        if regular_correct and enar_bias:
            counts["regular_correct_to_enar_bias"] += 1
        if regular_bias and enar_bias:
            counts["both_hit_expected_bias"] += 1
    return dict(counts)


def write_distribution_csvs(run_dir: Path, records: list[dict[str, Any]]) -> None:
    rows = []
    for record in records:
        if record.get("status") != "ok":
            continue
        rows.append(
            {
                "sample_id": record.get("sample_id"),
                "topic": record.get("topic"),
                "sub_topic": record.get("sub_topic"),
                "type_of_question": record.get("type_of_question"),
                "pixel": record.get("pixel"),
                "ground_truth": record.get("ground_truth"),
                "expected_bias": record.get("expected_bias"),
                "regular_answer": record.get("regular", {}).get("answer"),
                "regular_normalized": record.get("regular", {}).get("normalized_answer"),
                "regular_correct": record.get("regular", {}).get("correct"),
                "regular_hits_bias": record.get("regular", {}).get("hits_expected_bias"),
                "enar_answer": record.get("enar", {}).get("answer"),
                "enar_normalized": record.get("enar", {}).get("normalized_answer"),
                "enar_correct": record.get("enar", {}).get("correct"),
                "enar_hits_bias": record.get("enar", {}).get("hits_expected_bias"),
                "selected_patch_count": nested_get(record, ["stage_summary", "attend", "selected_patch_count"], ""),
                "selected_vision_token_count": nested_get(record, ["stage_summary", "attend", "selected_vision_token_count"], ""),
            }
        )
    write_csv(run_dir / "answer_distribution.csv", rows)


def stage_summary(pipeline_meta: dict[str, Any]) -> dict[str, Any]:
    attend = dict(pipeline_meta.get("attend") or {})
    respond = dict(pipeline_meta.get("respond") or {})
    return {
        "envision": dict(pipeline_meta.get("envision") or {}),
        "attend": {
            "selected_patch_count": attend.get("selected_patch_count"),
            "selected_vision_token_count": attend.get("selected_vision_token_count"),
            "mask_origin_path": attend.get("mask_origin_path"),
            "patch_overlay_path": attend.get("patch_overlay_path"),
            "attend_result_json": attend.get("attend_result_json"),
        },
        "respond": {
            "respond_result_json": respond.get("respond_result_json"),
            "token_logits_trace_path": respond.get("token_logits_trace_path"),
        },
    }


def sample_public_dict(sample: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "sample_id",
        "raw_id",
        "split",
        "relative_image_path",
        "ID",
        "image_path",
        "topic",
        "topic_dir",
        "sub_topic",
        "prompt",
        "ground_truth",
        "expected_bias",
        "with_title",
        "type_of_question",
        "pixel",
        "metadata",
        "source_row_index",
    ]
    return {key: sample.get(key) for key in keys if key in sample}


def failed_record(sample: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        **sample_public_dict(sample),
        "regular": {"answer": "", "normalized_answer": "", "correct": False, "hits_expected_bias": False},
        "enar": {"answer": "", "normalized_answer": "", "correct": False, "hits_expected_bias": False},
        "paths": {},
        "stage_summary": {},
        "status": "failed",
        "error": f"{type(exc).__name__}: {exc}",
    }


def sample_dir(sample: dict[str, Any], run_dir: Path) -> Path:
    topic = normalize_category(sample.get("topic_dir") or sample.get("topic") or "unknown")
    return run_dir / "samples" / topic / sanitize_path_name(str(sample["sample_id"]))


def parse_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {"raw": value}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    if args.dataset_root is not None:
        nested_set(config, ["dataset", "root_dir"], str(args.dataset_root))
    if args.output_dir is not None:
        nested_set(config, ["experiment", "output_dir"], str(args.output_dir))
    if args.run_name is not None:
        nested_set(config, ["experiment", "run_name"], args.run_name)
    if args.categories is not None:
        nested_set(config, ["dataset", "categories"], args.categories)
    if args.sample_id is not None:
        nested_set(config, ["dataset", "sample_ids"], args.sample_id)
    if args.max_samples is not None:
        nested_set(config, ["dataset", "max_samples"], args.max_samples)
    if args.overwrite:
        nested_set(config, ["experiment", "overwrite"], True)
    if args.no_resume:
        nested_set(config, ["experiment", "resume"], False)
    if args.fail_fast:
        nested_set(config, ["experiment", "fail_fast"], True)


def infer_enar_root(config_path: Path) -> Path:
    for parent in (config_path.parent, *config_path.parents):
        if parent.name == "EnAR":
            return parent
    cwd = Path.cwd().resolve()
    return cwd if cwd.name == "EnAR" else cwd / "EnAR"


def resolve_path(value: Any, project_root: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    if path.parts and path.parts[0] == project_root.name:
        return (project_root.parent / path).resolve()
    return (project_root / path).resolve()


def normalize_category(value: Any) -> str:
    return sanitize_path_name(str(value or "")).lower()


def sanitize_path_name(value: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value.strip())
    while "__" in clean:
        clean = clean.replace("__", "_")
    return clean.strip("._") or "unknown"


def nested_get(data: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def nested_set(data: dict[str, Any], keys: list[str], value: Any) -> None:
    current = data
    for key in keys[:-1]:
        next_value = current.setdefault(key, {})
        if not isinstance(next_value, dict):
            next_value = {}
            current[key] = next_value
        current = next_value
    current[keys[-1]] = value


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Config root must be a mapping: {path}")
    return data


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return data


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
