from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .cache import write_json


METHODS = ("regular", "enar")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_metrics(records: list[dict[str, Any]], *, dataset: str, split: str) -> dict[str, Any]:
    evaluated = [record for record in records if record.get("status") == "ok"]
    metrics: dict[str, Any] = {
        "dataset": dataset,
        "split": split,
        "num_total": len(records),
        "num_evaluated": len(evaluated),
        "regular": _method_summary(evaluated, "regular"),
        "enar": _method_summary(evaluated, "enar"),
    }
    metrics["delta"] = {
        "overall_accuracy": metrics["enar"]["overall_accuracy"] - metrics["regular"]["overall_accuracy"],
        "expected_bias_rate": metrics["enar"]["expected_bias_rate"] - metrics["regular"]["expected_bias_rate"],
    }
    for key, field in (
        ("by_topic", "topic"),
        ("by_sub_topic", "sub_topic"),
        ("by_question_type", "type_of_question"),
        ("by_with_title", "with_title"),
        ("by_pixel", "pixel"),
    ):
        metrics[key] = _group_summary(evaluated, field)
    return metrics


def write_reports(run_dir: str | Path, records: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    run_dir = Path(run_dir)
    write_jsonl(run_dir / "predictions.jsonl", records)
    write_json(run_dir / "metrics.json", metrics)
    _write_group_csv(run_dir / "metrics_by_topic.csv", metrics.get("by_topic", {}), "topic")
    _write_group_csv(run_dir / "metrics_by_sub_topic.csv", metrics.get("by_sub_topic", {}), "sub_topic")
    _write_group_csv(run_dir / "metrics_by_question_type.csv", metrics.get("by_question_type", {}), "type_of_question")
    _write_group_csv(run_dir / "metrics_by_with_title.csv", metrics.get("by_with_title", {}), "with_title")
    _write_group_csv(run_dir / "metrics_by_pixel.csv", metrics.get("by_pixel", {}), "pixel")
    write_jsonl(run_dir / "error_cases.jsonl", _error_cases(records))
    _write_markdown_report(run_dir / "report.md", metrics, records)


def _method_summary(records: list[dict[str, Any]], method: str) -> dict[str, Any]:
    total = len(records)
    correct = sum(1 for record in records if record.get(method, {}).get("correct") is True)
    bias_hits = sum(1 for record in records if record.get(method, {}).get("hits_expected_bias") is True)
    return {
        "evaluated_count": total,
        "correct_count": correct,
        "overall_accuracy": correct / total if total else 0.0,
        "expected_bias_hits": bias_hits,
        "expected_bias_rate": bias_hits / total if total else 0.0,
    }


def _group_summary(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get(field, ""))].append(record)
    output: dict[str, Any] = {}
    for value, rows in sorted(groups.items(), key=lambda item: item[0]):
        regular = _method_summary(rows, "regular")
        enar = _method_summary(rows, "enar")
        output[value] = {
            "count": len(rows),
            "regular_accuracy": regular["overall_accuracy"],
            "enar_accuracy": enar["overall_accuracy"],
            "delta_accuracy": enar["overall_accuracy"] - regular["overall_accuracy"],
            "regular_expected_bias_rate": regular["expected_bias_rate"],
            "enar_expected_bias_rate": enar["expected_bias_rate"],
            "delta_expected_bias_rate": enar["expected_bias_rate"] - regular["expected_bias_rate"],
        }
    return output


def _write_group_csv(path: Path, groups: dict[str, Any], label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        label,
        "count",
        "regular_accuracy",
        "enar_accuracy",
        "delta_accuracy",
        "regular_expected_bias_rate",
        "enar_expected_bias_rate",
        "delta_expected_bias_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for value, data in groups.items():
            row = {label: value}
            row.update(data)
            writer.writerow(row)


def _error_cases(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    errors = []
    for record in records:
        if record.get("status") != "ok":
            errors.append(record)
            continue
        regular_ok = record.get("regular", {}).get("correct")
        enar_ok = record.get("enar", {}).get("correct")
        if not regular_ok or not enar_ok:
            errors.append(record)
    return errors


def _write_markdown_report(path: Path, metrics: dict[str, Any], records: list[dict[str, Any]]) -> None:
    regular = metrics["regular"]
    enar = metrics["enar"]
    delta = metrics["delta"]
    lines = [
        "# EnAR VLMBias Evaluation Report",
        "",
        "## Run Summary",
        "",
        f"- Dataset: {metrics['dataset']}",
        f"- Split: {metrics['split']}",
        f"- Total samples: {metrics['num_total']}",
        f"- Evaluated samples: {metrics['num_evaluated']}",
        "",
        "## Overall Metrics",
        "",
        "| Method | Accuracy | Expected Bias Rate | Correct / Evaluated |",
        "| --- | ---: | ---: | ---: |",
        f"| Regular | {_pct(regular['overall_accuracy'])} | {_pct(regular['expected_bias_rate'])} | {regular['correct_count']} / {regular['evaluated_count']} |",
        f"| EnAR | {_pct(enar['overall_accuracy'])} | {_pct(enar['expected_bias_rate'])} | {enar['correct_count']} / {enar['evaluated_count']} |",
        f"| Delta | {_signed_pct(delta['overall_accuracy'])} | {_signed_pct(delta['expected_bias_rate'])} |  |",
        "",
        "## Topic Accuracy",
        "",
        "| Topic | Count | Regular | EnAR | Delta | Regular Bias | EnAR Bias |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for topic, row in metrics.get("by_topic", {}).items():
        lines.append(
            f"| {topic} | {row['count']} | {_pct(row['regular_accuracy'])} | "
            f"{_pct(row['enar_accuracy'])} | {_signed_pct(row['delta_accuracy'])} | "
            f"{_pct(row['regular_expected_bias_rate'])} | {_pct(row['enar_expected_bias_rate'])} |"
        )
    lines.extend([
        "",
        "## Error Index",
        "",
        "See `error_cases.jsonl` for samples where either method is incorrect or a sample failed.",
        "",
        "## Output Files",
        "",
        "- `predictions.jsonl`: one row per sample with answers and correctness flags.",
        "- `metrics.json`: machine-readable overall and grouped metrics.",
        "- `metrics_by_*.csv`: grouped summary tables.",
        "- `samples/{sample_id}/`: exported input image, per-sample result, and pipeline artifacts.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _signed_pct(value: float) -> str:
    return f"{100.0 * value:+.2f}%"
