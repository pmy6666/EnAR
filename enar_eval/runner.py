from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import yaml

from .cache import file_sha256, read_json, stable_hash, write_json
from .config import EvalConfig
from .dataset import VLMBiasSample, export_sample_image, load_vlmbias_samples
from .evaluator import AnswerEvaluator
from .reports import build_metrics, write_jsonl, write_reports


class VLMBiasEvalRunner:
    def __init__(self, config: EvalConfig) -> None:
        self.config = config
        self.evaluator = AnswerEvaluator(config.evaluation)
        self.config_hash = config.config_hash()

    def run(self) -> dict[str, Any]:
        started = time.time()
        run_dir = self.config.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        self.config.save_resolved_yaml(run_dir / "resolved_config.yaml")

        samples, manifest = load_vlmbias_samples(self.config.dataset)
        manifest["config_hash"] = self.config_hash
        manifest["dry_run"] = self.config.experiment.dry_run
        write_json(run_dir / "dataset_manifest.json", manifest)
        write_jsonl(run_dir / "sample_index.jsonl", [sample.to_public_dict() for sample in samples])

        records: list[dict[str, Any]] = []
        total = len(samples)
        for index, sample in enumerate(samples, start=1):
            if self._should_log(index):
                print(f"[enar_eval] {index}/{total} {sample.sample_id}")
            try:
                record = self._run_sample(sample)
            except Exception as exc:
                if self.config.runtime.fail_fast:
                    raise
                record = self._failed_record(sample, exc)
                write_json(self._sample_dir(sample) / "result.json", record)
                self._cleanup_sample_intermediates(self._sample_dir(sample))
            records.append(record)

        metrics = build_metrics(records, dataset=self.config.dataset.name, split=self.config.dataset.split)
        metrics["elapsed_seconds"] = round(time.time() - started, 4)
        metrics["config_hash"] = self.config_hash
        write_reports(run_dir, records, metrics)
        return metrics

    def _run_sample(self, sample: VLMBiasSample) -> dict[str, Any]:
        sample_dir = self._sample_dir(sample)
        result_path = sample_dir / "result.json"
        if self.config.experiment.resume and not self.config.experiment.overwrite:
            cached = read_json(result_path)
            if cached and cached.get("config_hash") == self.config_hash and cached.get("status") == "ok":
                return cached

        sample_dir.mkdir(parents=True, exist_ok=True)
        write_json(sample_dir / "sample.json", sample.to_public_dict())
        input_image = sample_dir / "input.png"
        if self.config.dataset.image_export.get("write_input_image", True):
            export_sample_image(sample, input_image, self.config.dataset.root_dir)
        image_hash = file_sha256(input_image) if input_image.is_file() else ""

        if self.config.experiment.dry_run:
            answers = self._dry_run_answers(sample)
            paths = {"input_image": str(input_image)}
        else:
            answers, paths = self._run_enar_pipeline(sample, input_image, sample_dir)

        regular = self.evaluator.evaluate(
            answers.get("regular_answer", ""),
            sample.ground_truth,
            sample.expected_bias,
            type_of_question=sample.type_of_question,
            metadata=sample.metadata,
        )
        enar = self.evaluator.evaluate(
            answers.get("enar_answer", ""),
            sample.ground_truth,
            sample.expected_bias,
            type_of_question=sample.type_of_question,
            metadata=sample.metadata,
        )
        record = {
            **sample.to_public_dict(),
            "regular": regular.to_dict(),
            "enar": enar.to_dict(),
            "paths": self._retained_paths(sample, paths),
            "status": "ok",
            "error": None,
            "config_hash": self.config_hash,
            "image_hash": image_hash,
        }
        write_json(result_path, record)
        self._cleanup_sample_intermediates(sample_dir)
        return record

    def _run_enar_pipeline(
        self,
        sample: VLMBiasSample,
        input_image: Path,
        sample_dir: Path,
    ) -> tuple[dict[str, str], dict[str, str]]:
        from pipeline.runner import EnARPipeline

        pipeline_yaml = sample_dir / "pipeline_config.yaml"
        pipeline_config = self._build_pipeline_config(sample, input_image, sample_dir / "pipeline")
        with pipeline_yaml.open("w", encoding="utf-8") as f:
            yaml.safe_dump(pipeline_config, f, sort_keys=False, allow_unicode=True)

        result = EnARPipeline.from_yaml(pipeline_yaml).run()
        metadata_path = Path(result.metadata_path)
        metadata = read_json(metadata_path) or {}
        respond = metadata.get("respond", {})
        answers = {
            "regular_answer": str(respond.get("regular_answer", "")),
            "enar_answer": str(respond.get("enar_answer", "")),
        }
        paths = {
            "input_image": str(input_image),
            "pipeline_config": str(pipeline_yaml),
            "pipeline_result": str(metadata_path),
            "envision": str(sample_dir / "pipeline" / "envision"),
            "attend": str(sample_dir / "pipeline" / "attend"),
            "respond": str(sample_dir / "pipeline" / "respond"),
        }
        return answers, paths

    def _build_pipeline_config(self, sample: VLMBiasSample, input_image: Path, output_dir: Path) -> dict[str, Any]:
        pipeline = self.config.pipeline
        prompt_config = pipeline.get("prompt", {}) if isinstance(pipeline.get("prompt", {}), dict) else {}
        stages = pipeline.get("stages", {}) if isinstance(pipeline.get("stages", {}), dict) else {}
        return {
            "paths": {
                "input_image": str(input_image),
                "output_dir": str(output_dir),
                "sd_model_dir": str(self.config.models.sd_model_dir),
                "llava_model_dir": str(self.config.models.llava_model_dir),
            },
            "prompt": {
                "question": sample.prompt,
                "envision_prompt": str(prompt_config.get("envision_prompt", "")),
                "negative_prompt": str(prompt_config.get("negative_prompt", "")),
            },
            "runtime": {
                "run_name": None,
                "device": self.config.runtime.device,
                "dtype": self.config.runtime.dtype,
                "seed": self.config.experiment.seed,
            },
            "stages": _strip_stage_enabled(stages),
        }

    def _dry_run_answers(self, sample: VLMBiasSample) -> dict[str, str]:
        strategy = (
            self.config.evaluation.get("dry_run_predictions", {}).get("strategy")
            if isinstance(self.config.evaluation.get("dry_run_predictions", {}), dict)
            else None
        )
        if strategy == "blank":
            return {"regular_answer": "", "enar_answer": ""}
        if strategy == "ground_truth":
            return {"regular_answer": sample.ground_truth, "enar_answer": sample.ground_truth}
        return {
            "regular_answer": sample.expected_bias,
            "enar_answer": sample.ground_truth,
        }

    def _failed_record(self, sample: VLMBiasSample, exc: Exception) -> dict[str, Any]:
        return {
            **sample.to_public_dict(),
            "regular": {"answer": "", "normalized_answer": "", "correct": False, "hits_expected_bias": False},
            "enar": {"answer": "", "normalized_answer": "", "correct": False, "hits_expected_bias": False},
            "paths": {},
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "config_hash": self.config_hash,
        }

    def _sample_dir(self, sample: VLMBiasSample) -> Path:
        return self.config.run_dir / "samples" / sample.sample_id

    def _should_log(self, index: int) -> bool:
        interval = max(1, int(self.config.runtime.log_interval))
        return index == 1 or index % interval == 0

    def _retained_paths(self, sample: VLMBiasSample, paths: dict[str, str]) -> dict[str, str]:
        if self.config.experiment.save_intermediate:
            return paths
        return {
            "result_json": str(self._sample_dir(sample) / "result.json"),
        }

    def _cleanup_sample_intermediates(self, sample_dir: Path) -> None:
        if self.config.experiment.save_intermediate:
            return
        for file_name in ("input.png", "sample.json", "pipeline_config.yaml"):
            path = sample_dir / file_name
            if path.exists():
                path.unlink()
        pipeline_dir = sample_dir / "pipeline"
        if pipeline_dir.exists():
            shutil.rmtree(pipeline_dir)


def _strip_stage_enabled(stages: dict[str, Any]) -> dict[str, Any]:
    output = json.loads(json.dumps(stages))
    for stage_data in output.values():
        if isinstance(stage_data, dict):
            stage_data.pop("enabled", None)
    return output
