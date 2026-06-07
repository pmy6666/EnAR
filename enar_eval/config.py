from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .cache import stable_hash


@dataclass
class ExperimentConfig:
    name: str = "vlmbias_enar_eval"
    run_name: str = "run_001"
    seed: int = 42
    output_dir: Path = Path("outputs/enar_eval/vlmbias")
    resume: bool = True
    overwrite: bool = False
    save_intermediate: bool = True
    dry_run: bool = False


@dataclass
class DatasetFilters:
    max_samples: int | None = None
    sample_ids: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    sub_topics: list[str] = field(default_factory=list)
    type_of_question: list[str] = field(default_factory=list)
    with_title: bool | None = None
    pixel: int | None = None


@dataclass
class DatasetConfig:
    name: str = "VLMBias"
    root_dir: Path = Path("toy_dataset/VLMBias")
    data_dir: Path = Path("toy_dataset/VLMBias/data")
    split: str = "main"
    data_files: dict[str, str] = field(default_factory=lambda: {
        "main": "data/main-*.parquet",
        "identification": "data/identification-*.parquet",
        "withtitle": "data/withtitle-*.parquet",
        "original": "data/original-*.parquet",
        "remove_background_q1q2": "data/remove_background_q1q2-*.parquet",
        "remove_background_q3": "data/remove_background_q3-*.parquet",
    })
    filters: DatasetFilters = field(default_factory=DatasetFilters)
    image_export: dict[str, Any] = field(default_factory=lambda: {
        "format": "png",
        "write_input_image": True,
    })


@dataclass
class ModelsConfig:
    llava_model_dir: Path = Path("pre_model/LLM/llava-1.5-7b-hf")
    sd_model_dir: Path = Path("pre_model/DDIM/stable-diffusion-v1-5")


@dataclass
class RuntimeConfig:
    device: str = "auto"
    dtype: str = "float16"
    num_workers: int = 1
    fail_fast: bool = False
    log_interval: int = 1
    torch_compile: bool = False


@dataclass
class EvalConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    pipeline: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    project_root: Path = Path.cwd()
    source_config: Path | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EvalConfig":
        path = Path(path).expanduser().resolve()
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise TypeError("Eval YAML root must be a mapping.")

        project_root = infer_enar_root(path)
        experiment_data = _mapping(data.get("experiment", {}), "experiment")
        dataset_data = _mapping(data.get("dataset", {}), "dataset")
        models_data = _mapping(data.get("models", {}), "models")
        runtime_data = _mapping(data.get("runtime", {}), "runtime")

        filters = DatasetFilters(**_mapping(dataset_data.get("filters", {}), "dataset.filters"))
        dataset = DatasetConfig(
            name=str(dataset_data.get("name", "VLMBias")),
            root_dir=resolve_path(dataset_data.get("root_dir", "toy_dataset/VLMBias"), project_root),
            data_dir=resolve_path(dataset_data.get("data_dir", "toy_dataset/VLMBias/data"), project_root),
            split=str(dataset_data.get("split", "main")),
            data_files=dict(dataset_data.get("data_files") or DatasetConfig().data_files),
            filters=filters,
            image_export=dict(dataset_data.get("image_export") or DatasetConfig().image_export),
        )

        experiment = ExperimentConfig(
            name=str(experiment_data.get("name", "vlmbias_enar_eval")),
            run_name=str(experiment_data.get("run_name", "run_001")),
            seed=int(experiment_data.get("seed", 42)),
            output_dir=resolve_path(experiment_data.get("output_dir", "outputs/enar_eval/vlmbias"), project_root),
            resume=bool(experiment_data.get("resume", True)),
            overwrite=bool(experiment_data.get("overwrite", False)),
            save_intermediate=bool(experiment_data.get("save_intermediate", True)),
            dry_run=bool(experiment_data.get("dry_run", False)),
        )

        models = ModelsConfig(
            llava_model_dir=resolve_path(models_data.get("llava_model_dir", ModelsConfig.llava_model_dir), project_root),
            sd_model_dir=resolve_path(models_data.get("sd_model_dir", ModelsConfig.sd_model_dir), project_root),
        )
        runtime = RuntimeConfig(
            device=str(runtime_data.get("device", "auto")),
            dtype=str(runtime_data.get("dtype", "float16")),
            num_workers=int(runtime_data.get("num_workers", 1)),
            fail_fast=bool(runtime_data.get("fail_fast", False)),
            log_interval=int(runtime_data.get("log_interval", 1)),
            torch_compile=bool(runtime_data.get("torch_compile", False)),
        )
        return cls(
            experiment=experiment,
            dataset=dataset,
            models=models,
            runtime=runtime,
            pipeline=dict(data.get("pipeline") or {}),
            evaluation=dict(data.get("evaluation") or {}),
            project_root=project_root,
            source_config=path,
        )

    @property
    def run_dir(self) -> Path:
        return self.experiment.output_dir / self.experiment.run_name

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": {
                "name": self.experiment.name,
                "run_name": self.experiment.run_name,
                "seed": self.experiment.seed,
                "output_dir": str(self.experiment.output_dir),
                "resume": self.experiment.resume,
                "overwrite": self.experiment.overwrite,
                "save_intermediate": self.experiment.save_intermediate,
                "dry_run": self.experiment.dry_run,
            },
            "dataset": {
                "name": self.dataset.name,
                "root_dir": str(self.dataset.root_dir),
                "data_dir": str(self.dataset.data_dir),
                "split": self.dataset.split,
                "data_files": self.dataset.data_files,
                "filters": {
                    "max_samples": self.dataset.filters.max_samples,
                    "sample_ids": self.dataset.filters.sample_ids,
                    "topics": self.dataset.filters.topics,
                    "sub_topics": self.dataset.filters.sub_topics,
                    "type_of_question": self.dataset.filters.type_of_question,
                    "with_title": self.dataset.filters.with_title,
                    "pixel": self.dataset.filters.pixel,
                },
                "image_export": self.dataset.image_export,
            },
            "models": {
                "llava_model_dir": str(self.models.llava_model_dir),
                "sd_model_dir": str(self.models.sd_model_dir),
            },
            "runtime": {
                "device": self.runtime.device,
                "dtype": self.runtime.dtype,
                "num_workers": self.runtime.num_workers,
                "fail_fast": self.runtime.fail_fast,
                "log_interval": self.runtime.log_interval,
                "torch_compile": self.runtime.torch_compile,
            },
            "pipeline": self.pipeline,
            "evaluation": self.evaluation,
            "project_root": str(self.project_root),
        }

    def config_hash(self) -> str:
        data = self.to_dict()
        data["experiment"].pop("resume", None)
        data["experiment"].pop("overwrite", None)
        return stable_hash(data)

    def save_resolved_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, allow_unicode=True)


def infer_enar_root(config_path: Path) -> Path:
    resolved = config_path.expanduser().resolve()
    for parent in (resolved.parent, *resolved.parents):
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


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping.")
    return value
