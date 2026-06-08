from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

from Attend.config import AttendConfig
from Envision.config import EnvisionConfig
from Respond.config import RespondConfig


TOP_LEVEL_KEYS = {"paths", "prompt", "runtime", "stages"}
PATH_KEYS = {"input_image", "output_dir", "sd_model_dir", "llava_model_dir"}


@dataclass
class EnARPipelineConfig:
    input_image: Path
    output_dir: Path
    question: str
    sd_model_dir: Path = Path("pre_model/DDIM/stable-diffusion-v1-5")
    llava_model_dir: Path = Path("pre_model/LLM/llava-1.5-7b-hf")
    run_name: str | None = None
    device: str | None = None
    dtype: str | None = None
    seed: int | None = None
    envision_prompt: str = ""
    negative_prompt: str = ""
    stages: Dict[str, Dict[str, Any]] | None = None
    project_root: Path = Path.cwd()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EnARPipelineConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise TypeError("Pipeline YAML root must be a mapping.")
        unknown = sorted(set(data) - TOP_LEVEL_KEYS)
        if unknown:
            raise ValueError(f"Unknown pipeline config sections: {unknown}")

        project_root = _infer_enar_root(path)
        paths = _require_mapping(data, "paths")
        prompt = _require_mapping(data, "prompt")
        runtime = _optional_mapping(data, "runtime")
        stages = _optional_mapping(data, "stages")

        missing_paths = sorted({"input_image", "output_dir"} - set(paths))
        if missing_paths:
            raise ValueError(f"Missing required paths keys: {missing_paths}")
        unknown_paths = sorted(set(paths) - PATH_KEYS)
        if unknown_paths:
            raise ValueError(f"Unknown paths keys: {unknown_paths}")

        if "question" not in prompt:
            raise ValueError("prompt.question is required.")

        config = cls(
            input_image=_resolve_path(paths["input_image"], project_root),
            output_dir=_resolve_path(paths["output_dir"], project_root),
            question=str(prompt["question"]),
            sd_model_dir=_resolve_path(paths.get("sd_model_dir", cls.sd_model_dir), project_root),
            llava_model_dir=_resolve_path(paths.get("llava_model_dir", cls.llava_model_dir), project_root),
            run_name=runtime.get("run_name"),
            device=runtime.get("device"),
            dtype=runtime.get("dtype"),
            seed=runtime.get("seed"),
            envision_prompt=str(prompt.get("envision_prompt", "")),
            negative_prompt=str(prompt.get("negative_prompt", "")),
            stages=dict(stages),
            project_root=project_root,
        )
        config.validate()
        return config

    @property
    def run_output_dir(self) -> Path:
        return self.output_dir / self.run_name if self.run_name else self.output_dir

    @property
    def envision_output_dir(self) -> Path:
        return self.run_output_dir / "envision"

    @property
    def attend_output_dir(self) -> Path:
        return self.run_output_dir / "attend"

    @property
    def respond_output_dir(self) -> Path:
        return self.run_output_dir / "respond"

    def validate(self) -> None:
        if not self.input_image.is_file():
            raise FileNotFoundError(f"paths.input_image does not exist: {self.input_image}")
        if not self.sd_model_dir.is_dir():
            raise FileNotFoundError(f"paths.sd_model_dir does not exist: {self.sd_model_dir}")
        if not self.llava_model_dir.is_dir():
            raise FileNotFoundError(f"paths.llava_model_dir does not exist: {self.llava_model_dir}")
        if not self.question.strip():
            raise ValueError("prompt.question is required.")
        unknown_stages = sorted(set(self.stages or {}) - {"envision", "attend", "respond"})
        if unknown_stages:
            raise ValueError(f"Unknown stages keys: {unknown_stages}")

    def build_envision_config(self) -> EnvisionConfig:
        data = _stage_data(self.stages, "envision")
        data = _merge_stage_data(
            data,
            {
                "paths": {
                    "sd_model_dir": str(self.sd_model_dir),
                    "input_image": str(self.input_image),
                    "output_dir": str(self.envision_output_dir),
                },
                "image": {
                    "preprocess_mode": "pad",
                    "pad_color": [127, 127, 127],
                },
                "prompt": {
                    "prompt": self.envision_prompt,
                    "negative_prompt": self.negative_prompt,
                },
            },
        )
        _apply_runtime_defaults(data, "runtime", _envision_device(self.device), self.dtype, self.seed)
        _normalize_envision_runtime_device(data)
        return EnvisionConfig.from_dict(data)

    def build_attend_config(self, envision_config: EnvisionConfig) -> AttendConfig:
        data = _stage_data(self.stages, "attend")
        data = _merge_stage_data(
            data,
            {
                "paths": {
                    "llava_model_dir": str(self.llava_model_dir),
                    "original_image": str(envision_config.output_dir / "preprocessed.png"),
                    "impression_image": str(envision_config.output_dir / "impression.png"),
                    "uncertainty_map": str(envision_config.output_dir / "uncertainty_map.npy"),
                    "envision_metadata": str(envision_config.output_dir / "metadata.json"),
                    "output_dir": str(self.attend_output_dir),
                }
            },
        )
        _apply_runtime_defaults(data, "model", self.device, self.dtype, None)
        config = AttendConfig.from_dict(data)
        config.resolve_paths(self.project_root)
        return config

    def build_respond_config(self, attend_config: AttendConfig) -> RespondConfig:
        data = _stage_data(self.stages, "respond")
        data = _merge_stage_data(
            data,
            {
                "paths": {
                    "llava_model_dir": str(self.llava_model_dir),
                    "image_path": str(self.envision_output_dir / "preprocessed.png"),
                    "attend_result_json": str(attend_config.output_dir / "attend_result.json"),
                    "output_dir": str(self.respond_output_dir),
                },
                "generation": {"question": self.question},
            },
        )
        _apply_runtime_defaults(data, "model", self.device, self.dtype, None)
        config = RespondConfig.from_dict(data)
        config.resolve_paths(self.project_root)
        return config

    def save_resolved_yaml(
        self,
        path: str | Path,
        envision: EnvisionConfig,
        attend: AttendConfig,
        respond: RespondConfig,
    ) -> None:
        output = {
            "paths": {
                "input_image": str(self.input_image),
                "output_dir": str(self.output_dir),
                "sd_model_dir": str(self.sd_model_dir),
                "llava_model_dir": str(self.llava_model_dir),
            },
            "prompt": {
                "question": self.question,
                "envision_prompt": self.envision_prompt,
                "negative_prompt": self.negative_prompt,
            },
            "runtime": {
                "run_name": self.run_name,
                "device": self.device,
                "dtype": self.dtype,
                "seed": self.seed,
            },
            "resolved_stage_configs": {
                "envision": envision.to_yaml_dict(),
                "attend": attend.to_yaml_dict(),
                "respond": respond.to_yaml_dict(),
            },
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(output, f, sort_keys=False, allow_unicode=True)


def _infer_enar_root(config_path: Path) -> Path:
    resolved = config_path.expanduser().resolve()
    for parent in (resolved.parent, *resolved.parents):
        if parent.name == "EnAR":
            return parent
    cwd = Path.cwd().resolve()
    return cwd if cwd.name == "EnAR" else cwd / "EnAR"


def _resolve_path(value: Any, project_root: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    if path.parts and path.parts[0] == project_root.name:
        return (project_root.parent / path).resolve()
    return (project_root / path).resolve()


def _require_mapping(data: Dict[str, Any], section: str) -> Dict[str, Any]:
    value = data.get(section)
    if not isinstance(value, dict):
        raise ValueError(f"{section} section is required and must be a mapping.")
    return value


def _optional_mapping(data: Dict[str, Any], section: str) -> Dict[str, Any]:
    value = data.get(section, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{section} section must be a mapping.")
    return value


def _stage_data(stages: Dict[str, Dict[str, Any]] | None, stage: str) -> Dict[str, Any]:
    value = (stages or {}).get(stage, {})
    if not isinstance(value, dict):
        raise TypeError(f"stages.{stage} must be a mapping.")
    return _deep_copy_mapping(value)


def _merge_stage_data(stage_data: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    merged = _deep_copy_mapping(stage_data)
    for section, values in defaults.items():
        current = merged.setdefault(section, {})
        if not isinstance(current, dict):
            raise TypeError(f"Stage section '{section}' must be a mapping.")
        for key, value in values.items():
            current[key] = value
    return merged


def _apply_runtime_defaults(
    data: Dict[str, Any],
    section: str,
    device: str | None,
    dtype: str | None,
    seed: int | None,
) -> None:
    if section in data and data[section] is not None and not isinstance(data[section], dict):
        raise TypeError(f"Stage section '{section}' must be a mapping.")
    target = data.setdefault(section, {})
    if device is not None:
        target.setdefault("device", device)
    if dtype is not None:
        target.setdefault("dtype", dtype)
    if seed is not None:
        target.setdefault("seed", seed)


def _envision_device(device: str | None) -> str | None:
    if device is None:
        return None
    if str(device).lower() == "auto":
        return None
    return device


def _normalize_envision_runtime_device(data: Dict[str, Any]) -> None:
    runtime = data.get("runtime")
    if not isinstance(runtime, dict):
        return
    if str(runtime.get("device")).lower() == "auto":
        runtime["device"] = None


def _deep_copy_mapping(data: Dict[str, Any]) -> Dict[str, Any]:
    copied: Dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            copied[key] = _deep_copy_mapping(value)
        else:
            copied[key] = value
    return copied
