from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


YAML_SECTION_KEYS = {"paths", "generation", "contrastive", "model"}
YAML_TO_FLAT_KEYS = {
    "paths": {"llava_model_dir", "image_path", "attend_result_json", "output_dir"},
    "generation": {"question", "max_new_tokens", "do_sample", "temperature", "top_p", "seed"},
    "contrastive": {"alpha", "use_apc", "apc_beta", "padding_strategy", "save_decode_trace"},
    "model": {"device", "dtype", "vision_feature_select_strategy", "num_additional_image_tokens"},
}
FLAT_TO_YAML_SECTION = {
    key: section
    for section, keys in YAML_TO_FLAT_KEYS.items()
    for key in keys
}
PATH_KEYS = {"llava_model_dir", "image_path", "attend_result_json", "output_dir"}


@dataclass
class RespondConfig:
    llava_model_dir: Path = Path("EnAR/pre_model/LLM/llava-1.5-7b-hf")
    image_path: Optional[Path] = None
    question: str = ""
    attend_result_json: Optional[Path] = None
    output_dir: Optional[Path] = None
    alpha: float = 1.0
    max_new_tokens: int = 64
    do_sample: bool = True
    temperature: float = 1.0
    top_p: float = 1.0
    seed: int | None = 42
    use_apc: bool = True
    apc_beta: float = 0.1
    padding_strategy: str = "pad_token_embedding"
    save_decode_trace: bool = True
    device: str = "auto"
    dtype: str = "float16"
    vision_feature_select_strategy: str = "default"
    num_additional_image_tokens: int = 0

    def __post_init__(self) -> None:
        for key in PATH_KEYS:
            value = getattr(self, key)
            if value is not None:
                setattr(self, key, Path(value))

    @classmethod
    def from_yaml(cls, path: str | Path, project_root: str | Path | None = None) -> "RespondConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        config = cls.from_dict(data)
        config.resolve_paths(project_root or _infer_project_root(path))
        return config

    @classmethod
    def from_file(cls, path: str | Path, project_root: str | Path | None = None) -> "RespondConfig":
        return cls.from_yaml(path, project_root)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RespondConfig":
        flat = cls._flatten_yaml_dict(data)
        return cls(**flat)

    @staticmethod
    def _flatten_yaml_dict(data: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(data, dict):
            raise TypeError("YAML config root must be a mapping.")
        flat: Dict[str, Any] = {}
        for key, value in data.items():
            if key in YAML_SECTION_KEYS:
                if value is None:
                    continue
                if not isinstance(value, dict):
                    raise TypeError(f"YAML section '{key}' must be a mapping.")
                unknown = sorted(set(value) - YAML_TO_FLAT_KEYS[key])
                if unknown:
                    raise ValueError(f"Unknown keys in YAML section '{key}': {unknown}")
                flat.update(value)
            else:
                flat[key] = value
        unknown = sorted(set(flat) - set(FLAT_TO_YAML_SECTION))
        if unknown:
            raise ValueError(f"Unknown Respond config keys: {unknown}")
        return flat

    def resolve_paths(self, project_root: str | Path) -> None:
        root = Path(project_root).expanduser().resolve()
        for key in PATH_KEYS:
            value = getattr(self, key)
            if value is None:
                continue
            path = value.expanduser()
            if not path.is_absolute():
                path = root / path
            setattr(self, key, path.resolve())

    def validate(self, check_model_dir: bool = True) -> None:
        for key in ("image_path", "attend_result_json", "output_dir"):
            if getattr(self, key) is None:
                raise ValueError(f"{key} is required.")
        if not self.question.strip():
            raise ValueError("question is required.")
        if not self.image_path.is_file():
            raise FileNotFoundError(f"image_path does not exist: {self.image_path}")
        if not self.attend_result_json.is_file():
            raise FileNotFoundError(f"attend_result_json does not exist: {self.attend_result_json}")
        if check_model_dir and not self.llava_model_dir.is_dir():
            raise FileNotFoundError(f"llava_model_dir does not exist: {self.llava_model_dir}")
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative.")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1.")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive.")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1].")
        if not 0 <= self.apc_beta <= 1:
            raise ValueError("apc_beta must be in [0, 1].")
        if self.seed is not None and int(self.seed) < 0:
            raise ValueError("seed must be non-negative when provided.")
        valid_padding = {
            "pad_token_embedding",
            "zero_embedding",
            "mean_visual_embedding",
            "matched_mean_visual_embedding",
        }
        if self.padding_strategy not in valid_padding:
            raise ValueError(
                "padding_strategy must be one of: "
                + ", ".join(sorted(valid_padding))
            )
        if self.dtype not in {"float16", "float32", "bfloat16"}:
            raise ValueError("dtype must be one of: float16, float32, bfloat16.")
        if self.vision_feature_select_strategy not in {"default", "full"}:
            raise ValueError("vision_feature_select_strategy must be 'default' or 'full'.")

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        for key in PATH_KEYS:
            if data[key] is not None:
                data[key] = str(data[key])
        return data

    def to_yaml_dict(self) -> Dict[str, Dict[str, Any]]:
        flat = self.to_dict()
        grouped: Dict[str, Dict[str, Any]] = {section: {} for section in YAML_SECTION_KEYS}
        for key, value in flat.items():
            grouped[FLAT_TO_YAML_SECTION[key]][key] = value
        return {section: grouped[section] for section in ("paths", "generation", "contrastive", "model")}

    def save_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_yaml_dict(), f, sort_keys=False, allow_unicode=True)


def _infer_project_root(config_path: Path) -> Path:
    resolved = config_path.expanduser().resolve()
    for parent in (resolved.parent, *resolved.parents):
        if parent.name == "EnAR":
            return parent.parent
    return Path.cwd().resolve()
