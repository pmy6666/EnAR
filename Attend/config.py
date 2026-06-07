from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


YAML_SECTION_KEYS = {"paths", "model", "attention", "visualization"}

YAML_TO_FLAT_KEYS = {
    "paths": {
        "llava_model_dir",
        "original_image",
        "impression_image",
        "uncertainty_map",
        "envision_metadata",
        "output_dir",
    },
    "model": {
        "vision_feature_select_strategy",
        "num_additional_image_tokens",
        "device",
        "dtype",
    },
    "attention": {
        "vision_layer_number",
        "attention_top_ratio",
        "uncertainty_top_ratio",
        "padding_ratio_limit",
        "uncertainty_weight",
    },
    "visualization": {
        "save_raw_arrays",
        "save_heatmaps",
        "save_patch_overlay",
        "save_mask_origin",
        "save_source_masks",
        "mask_origin_mode",
        "mask_origin_alpha",
    },
}

FLAT_TO_YAML_SECTION = {
    key: section
    for section, keys in YAML_TO_FLAT_KEYS.items()
    for key in keys
}

PATH_KEYS = {
    "llava_model_dir",
    "original_image",
    "impression_image",
    "uncertainty_map",
    "envision_metadata",
    "output_dir",
}


@dataclass
class AttendConfig:
    llava_model_dir: Path = Path("EnAR/pre_model/LLM/llava-1.5-7b-hf")
    original_image: Optional[Path] = None
    impression_image: Optional[Path] = None
    uncertainty_map: Optional[Path] = None
    envision_metadata: Optional[Path] = None
    output_dir: Optional[Path] = None
    vision_feature_select_strategy: str = "full"
    num_additional_image_tokens: int = 1
    device: str = "auto"
    dtype: str = "float16"
    vision_layer_number: int = 6
    attention_top_ratio: float = 0.10
    uncertainty_top_ratio: float = 0.05
    padding_ratio_limit: float = 0.10
    uncertainty_weight: float = 1.0
    save_raw_arrays: bool = True
    save_heatmaps: bool = True
    save_patch_overlay: bool = True
    save_mask_origin: bool = True
    save_source_masks: bool = True
    mask_origin_mode: str = "binary"
    mask_origin_alpha: float = 0.45

    def __post_init__(self) -> None:
        for key in PATH_KEYS:
            value = getattr(self, key)
            if value is not None:
                setattr(self, key, Path(value))

    @classmethod
    def from_file(cls, path: str | Path, project_root: str | Path | None = None) -> "AttendConfig":
        return cls.from_yaml(path, project_root)

    @classmethod
    def from_yaml(cls, path: str | Path, project_root: str | Path | None = None) -> "AttendConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        config = cls.from_dict(data)
        root = Path(project_root) if project_root is not None else _infer_project_root(path)
        config.resolve_paths(root)
        return config

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AttendConfig":
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
        unknown_flat_keys = sorted(set(flat) - set(FLAT_TO_YAML_SECTION))
        if unknown_flat_keys:
            raise ValueError(f"Unknown Attend config keys: {unknown_flat_keys}")
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
        required = ("original_image", "impression_image", "uncertainty_map", "output_dir")
        for key in required:
            if getattr(self, key) is None:
                raise ValueError(f"{key} is required.")
        for key in ("original_image", "impression_image", "uncertainty_map"):
            path = getattr(self, key)
            if not path.is_file():
                raise FileNotFoundError(f"{key} does not exist: {path}")
        if self.envision_metadata is not None and not self.envision_metadata.is_file():
            raise FileNotFoundError(f"envision_metadata does not exist: {self.envision_metadata}")
        if check_model_dir and not self.llava_model_dir.is_dir():
            raise FileNotFoundError(f"llava_model_dir does not exist: {self.llava_model_dir}")
        if self.vision_layer_number < 1:
            raise ValueError("vision_layer_number must be >= 1.")
        for key in ("attention_top_ratio", "uncertainty_top_ratio", "padding_ratio_limit"):
            value = getattr(self, key)
            if not 0 < value <= 1:
                raise ValueError(f"{key} must be in (0, 1].")
        if self.uncertainty_weight < 0:
            raise ValueError("uncertainty_weight must be non-negative.")
        if self.num_additional_image_tokens < 0:
            raise ValueError("num_additional_image_tokens must be non-negative.")
        if self.vision_feature_select_strategy not in {"full", "default"}:
            raise ValueError("vision_feature_select_strategy must be 'full' or 'default'.")
        if self.dtype not in {"float16", "float32", "bfloat16"}:
            raise ValueError("dtype must be one of: float16, float32, bfloat16.")
        if self.mask_origin_mode not in {"binary", "overlay"}:
            raise ValueError("mask_origin_mode must be 'binary' or 'overlay'.")
        if not 0 <= self.mask_origin_alpha <= 1:
            raise ValueError("mask_origin_alpha must be in [0, 1].")

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
        return {
            section: grouped[section]
            for section in ("paths", "model", "attention", "visualization")
        }

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
