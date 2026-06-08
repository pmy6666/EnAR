from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


YAML_SECTION_KEYS = {
    "paths",
    "image",
    "ddim",
    "langevin",
    "prompt",
    "runtime",
}

YAML_TO_FLAT_KEYS = {
    "paths": {"sd_model_dir", "input_image", "output_dir"},
    "image": {"image_size", "preprocess_mode", "pad_color"},
    "ddim": {"num_ddim_steps", "inversion_step_T", "guidance_scale"},
    "langevin": {"langevin_steps_M", "sample_count_K", "eta_start", "eta_end", "temperature_tau"},
    "prompt": {"prompt", "negative_prompt"},
    "runtime": {"seed", "dtype", "device", "debug"},
}

FLAT_TO_YAML_SECTION = {
    key: section
    for section, keys in YAML_TO_FLAT_KEYS.items()
    for key in keys
}


@dataclass
class EnvisionConfig:
    sd_model_dir: Path = Path("EnAR/pre_model/DDIM/stable-diffusion-v1-5")
    input_image: Optional[Path] = None
    output_dir: Optional[Path] = None
    image_size: int = 512
    preprocess_mode: str = "pad"
    pad_color: tuple[int, int, int] = (127, 127, 127)
    num_ddim_steps: int = 50
    inversion_step_T: int = 30
    langevin_steps_M: int = 10
    sample_count_K: int = 4
    eta_start: float = 1e-2
    eta_end: float = 1e-4
    temperature_tau: float = 0.1
    prompt: str = ""
    negative_prompt: str = ""
    seed: int = 42
    dtype: str = "float16"
    device: Optional[str] = None
    guidance_scale: float = 1.0
    debug: bool = False

    def __post_init__(self) -> None:
        self.sd_model_dir = Path(self.sd_model_dir)
        self.input_image = Path(self.input_image) if self.input_image is not None else None
        self.output_dir = Path(self.output_dir) if self.output_dir is not None else None
        self.pad_color = tuple(int(value) for value in self.pad_color)

    @classmethod
    def from_file(cls, path: str | Path) -> "EnvisionConfig":
        return cls.from_yaml(path)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EnvisionConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnvisionConfig":
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
                allowed_keys = YAML_TO_FLAT_KEYS[key]
                unknown = sorted(set(value) - allowed_keys)
                if unknown:
                    raise ValueError(f"Unknown keys in YAML section '{key}': {unknown}")
                flat.update(value)
            else:
                flat[key] = value

        unknown_flat_keys = sorted(set(flat) - set(FLAT_TO_YAML_SECTION))
        if unknown_flat_keys:
            raise ValueError(f"Unknown Envision config keys: {unknown_flat_keys}")
        return flat

    def validate(self) -> None:
        if self.input_image is None:
            raise ValueError("input_image is required.")
        if self.output_dir is None:
            raise ValueError("output_dir is required.")
        if not self.input_image.is_file():
            raise FileNotFoundError(f"input_image does not exist: {self.input_image}")
        if not self.sd_model_dir.is_dir():
            raise FileNotFoundError(f"sd_model_dir does not exist: {self.sd_model_dir}")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive.")
        if self.preprocess_mode not in {"pad", "center_crop"}:
            raise ValueError("preprocess_mode must be one of: pad, center_crop.")
        if len(self.pad_color) != 3:
            raise ValueError("pad_color must contain three RGB values.")
        if not all(0 <= int(value) <= 255 for value in self.pad_color):
            raise ValueError("pad_color values must be in [0, 255].")
        if self.num_ddim_steps <= 0:
            raise ValueError("num_ddim_steps must be positive.")
        if not 0 <= self.inversion_step_T <= self.num_ddim_steps:
            raise ValueError("inversion_step_T must be between 0 and num_ddim_steps.")
        if self.langevin_steps_M < 0:
            raise ValueError("langevin_steps_M must be non-negative.")
        if self.sample_count_K <= 0:
            raise ValueError("sample_count_K must be positive.")
        if self.dtype not in {"float16", "float32", "bfloat16"}:
            raise ValueError("dtype must be one of: float16, float32, bfloat16.")

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["pad_color"] = list(self.pad_color)
        for key in ("sd_model_dir", "input_image", "output_dir"):
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
            for section in ("paths", "image", "ddim", "langevin", "prompt", "runtime")
        }

    def save_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_yaml_dict(), f, sort_keys=False, allow_unicode=True)
