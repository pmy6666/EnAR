from __future__ import annotations

from pathlib import Path

from .config import AttendConfig


class AttendYamlConfigLoader:
    def __init__(self, project_root: str | Path | None = None) -> None:
        self.project_root = Path(project_root).resolve() if project_root is not None else None

    def load(self, config_yaml: str | Path, check_model_dir: bool = True) -> AttendConfig:
        config = AttendConfig.from_yaml(config_yaml, self.project_root)
        config.validate(check_model_dir=check_model_dir)
        return config

    def save_resolved_snapshot(self, config: AttendConfig) -> Path:
        if config.output_dir is None:
            raise ValueError("output_dir is required before saving resolved config.")
        path = config.output_dir / "resolved_config.yaml"
        config.save_yaml(path)
        return path
