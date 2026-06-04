from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


class AttendOutputWriter:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_array(self, name: str, array: np.ndarray) -> str:
        path = self.output_dir / name
        np.save(path, array)
        return str(path)

    def save_result_json(self, data: dict[str, Any], filename: str = "attend_result.json") -> str:
        path = self.output_dir / filename
        with path.open("w", encoding="utf-8") as f:
            json.dump(_to_jsonable(data), f, ensure_ascii=False, indent=2)
        return str(path)


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping) or hasattr(value, "items"):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "__dict__"):
        return {str(key): _to_jsonable(item) for key, item in vars(value).items()}
    return str(value)
