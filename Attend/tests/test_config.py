from pathlib import Path

import pytest

from Attend.config import AttendConfig
from Attend.yaml_loader import AttendYamlConfigLoader


def test_config_loads_grouped_yaml_and_resolves_paths(tmp_path: Path):
    root = tmp_path
    model = root / "EnAR/pre_model/LLM/llava-1.5-7b-hf"
    original = root / "inputs/original.png"
    impression = root / "inputs/impression.png"
    uncertainty = root / "inputs/uncertainty.npy"
    out = root / "outputs/attend/demo"
    model.mkdir(parents=True)
    original.parent.mkdir(parents=True)
    original.write_bytes(b"x")
    impression.write_bytes(b"x")
    uncertainty.write_bytes(b"x")
    yaml_path = root / "EnAR/Attend/attend_config.yaml"
    yaml_path.parent.mkdir(parents=True)
    yaml_path.write_text(
        """
paths:
  llava_model_dir: EnAR/pre_model/LLM/llava-1.5-7b-hf
  original_image: inputs/original.png
  impression_image: inputs/impression.png
  uncertainty_map: inputs/uncertainty.npy
  output_dir: outputs/attend/demo
model:
  vision_feature_select_strategy: full
  num_additional_image_tokens: 1
  device: auto
  dtype: float32
attention:
  vision_layer_number: 6
  attention_top_ratio: 0.1
  uncertainty_top_ratio: 0.05
  padding_ratio_limit: 0.1
  uncertainty_weight: 1.0
visualization:
  save_raw_arrays: true
  save_heatmaps: true
  save_patch_overlay: true
  save_mask_origin: true
  mask_origin_mode: binary
  mask_origin_alpha: 0.45
""",
        encoding="utf-8",
    )

    cfg = AttendYamlConfigLoader(project_root=root).load(yaml_path)
    assert cfg.original_image == original.resolve()
    assert cfg.output_dir == out.resolve()
    snapshot = AttendYamlConfigLoader(project_root=root).save_resolved_snapshot(cfg)
    assert snapshot.is_file()


def test_config_rejects_invalid_ratio(tmp_path: Path):
    original = tmp_path / "o.png"
    impression = tmp_path / "i.png"
    uncertainty = tmp_path / "u.npy"
    original.write_bytes(b"x")
    impression.write_bytes(b"x")
    uncertainty.write_bytes(b"x")
    cfg = AttendConfig(
        original_image=original,
        impression_image=impression,
        uncertainty_map=uncertainty,
        output_dir=tmp_path / "out",
        attention_top_ratio=0,
    )
    with pytest.raises(ValueError):
        cfg.validate(check_model_dir=False)
