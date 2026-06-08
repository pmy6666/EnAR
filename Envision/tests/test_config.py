from pathlib import Path

import pytest

from Envision.config import EnvisionConfig


def test_config_roundtrip(tmp_path: Path):
    image = tmp_path / "in.png"
    model = tmp_path / "model"
    out = tmp_path / "out"
    image.write_bytes(b"x")
    model.mkdir()
    cfg = EnvisionConfig(input_image=image, output_dir=out, sd_model_dir=model)
    cfg.validate()
    data = cfg.to_dict()
    assert data["input_image"].endswith("in.png")


def test_config_loads_grouped_yaml(tmp_path: Path):
    image = tmp_path / "in.png"
    model = tmp_path / "model"
    out = tmp_path / "out"
    image.write_bytes(b"x")
    model.mkdir()
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        f"""
paths:
  sd_model_dir: {model}
  input_image: {image}
  output_dir: {out}
image:
  image_size: 128
  preprocess_mode: pad
  pad_color: [127, 127, 127]
ddim:
  num_ddim_steps: 20
  inversion_step_T: 8
  guidance_scale: 1.0
langevin:
  langevin_steps_M: 2
  sample_count_K: 2
  eta_start: 0.01
  eta_end: 0.0001
  temperature_tau: 0.1
prompt:
  prompt: ""
  negative_prompt: ""
runtime:
  seed: 7
  dtype: float32
  device: null
  debug: false
""",
        encoding="utf-8",
    )
    cfg = EnvisionConfig.from_yaml(yaml_path)
    cfg.validate()
    assert cfg.image_size == 128
    assert cfg.preprocess_mode == "pad"
    assert cfg.pad_color == (127, 127, 127)
    assert cfg.seed == 7
    assert cfg.to_yaml_dict()["paths"]["input_image"].endswith("in.png")


def test_config_requires_input():
    with pytest.raises(ValueError):
        EnvisionConfig(output_dir=Path("out")).validate()
