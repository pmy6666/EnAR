from pathlib import Path

import pytest

from Respond.config import RespondConfig
from Respond.prompt_builder import LlavaPromptBuilder


def test_config_loads_grouped_yaml_and_resolves_paths(tmp_path: Path):
    root = tmp_path
    model = root / "EnAR/pre_model/LLM/llava-1.5-7b-hf"
    image = root / "inputs/original.png"
    attend = root / "outputs/attend/run_001/attend_result.json"
    output = root / "outputs/respond/run_001"
    model.mkdir(parents=True)
    image.parent.mkdir(parents=True)
    attend.parent.mkdir(parents=True)
    image.write_bytes(b"x")
    attend.write_text('{"selected_patch_indices": [0]}', encoding="utf-8")
    yaml_path = root / "EnAR/Respond/respond_config.yaml"
    yaml_path.parent.mkdir(parents=True)
    yaml_path.write_text(
        """
paths:
  llava_model_dir: EnAR/pre_model/LLM/llava-1.5-7b-hf
  image_path: inputs/original.png
  attend_result_json: outputs/attend/run_001/attend_result.json
  output_dir: outputs/respond/run_001
generation:
  question: What is this?
  max_new_tokens: 8
  do_sample: false
  temperature: 1.0
  top_p: 1.0
contrastive:
  alpha: 0.5
  use_apc: false
  apc_beta: 0.1
  padding_strategy: zero_embedding
  save_decode_trace: true
model:
  device: cpu
  dtype: float32
  vision_feature_select_strategy: default
  num_additional_image_tokens: 0
""",
        encoding="utf-8",
    )

    cfg = RespondConfig.from_yaml(yaml_path, project_root=root)
    assert cfg.image_path == image.resolve()
    assert cfg.output_dir == output.resolve()
    cfg.validate(check_model_dir=True)


def test_config_rejects_invalid_alpha(tmp_path: Path):
    image = tmp_path / "image.png"
    attend = tmp_path / "attend.json"
    image.write_bytes(b"x")
    attend.write_text("{}", encoding="utf-8")
    cfg = RespondConfig(
        image_path=image,
        attend_result_json=attend,
        output_dir=tmp_path / "out",
        question="Q?",
        alpha=-0.1,
    )
    with pytest.raises(ValueError):
        cfg.validate(check_model_dir=False)


def test_prompt_builder_uses_llava_1_5_format():
    prompt = LlavaPromptBuilder().build("Describe it.")
    assert prompt == "USER: <image>\nDescribe it.\nASSISTANT:"
