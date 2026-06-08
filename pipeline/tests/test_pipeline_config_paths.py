from pathlib import Path

from pipeline.config import EnARPipelineConfig


def test_pipeline_wires_processed_image_to_attend_and_respond(tmp_path: Path):
    root = tmp_path / "EnAR"
    image = root / "inputs/image.png"
    sd_model = root / "pre_model/DDIM/stable-diffusion-v1-5"
    llava_model = root / "pre_model/LLM/llava-1.5-7b-hf"
    image.parent.mkdir(parents=True)
    sd_model.mkdir(parents=True)
    llava_model.mkdir(parents=True)
    image.write_bytes(b"x")
    config = EnARPipelineConfig(
        input_image=image,
        output_dir=root / "outputs/pipeline",
        question="What is shown?",
        sd_model_dir=sd_model,
        llava_model_dir=llava_model,
        run_name="run_001",
        project_root=root,
    )

    envision = config.build_envision_config()
    attend = config.build_attend_config(envision)
    respond = config.build_respond_config(attend)

    assert attend.original_image == envision.output_dir / "preprocessed.png"
    assert respond.image_path == config.envision_output_dir / "preprocessed.png"
