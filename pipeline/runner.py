from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from Attend.pipeline import AttendPipeline, AttendResult
from Envision.pipeline import EnvisionPipeline, EnvisionResult
from Respond.pipeline import RespondPipeline, RespondResult

from .config import EnARPipelineConfig


@dataclass
class EnARPipelineResult:
    envision: EnvisionResult
    attend: AttendResult
    respond: RespondResult
    metadata_path: str


class EnARPipeline:
    def __init__(self, config: EnARPipelineConfig) -> None:
        self.config = config

    @classmethod
    def from_yaml(cls, config_yaml: str | Path) -> "EnARPipeline":
        return cls(EnARPipelineConfig.from_yaml(config_yaml))

    def run(self) -> EnARPipelineResult:
        started = time.time()
        run_dir = self.config.run_output_dir
        run_dir.mkdir(parents=True, exist_ok=True)

        envision_config = self.config.build_envision_config()
        attend_config = self.config.build_attend_config(envision_config)
        respond_config = self.config.build_respond_config(attend_config)
        self.config.save_resolved_yaml(
            run_dir / "resolved_pipeline_config.yaml",
            envision_config,
            attend_config,
            respond_config,
        )

        envision_result = EnvisionPipeline(envision_config).run()
        attend_result = AttendPipeline(attend_config).run()
        respond_result = RespondPipeline(respond_config).run()

        metadata = {
            "input_image": str(self.config.input_image),
            "question": self.config.question,
            "run_output_dir": str(run_dir),
            "envision": {
                "original_image_path": envision_result.original_image_path,
                "preprocessed_image_path": envision_result.preprocessed_image_path,
                "impression_image_path": envision_result.impression_image_path,
                "uncertainty_map_path": envision_result.uncertainty_map_path,
                "uncertainty_heatmap_path": envision_result.uncertainty_heatmap_path,
                "metadata_path": envision_result.metadata_path,
            },
            "attend": {
                "selected_patch_count": len(attend_result.selected_patch_indices),
                "selected_vision_token_count": len(attend_result.selected_vision_token_indices),
                "mask_origin_path": attend_result.mask_origin_path,
                "patch_overlay_path": attend_result.patch_overlay_path,
                "attend_result_json": attend_result.attend_result_json,
            },
            "respond": {
                "regular_answer": respond_result.regular_answer,
                "enar_answer": respond_result.enar_answer,
                "respond_result_json": respond_result.respond_result_json,
                "token_logits_trace_path": respond_result.token_logits_trace_path,
            },
            "elapsed_seconds": round(time.time() - started, 4),
        }
        metadata_path = run_dir / "pipeline_result.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return EnARPipelineResult(
            envision=envision_result,
            attend=attend_result,
            respond=respond_result,
            metadata_path=str(metadata_path),
        )
