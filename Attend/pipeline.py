from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .attention_extractor import VisionAttentionExtractor
from .config import AttendConfig
from .contrastive import ContrastiveAttentionComputer
from .mask_mapper import MaskOriginMapper, SOURCE_LABEL_ENCODING
from .model_loader import LlavaVisionLoader, read_vision_config
from .output_writer import AttendOutputWriter
from .preprocessor import LlavaImagePreprocessor
from .token_selector import CounterfactualTokenSelector
from .uncertainty_mapper import UncertaintyPatchMapper
from .visualizer import AttendVisualizer
from .yaml_loader import AttendYamlConfigLoader


@dataclass
class AttendResult:
    selected_patch_indices: list[int]
    selected_vision_token_indices: list[int]
    mask_origin_path: str
    patch_overlay_path: str | None
    attend_result_json: str


class AttendPipeline:
    def __init__(self, config: AttendConfig) -> None:
        self.config = config

    @classmethod
    def from_yaml(cls, config_yaml: str | Path, project_root: str | Path | None = None) -> "AttendPipeline":
        loader = AttendYamlConfigLoader(project_root)
        return cls(loader.load(config_yaml))

    def run(self) -> AttendResult:
        self.config.validate()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.save_yaml(self.config.output_dir / "resolved_config.yaml")

        vision_meta = read_vision_config(self.config.llava_model_dir)
        loader = LlavaVisionLoader(
            self.config.llava_model_dir,
            self.config.vision_feature_select_strategy,
            self.config.num_additional_image_tokens,
            self.config.device,
            self.config.dtype,
        )
        components = loader.load()
        if self.config.vision_layer_number > int(vision_meta.get("num_hidden_layers", 24)):
            raise ValueError(
                f"vision_layer_number={self.config.vision_layer_number} exceeds "
                f"vision encoder layers={vision_meta.get('num_hidden_layers')}."
            )

        prep = LlavaImagePreprocessor(
            components.processor,
            image_size=components.image_size,
        ).run(self.config.original_image, self.config.impression_image)
        envision_metadata = _load_json_safely(self.config.envision_metadata) if self.config.envision_metadata else None

        extractor = VisionAttentionExtractor(components.vision_tower, components.device)
        original_attention = extractor.extract(
            prep.pixel_values_original.to(dtype=components.dtype),
            self.config.vision_layer_number,
        )
        impression_attention = extractor.extract(
            prep.pixel_values_impression.to(dtype=components.dtype),
            self.config.vision_layer_number,
        )
        # print("*" * 15)
        # print(f"impression attention shape: {impression_attention.attention_scores.shape}")
        # print("*" * 15)
        contrastive = ContrastiveAttentionComputer().compute(
            original_attention.attention_scores,
            impression_attention.attention_scores,
        )
        uncertainty = UncertaintyPatchMapper(
            image_size=components.image_size,
            patch_size=components.patch_size,
        ).map_file_with_geometry(
            self.config.uncertainty_map,
            envision_metadata,
            prep.preprocess_meta,
        )
        selection = CounterfactualTokenSelector().select(
            contrastive.delta_scores,
            uncertainty.patch_scores,
            self.config.attention_top_ratio,
            self.config.uncertainty_top_ratio,
            self.config.padding_ratio_limit,
            self.config.uncertainty_weight,
            original_attention.token_layout_meta.get("has_cls_token", True),
        )

        writer = AttendOutputWriter(self.config.output_dir)
        array_paths = {}
        if self.config.save_raw_arrays:
            array_paths["contrastive_attention"] = writer.save_array(
                "contrastive_attention.npy",
                contrastive.delta_scores,
            )
            array_paths["uncertainty_patch_scores"] = writer.save_array(
                "uncertainty_patch_scores.npy",
                uncertainty.patch_scores,
            )
            array_paths["source_label_grid"] = writer.save_array(
                "source_label_grid.npy",
                selection.source_label_grid,
            )

        visualizer = AttendVisualizer(
            self.config.output_dir,
            image_size=components.image_size,
            patch_size=components.patch_size,
        )
        image_paths = {}
        if self.config.save_heatmaps:
            image_paths["original_attention_heatmap"] = visualizer.save_heatmap(
                original_attention.attention_scores.reshape(components.patch_grid),
                "original_attention_heatmap.png",
            )
            image_paths["counterfactual_attention_heatmap"] = visualizer.save_heatmap(
                impression_attention.attention_scores.reshape(components.patch_grid),
                "counterfactual_attention_heatmap.png",
            )
            image_paths["contrastive_attention_heatmap"] = visualizer.save_heatmap(
                contrastive.delta_grid,
                "contrastive_attention_heatmap.png",
            )
            image_paths["uncertainty_patch_heatmap"] = visualizer.save_heatmap(
                uncertainty.patch_grid,
                "uncertainty_patch_heatmap.png",
            )
            image_paths["fused_score_heatmap"] = visualizer.save_heatmap(
                selection.score.reshape(components.patch_grid),
                "fused_score_heatmap.png",
            )
        image_paths["selected_patch_mask"] = visualizer.save_selected_patch_mask(
            selection.union_patch_mask_grid
        )
        patch_overlay_path = None
        if self.config.save_patch_overlay:
            patch_overlay_path = visualizer.save_patch_overlay(
                self.config.original_image,
                selection.union_patch_mask_grid,
            )
            image_paths["patch_overlay"] = patch_overlay_path

        source_label_paths = {}
        if self.config.save_source_masks:
            source_label_paths["patch_source_mask"] = visualizer.save_source_label_patch_mask(
                selection.source_label_grid
            )
            source_label_paths["patch_source_overlay"] = visualizer.save_source_label_patch_overlay(
                self.config.original_image,
                selection.source_label_grid,
            )

        mask_mapper = MaskOriginMapper(
            patch_size=components.patch_size,
            vision_input_size=(components.image_size, components.image_size),
            alpha=self.config.mask_origin_alpha,
        )
        mask_result = mask_mapper.map_and_save(
            selection.union_patch_mask_grid,
            self.config.original_image,
            prep.preprocess_meta,
            self.config.output_dir,
            save_overlay=True,
        )
        image_paths["mask_origin"] = mask_result.mask_origin_path
        if mask_result.mask_origin_overlay_path:
            image_paths["mask_origin_overlay"] = mask_result.mask_origin_overlay_path

        label_mask_result = None
        if self.config.save_source_masks:
            label_mask_result = mask_mapper.map_label_and_save(
                selection.source_label_grid,
                self.config.original_image,
                prep.preprocess_meta,
                self.config.output_dir,
                save_overlay=True,
            )
            source_label_paths["mask_origin_label"] = label_mask_result.label_mask_origin_path
            source_label_paths["mask_origin_color"] = label_mask_result.label_mask_origin_color_path
            if label_mask_result.label_mask_origin_overlay_path:
                source_label_paths[
                    "mask_origin_three_color_overlay"
                ] = label_mask_result.label_mask_origin_overlay_path

        result_data = {
            "original_image": str(self.config.original_image),
            "impression_image": str(self.config.impression_image),
            "uncertainty_map": str(self.config.uncertainty_map),
            "envision_metadata": str(self.config.envision_metadata) if self.config.envision_metadata else None,
            "vision_layer_number": self.config.vision_layer_number,
            "patch_grid": list(components.patch_grid),
            "patch_size": components.patch_size,
            "has_cls_token": original_attention.token_layout_meta.get("has_cls_token", True),
            "h_attn_patch_indices": selection.h_attn,
            "h_unc_patch_indices": selection.h_unc,
            "h_union_raw_patch_indices": selection.h_union_raw,
            "selected_patch_indices": selection.h_final,
            "selected_vision_token_indices": selection.vision_token_indices,
            "source_label_encoding": {str(k): v for k, v in SOURCE_LABEL_ENCODING.items()},
            "source_counts": selection.source_counts,
            "source_label_paths": source_label_paths,
            "mask_origin_path": mask_result.mask_origin_path,
            "mask_origin_overlay_path": mask_result.mask_origin_overlay_path,
            "mask_origin_mapping_meta": mask_result.meta,
            "mask_origin_label_mapping_meta": label_mask_result.meta if label_mask_result else None,
            "attention_top_ratio": self.config.attention_top_ratio,
            "uncertainty_top_ratio": self.config.uncertainty_top_ratio,
            "padding_ratio_limit": self.config.padding_ratio_limit,
            "uncertainty_weight": self.config.uncertainty_weight,
            "score_stats": {
                "original_attention": _array_stats(original_attention.attention_scores),
                "impression_attention": _array_stats(impression_attention.attention_scores),
                "delta_attention": _array_stats(contrastive.delta_scores),
                "uncertainty_patch": _array_stats(uncertainty.patch_scores),
                "combined_selection_score": _array_stats(selection.score),
            },
            "token_layout_meta": original_attention.token_layout_meta,
            "impression_token_layout_meta": impression_attention.token_layout_meta,
            "uncertainty_mapping_meta": uncertainty.meta,
            "preprocess_meta": prep.preprocess_meta,
            "array_paths": array_paths,
            "image_paths": image_paths,
        }
        if self.config.envision_metadata:
            result_data["envision_metadata_content"] = envision_metadata
        result_json = writer.save_result_json(result_data)
        return AttendResult(
            selected_patch_indices=selection.h_final,
            selected_vision_token_indices=selection.vision_token_indices,
            mask_origin_path=mask_result.mask_origin_path,
            patch_overlay_path=patch_overlay_path,
            attend_result_json=result_json,
        )


def _load_json_safely(path: Path):
    try:
        with Path(path).open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        return {"load_error": str(exc)}


def _array_stats(values) -> dict[str, float]:
    import numpy as np

    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }
