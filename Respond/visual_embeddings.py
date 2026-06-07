from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class VisualEmbeddingResult:
    visual_embeddings: Any
    visual_token_layout: dict[str, Any]


class VisualEmbeddingExtractor:
    def __init__(
        self,
        model: Any,
        vision_feature_select_strategy: str = "default",
        vision_feature_layer: int | list[int] | None = None,
    ) -> None:
        self.model = model
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer

    def extract(self, pixel_values: Any, attend_result_json: str | Path | dict[str, Any] | None = None) -> VisualEmbeddingResult:
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("torch is required to extract visual embeddings.") from exc

        with torch.inference_mode():
            image_features = self._get_image_features(pixel_values)
        if isinstance(image_features, (list, tuple)):
            image_features = torch.stack(list(image_features), dim=0)
        layout = build_visual_token_layout(image_features, attend_result_json)
        return VisualEmbeddingResult(image_features, layout)

    def _get_image_features(self, pixel_values: Any) -> Any:
        get_image_features = getattr(self.model, "get_image_features", None)
        if get_image_features is not None:
            out = get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=self.vision_feature_layer,
                vision_feature_select_strategy=self.vision_feature_select_strategy,
                return_dict=True,
            )
            return getattr(out, "pooler_output", out)
        base = getattr(self.model, "model", self.model)
        get_image_features = getattr(base, "get_image_features", None)
        if get_image_features is None:
            raise AttributeError("LLaVA model does not expose get_image_features.")
        out = get_image_features(
            pixel_values=pixel_values,
            vision_feature_layer=self.vision_feature_layer,
            vision_feature_select_strategy=self.vision_feature_select_strategy,
            return_dict=True,
        )
        return getattr(out, "pooler_output", out)


def build_visual_token_layout(visual_embeddings: Any, attend_result_json: str | Path | dict[str, Any] | None) -> dict[str, Any]:
    attend = load_attend_result(attend_result_json) if attend_result_json is not None else {}
    token_count = int(visual_embeddings.shape[-2])
    selected_patch_indices = [int(x) for x in attend.get("selected_patch_indices", [])]
    has_cls_token = bool(attend.get("has_cls_token", False))
    num_patches = _num_patches_from_attend(attend)
    if num_patches is not None and token_count == num_patches:
        selected_vision_token_indices = selected_patch_indices
        has_cls_token = False
    else:
        selected_vision_token_indices = attend.get("selected_vision_token_indices")
        if selected_vision_token_indices is None:
            offset = 1 if has_cls_token else 0
            selected_vision_token_indices = [idx + offset for idx in selected_patch_indices]
    selected_vision_token_indices = [
        int(idx) for idx in selected_vision_token_indices if 0 <= int(idx) < token_count
    ]
    return {
        "token_count": token_count,
        "hidden_size": int(visual_embeddings.shape[-1]),
        "selected_patch_indices": selected_patch_indices,
        "selected_vision_token_indices": selected_vision_token_indices,
        "has_cls_token": has_cls_token,
        "patch_grid": attend.get("patch_grid"),
        "patch_size": attend.get("patch_size"),
    }


def load_attend_result(path_or_data: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(path_or_data, dict):
        return path_or_data
    with Path(path_or_data).open("r", encoding="utf-8") as f:
        return json.load(f)


def _num_patches_from_attend(attend: dict[str, Any]) -> int | None:
    patch_grid = attend.get("patch_grid")
    if isinstance(patch_grid, (list, tuple)) and len(patch_grid) == 2:
        return int(patch_grid[0]) * int(patch_grid[1])
    selected_patch_indices = attend.get("selected_patch_indices") or []
    if selected_patch_indices:
        return max(int(idx) for idx in selected_patch_indices) + 1
    return None
