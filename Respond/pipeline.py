from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import RespondConfig
from .generation_loop import ContrastiveGenerationLoop
from .input_encoder import MultimodalInputEncoder
from .model_loader import LlavaGenerationLoader
from .output_writer import RespondOutputWriter
from .padded_visual_builder import PaddedVisualInputBuilder
from .prompt_builder import LlavaPromptBuilder
from .regular_generation import RegularGenerationRunner
from .visual_embeddings import VisualEmbeddingExtractor, load_attend_result


@dataclass
class RespondResult:
    regular_answer: str
    enar_answer: str
    respond_result_json: str
    token_logits_trace_path: str | None = None


class RespondPipeline:
    def __init__(self, config: RespondConfig) -> None:
        self.config = config

    @classmethod
    def from_yaml(cls, config_yaml: str | Path, project_root: str | Path | None = None) -> "RespondPipeline":
        return cls(RespondConfig.from_yaml(config_yaml, project_root))

    def run(self) -> RespondResult:
        self.config.validate()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.save_yaml(self.config.output_dir / "resolved_config.yaml")

        attend = load_attend_result(self.config.attend_result_json)
        prompt = LlavaPromptBuilder().build(self.config.question)
        components = LlavaGenerationLoader(
            self.config.llava_model_dir,
            device=self.config.device,
            dtype=self.config.dtype,
            vision_feature_select_strategy=self.config.vision_feature_select_strategy,
            num_additional_image_tokens=self.config.num_additional_image_tokens,
        ).load()
        image_token_index = int(getattr(components.model.config, "image_token_index"))
        encoded = MultimodalInputEncoder(components.processor, components.device).encode(
            self.config.image_path,
            prompt,
            image_token_index=image_token_index,
        )

        regular_answer = RegularGenerationRunner(
            components.model,
            components.processor,
            components.tokenizer,
        ).run(
            encoded,
            self.config.max_new_tokens,
            self.config.do_sample,
            self.config.temperature,
            self.config.top_p,
            self.config.seed,
        )

        visual_result = VisualEmbeddingExtractor(
            components.model,
            vision_feature_select_strategy=self.config.vision_feature_select_strategy,
            vision_feature_layer=components.generation_meta.get("vision_feature_layer", -2),
        ).extract(encoded.pixel_values, attend)
        # print("*" * 20)
        # print(f"visual_result.shape = {visual_result.visual_embeddings.shape}")
        # print("*" * 20)
        padded_result = PaddedVisualInputBuilder(
            components.model,
            components.tokenizer,
            self.config.padding_strategy,
        ).build(
            visual_result.visual_embeddings,
            visual_result.visual_token_layout["selected_vision_token_indices"],
        )
        selected_sequence_positions = [
            encoded.image_token_positions[idx]
            for idx in padded_result.padding_meta["selected_vision_token_indices"]
            if 0 <= idx < len(encoded.image_token_positions)
        ]
        generation = ContrastiveGenerationLoop(
            components.model,
            components.tokenizer,
            image_token_index=image_token_index,
            alpha=self.config.alpha,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=self.config.do_sample,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            use_apc=self.config.use_apc,
            apc_beta=self.config.apc_beta,
            seed=self.config.seed,
        ).run(
            encoded.input_ids,
            encoded.attention_mask,
            visual_result.visual_embeddings,
            padded_result.visual_embeddings_padded,
        )

        writer = RespondOutputWriter(self.config.output_dir)
        regular_path = writer.save_text("answer_regular.txt", regular_answer)
        enar_path = writer.save_text("answer_enar.txt", generation.decoded_text)
        result_data = {
            "image_path": str(self.config.image_path),
            "question": self.config.question,
            "attend_result_json": str(self.config.attend_result_json),
            "selected_patch_indices": attend.get("selected_patch_indices", []),
            "selected_vision_token_indices": visual_result.visual_token_layout["selected_vision_token_indices"],
            "alpha": self.config.alpha,
            "decode_mode": "vcd_sampling" if self.config.do_sample else "greedy_debug",
            "do_sample": self.config.do_sample,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "seed": self.config.seed,
            "padding_strategy": padded_result.padding_meta["actual_strategy"],
            "padding_meta": padded_result.padding_meta,
            "selected_sequence_positions": selected_sequence_positions,
            "regular_answer": regular_answer,
            "enar_answer": generation.decoded_text,
            "answer_regular_path": regular_path,
            "answer_enar_path": enar_path,
            "max_new_tokens": self.config.max_new_tokens,
            "use_apc": self.config.use_apc,
            "apc_beta": self.config.apc_beta,
            "prompt": prompt,
            "prompt_len": encoded.prompt_len,
            "input_meta": encoded.meta,
            "visual_token_layout": visual_result.visual_token_layout,
            "generation_meta": components.generation_meta,
        }
        result_json = writer.save_result(
            result_data,
            generation.decode_trace,
            self.config.save_decode_trace,
            generation.token_logits_trace,
        )
        return RespondResult(
            regular_answer,
            generation.decoded_text,
            result_json,
            result_data.get("token_logits_trace_path"),
        )


def _load_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
