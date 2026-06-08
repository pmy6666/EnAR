from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_PARENT = PROJECT_ROOT.parent
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from EnAR.Attend.config import AttendConfig
from EnAR.Attend.pipeline import AttendPipeline
from EnAR.Envision.config import EnvisionConfig
from EnAR.Envision.pipeline import EnvisionPipeline
from EnAR.Respond.config import RespondConfig
from EnAR.Respond.generation_loop import ContrastiveGenerationLoop
from EnAR.Respond.input_encoder import MultimodalInputEncoder
from EnAR.Respond.model_loader import LlavaGenerationLoader
from EnAR.Respond.padded_visual_builder import PaddedVisualInputBuilder
from EnAR.Respond.pipeline import RespondPipeline
from EnAR.Respond.prompt_builder import LlavaPromptBuilder
from EnAR.Respond.regular_generation import RegularGenerationRunner
from EnAR.Respond.visual_embeddings import VisualEmbeddingExtractor, load_attend_result


QUESTION = "How many legs does this wolf have?"
ALPHAS = [0.2, 0.5, 1.0, 1.5, 2.0]
BETAS = [0.01, 0.03, 0.05, 0.1, 0.2]
LAYERS = [4, 8, 12, 16, 20]
ATTENTION_TOP_RATIOS = [0.05, 0.10, 0.20]
UNCERTAINTY_TOP_RATIOS = [0.03, 0.05, 0.15]


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.reuse_stage_outputs:
        stage_dirs = {
            "envision": args.envision_dir.resolve(),
            "attend": args.attend_dir.resolve(),
            "respond": args.respond_dir.resolve(),
        }
    else:
        stage_dirs = run_stages(args, output_dir)

    collect_visual_debug(stage_dirs["envision"], stage_dirs["attend"], stage_dirs["respond"], output_dir)
    write_debug_config(args, stage_dirs, output_dir)

    summary = run_respond_diagnostics(args, stage_dirs["attend"], stage_dirs["respond"], output_dir)
    summary.update(build_static_summary(stage_dirs["envision"], stage_dirs["attend"], stage_dirs["respond"]))
    (output_dir / "debug_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print_core_conclusion(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Case-driven EnAR debug script for the five-leg wolf sample.")
    parser.add_argument("--image-path", type=Path, default=PROJECT_ROOT / "Envision/image/data/wolf_5.png")
    parser.add_argument("--question", default=QUESTION)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/debug_five_leg_wolf")
    parser.add_argument("--llava-model-dir", type=Path, default=PROJECT_ROOT / "pre_model/LLM/llava-1.5-7b-hf")
    parser.add_argument("--sd-model-dir", type=Path, default=PROJECT_ROOT / "pre_model/DDIM/stable-diffusion-v1-5")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--reuse-stage-outputs", action="store_true")
    parser.add_argument("--skip-envision", action="store_true")
    parser.add_argument("--skip-attend-ablation", action="store_true")
    parser.add_argument("--skip-alpha-beta-sweep", action="store_true")
    parser.add_argument("--sweep-limit", type=int, default=0, help="Maximum alpha/beta pairs to evaluate; 0 disables the sweep by default.")
    parser.add_argument("--envision-dir", type=Path, default=PROJECT_ROOT / "outputs/envision/run_001")
    parser.add_argument("--attend-dir", type=Path, default=PROJECT_ROOT / "outputs/attend/run_001")
    parser.add_argument("--respond-dir", type=Path, default=PROJECT_ROOT / "outputs/respond/run_001")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--preprocess-mode", choices=["pad", "center_crop"], default="pad")
    parser.add_argument("--pad-color", type=int, nargs=3, default=[127, 127, 127], metavar=("R", "G", "B"))
    parser.add_argument("--num-ddim-steps", type=int, default=50)
    parser.add_argument("--inversion-step-T", type=int, default=30)
    parser.add_argument("--langevin-steps", type=int, default=10)
    parser.add_argument("--eta-start", type=float, default=0.05)
    parser.add_argument("--eta-end", type=float, default=0.005)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--attention-layer", type=int, default=8)
    parser.add_argument("--attention-top-ratio", type=float, default=0.10)
    parser.add_argument("--uncertainty-top-ratio", type=float, default=0.05)
    parser.add_argument("--uncertainty-weight", type=float, default=1.0)
    parser.add_argument("--padding-ratio-limit", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.03)
    parser.add_argument("--use-apc", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser.parse_args()


def run_stages(args: argparse.Namespace, output_dir: Path) -> dict[str, Path]:
    stage_root = output_dir / "stage_outputs"
    envision_dir = stage_root / "envision"
    attend_dir = stage_root / "attend"
    respond_dir = stage_root / "respond"
    if args.skip_envision:
        envision_dir = args.envision_dir.resolve()
    else:
        envision_config = EnvisionConfig(
            sd_model_dir=args.sd_model_dir,
            input_image=args.image_path,
            output_dir=envision_dir,
            num_ddim_steps=args.num_ddim_steps,
            inversion_step_T=args.inversion_step_T,
            langevin_steps_M=args.langevin_steps,
            sample_count_K=args.k,
            preprocess_mode=args.preprocess_mode,
            pad_color=tuple(args.pad_color),
            eta_start=args.eta_start,
            eta_end=args.eta_end,
            temperature_tau=args.tau,
            dtype=args.dtype,
            device=None if args.device == "auto" else args.device,
        )
        EnvisionPipeline(envision_config).run()

    attend_config = AttendConfig(
        llava_model_dir=args.llava_model_dir,
        original_image=envision_dir / "preprocessed.png",
        impression_image=envision_dir / "impression.png",
        uncertainty_map=envision_dir / "uncertainty_map.npy",
        envision_metadata=envision_dir / "metadata.json",
        output_dir=attend_dir,
        vision_layer_number=args.attention_layer,
        attention_top_ratio=args.attention_top_ratio,
        uncertainty_top_ratio=args.uncertainty_top_ratio,
        padding_ratio_limit=args.padding_ratio_limit,
        uncertainty_weight=args.uncertainty_weight,
        device=args.device,
        dtype=args.dtype,
    )
    AttendPipeline(attend_config).run()

    respond_config = RespondConfig(
        llava_model_dir=args.llava_model_dir,
        image_path=envision_dir / "preprocessed.png",
        attend_result_json=attend_dir / "attend_result.json",
        output_dir=respond_dir,
        question=args.question,
        alpha=args.alpha,
        use_apc=args.use_apc,
        apc_beta=args.beta,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        dtype=args.dtype,
        vision_feature_select_strategy="default",
        num_additional_image_tokens=1,
    )
    RespondPipeline(respond_config).run()
    if not args.skip_attend_ablation:
        run_attend_ablation(args, envision_dir, output_dir)
    return {"envision": envision_dir, "attend": attend_dir, "respond": respond_dir}


def run_attend_ablation(args: argparse.Namespace, envision_dir: Path, output_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    ablation_root = output_dir / "attend_ablation_runs"
    modes = [
        ("attention_only", 1.0, 1e-6, 0.0),
        ("uncertainty_only", 1e-6, 0.10, 1.0),
        ("attention_plus_uncertainty", None, None, args.uncertainty_weight),
    ]
    for layer in LAYERS:
        for top_ratio in ATTENTION_TOP_RATIOS:
            for unc_ratio in UNCERTAINTY_TOP_RATIOS:
                for mode, mode_attn, mode_unc, unc_weight in modes:
                    run_dir = ablation_root / f"layer_{layer:02d}_{mode}_a{top_ratio:g}_u{unc_ratio:g}"
                    config = AttendConfig(
                        llava_model_dir=args.llava_model_dir,
                        original_image=envision_dir / "preprocessed.png",
                        impression_image=envision_dir / "impression.png",
                        uncertainty_map=envision_dir / "uncertainty_map.npy",
                        envision_metadata=envision_dir / "metadata.json",
                        output_dir=run_dir,
                        vision_layer_number=layer,
                        attention_top_ratio=mode_attn if mode_attn is not None else top_ratio,
                        uncertainty_top_ratio=mode_unc if mode_unc is not None else unc_ratio,
                        padding_ratio_limit=args.padding_ratio_limit,
                        uncertainty_weight=unc_weight,
                        device=args.device,
                        dtype=args.dtype,
                    )
                    try:
                        AttendPipeline(config).run()
                        attend = load_json(run_dir / "attend_result.json")
                        rows.append(ablation_row(mode, layer, top_ratio, unc_ratio, attend, None))
                    except Exception as exc:
                        rows.append({"mode": mode, "layer": layer, "attention_top_ratio": top_ratio, "uncertainty_top_ratio": unc_ratio, "error": str(exc)})
    write_csv(output_dir / "ablation_results.csv", rows)


def ablation_row(mode: str, layer: int, attention_top_ratio: float, uncertainty_top_ratio: float, attend: dict[str, Any], error: str | None) -> dict[str, Any]:
    selected = attend.get("selected_patch_indices", [])
    patch_grid = attend.get("patch_grid", [24, 24])
    coords = patch_indices_to_coords(selected[:20], patch_grid, attend.get("patch_size", 14))
    score_stats = attend.get("score_stats", {})
    return {
        "mode": mode,
        "layer": layer,
        "attention_top_ratio": attention_top_ratio,
        "uncertainty_top_ratio": uncertainty_top_ratio,
        "selected_count": len(selected),
        "selected_bbox_grid_xyxy": json.dumps(patch_bbox(selected, patch_grid)),
        "top_selected_patch_indices": json.dumps(selected[:20]),
        "top_selected_patch_coords": json.dumps(coords),
        "delta_attention_std": nested_get(score_stats, ["delta_attention", "std"]),
        "uncertainty_patch_std": nested_get(score_stats, ["uncertainty_patch", "std"]),
        "source_counts": json.dumps(attend.get("source_counts", {}), ensure_ascii=False),
        "error": error,
    }


def collect_visual_debug(envision_dir: Path, attend_dir: Path, respond_dir: Path, output_dir: Path) -> None:
    copy_named(envision_dir / "original.png", output_dir / "original.png")
    copy_named(envision_dir / "preprocessed.png", output_dir / "processed.png")
    copy_named(envision_dir / "impression.png", output_dir / "representative_impression.png")
    copy_named(envision_dir / "uncertainty_heatmap.png", output_dir / "uncertainty_heatmap.png")
    copy_named(attend_dir / "original_attention_heatmap.png", output_dir / "original_attention_heatmap.png")
    copy_named(attend_dir / "counterfactual_attention_heatmap.png", output_dir / "counterfactual_attention_heatmap.png")
    copy_named(attend_dir / "contrastive_attention_heatmap.png", output_dir / "delta_attention_heatmap.png")
    copy_named(attend_dir / "fused_score_heatmap.png", output_dir / "fused_score_heatmap.png")
    copy_named(attend_dir / "selected_patch_mask.png", output_dir / "selected_mask.png")
    copy_named(attend_dir / "mask_origin_overlay.png", output_dir / "selected_mask_overlay.png")
    copy_named(respond_dir / "answer_regular.txt", output_dir / "baseline_answer.txt")
    copy_named(respond_dir / "answer_enar.txt", output_dir / "final_answer.txt")

    original = output_dir / "processed.png"
    heatmap = output_dir / "uncertainty_heatmap.png"
    if original.is_file() and heatmap.is_file():
        save_overlay(original, heatmap, output_dir / "uncertainty_overlay.png")

    samples_out = output_dir / "visual_impressions"
    samples_out.mkdir(exist_ok=True)
    sample_paths = sorted((envision_dir / "samples").glob("sample_*.png"))
    for sample in sample_paths:
        copy_named(sample, samples_out / sample.name)
    if not (output_dir / "original_attention_heatmap.png").is_file() and (attend_dir / "contrastive_attention.npy").is_file():
        save_heatmap_from_npy(attend_dir / "contrastive_attention.npy", output_dir / "original_attention_heatmap.png")
    write_visual_impression_metrics(envision_dir, output_dir)
    write_debug_shapes(envision_dir, attend_dir, respond_dir, output_dir)


def run_respond_diagnostics(args: argparse.Namespace, attend_dir: Path, respond_dir: Path, output_dir: Path) -> dict[str, Any]:
    respond = load_json(respond_dir / "respond_result.json") if (respond_dir / "respond_result.json").is_file() else {}
    baseline_answer = read_text(respond_dir / "answer_regular.txt") or respond.get("regular_answer", "")
    final_answer = read_text(respond_dir / "answer_enar.txt") or respond.get("enar_answer", "")
    trace_path = Path(respond.get("decode_trace_path", respond_dir / "decode_trace.json"))

    token_rows: list[dict[str, Any]] = []
    alpha_beta_rows: list[dict[str, Any]] = []
    best_alpha_beta = None
    model_debug = None
    try:
        model_debug = compute_model_token_debug(args, attend_dir, output_dir)
        token_rows = model_debug["token_rows"]
        alpha_beta_rows = model_debug["alpha_beta_rows"]
        best_alpha_beta = model_debug["best_alpha_beta"]
    except Exception as exc:
        token_rows = token_rows_from_existing_trace(trace_path)
        alpha_beta_rows = []
        best_alpha_beta = None
        (output_dir / "model_debug_error.txt").write_text(
            f"{exc}\n\n{traceback.format_exc()}",
            encoding="utf-8",
        )

    write_csv(output_dir / "token_probability_table.csv", token_rows)
    if alpha_beta_rows:
        write_csv(output_dir / "alpha_beta_sweep.csv", alpha_beta_rows)

    answer_is_five = contains_five(final_answer)
    answer_is_four = contains_four(final_answer)
    before = candidate_probs_from_rows(token_rows, "original_prob")
    after = candidate_probs_from_rows(token_rows, "final_prob")
    return {
        "baseline_answer": baseline_answer,
        "final_answer": final_answer,
        "answer_is_five": answer_is_five,
        "answer_is_four": answer_is_four,
        "best_alpha_beta": best_alpha_beta,
        "p_four_original": before.get("four"),
        "p_five_original": before.get("five"),
        "p_4_original": before.get("4"),
        "p_5_original": before.get("5"),
        "p_four_after_enar": after.get("four"),
        "p_five_after_enar": after.get("five"),
        "p_4_after_enar": after.get("4"),
        "p_5_after_enar": after.get("5"),
        "model_debug_available": model_debug is not None,
    }


def compute_model_token_debug(args: argparse.Namespace, attend_dir: Path, output_dir: Path) -> dict[str, Any]:
    attend = load_attend_result(attend_dir / "attend_result.json")
    components = LlavaGenerationLoader(
        args.llava_model_dir,
        device=args.device,
        dtype=args.dtype,
        vision_feature_select_strategy="default",
        num_additional_image_tokens=1,
    ).load()
    image_token_index = int(getattr(components.model.config, "image_token_index"))
    prompt = LlavaPromptBuilder().build(args.question)
    number_slot_prompt = prompt + " The wolf in the image has"
    encoded = MultimodalInputEncoder(components.processor, components.device).encode(
        attend["original_image"],
        number_slot_prompt,
        image_token_index=image_token_index,
    )
    visual_result = VisualEmbeddingExtractor(
        components.model,
        vision_feature_select_strategy="default",
        vision_feature_layer=components.generation_meta.get("vision_feature_layer", -2),
    ).extract(encoded.pixel_values, attend)
    padded_result = PaddedVisualInputBuilder(
        components.model,
        components.tokenizer,
        "matched_mean_visual_embedding",
    ).build(
        visual_result.visual_embeddings,
        visual_result.visual_token_layout["selected_vision_token_indices"],
    )

    base_loop = ContrastiveGenerationLoop(
        components.model,
        components.tokenizer,
        image_token_index=image_token_index,
        alpha=args.alpha,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        temperature=args.temperature,
        top_p=args.top_p,
        use_apc=args.use_apc,
        apc_beta=args.beta,
    )
    base_debug = base_loop.next_token_debug(
        encoded.input_ids,
        encoded.attention_mask,
        visual_result.visual_embeddings,
        padded_result.visual_embeddings_padded,
    )
    token_rows = flatten_candidate_rows(
        base_debug["candidate_token_probabilities"],
        step=0,
        alpha=args.alpha,
        beta=args.beta,
        final_answer="",
        context="number_slot_after_prefix",
    )

    alpha_beta_rows: list[dict[str, Any]] = []
    best = None
    if not args.skip_alpha_beta_sweep and args.sweep_limit > 0:
        sweep_count = 0
        for alpha in ALPHAS:
            for beta in BETAS:
                if sweep_count >= args.sweep_limit:
                    break
                loop = ContrastiveGenerationLoop(
                    components.model,
                    components.tokenizer,
                    image_token_index=image_token_index,
                    alpha=alpha,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    use_apc=True,
                    apc_beta=beta,
                )
                debug = loop.next_token_debug(
                    encoded.input_ids,
                    encoded.attention_mask,
                    visual_result.visual_embeddings,
                    padded_result.visual_embeddings_padded,
                )
                probs = table_probs(debug["candidate_token_probabilities"], "final_prob")
                row = {
                    "alpha": alpha,
                    "beta": beta,
                    "selected_token": debug["selected_token"],
                    "selected_token_id": debug["selected_token_id"],
                    "p_four": probs.get("four"),
                    "p_five": probs.get("five"),
                    "p_4": probs.get("4"),
                    "p_5": probs.get("5"),
                    "outputs_five": debug["selected_token"].strip().lower() in {"five", "5"},
                    "language_degenerate": False,
                    "apc_kept_count": nested_get(debug, ["apc", "kept_count"]),
                    "apc_filtered_count": nested_get(debug, ["apc", "filtered_count"]),
                }
                alpha_beta_rows.append(row)
                if best is None or score_alpha_beta(row) > score_alpha_beta(best):
                    best = row
                sweep_count += 1
            if sweep_count >= args.sweep_limit:
                break
    return {"token_rows": token_rows, "alpha_beta_rows": alpha_beta_rows, "best_alpha_beta": best}


def token_rows_from_existing_trace(trace_path: Path) -> list[dict[str, Any]]:
    if not trace_path.is_file():
        return []
    trace = load_json(trace_path).get("steps", [])
    selected = next((step for step in trace if any(row.get("candidate") in {"four", "five"} for row in step.get("candidate_token_probabilities", []))), None)
    if selected and selected.get("candidate_token_probabilities"):
        return flatten_candidate_rows(selected["candidate_token_probabilities"], selected.get("step", 0), selected.get("alpha"), nested_get(selected, ["apc", "beta"]), "")
    four_step = next((step for step in trace if step.get("selected_token", "").strip().lower() in {"four", "five", "4", "5"}), trace[0] if trace else {})
    rows = []
    for candidate in ("four", "five", "4", "5"):
        row = {"candidate": candidate, "step": four_step.get("step"), "note": "rerun script after trace instrumentation for candidate probabilities"}
        rows.append(row)
    return rows


def flatten_candidate_rows(
    table: list[dict[str, Any]],
    step: int,
    alpha: float | None,
    beta: float | None,
    final_answer: str,
    context: str = "",
) -> list[dict[str, Any]]:
    rows = []
    for row in table:
        rows.append(
            {
                "step": step,
                "candidate": row.get("candidate"),
                "token_ids": json.dumps(row.get("token_ids", [])),
                "tokens": json.dumps(row.get("tokens", []), ensure_ascii=False),
                "scoring": row.get("scoring"),
                "original_logit": row.get("original_logit"),
                "original_prob": row.get("original_prob"),
                "counterfactual_logit": row.get("counterfactual_logit"),
                "counterfactual_prob": row.get("counterfactual_prob"),
                "contrastive_logit": row.get("contrastive_logit"),
                "contrastive_prob": row.get("contrastive_prob"),
                "final_logit": row.get("final_logit"),
                "final_prob": row.get("final_prob"),
                "alpha": alpha,
                "beta": beta,
                "final_answer": final_answer,
                "context": context,
            }
        )
    return rows


def build_static_summary(envision_dir: Path, attend_dir: Path, respond_dir: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if (envision_dir / "metadata.json").is_file():
        meta = load_json(envision_dir / "metadata.json")
        summary["envision"] = {
            "K": nested_get(meta, ["config", "sample_count_K"]),
            "T": nested_get(meta, ["timestep_T"]),
            "inversion_step_T": nested_get(meta, ["config", "inversion_step_T"]),
            "langevin_steps": nested_get(meta, ["config", "langevin_steps_M"]),
            "eta_start": nested_get(meta, ["config", "eta_start"]),
            "eta_end": nested_get(meta, ["config", "eta_end"]),
            "tau": nested_get(meta, ["config", "temperature_tau"]),
            "representative_index": meta.get("representative_index"),
            "sample_diff_stats": meta.get("sample_diff_stats"),
            "uncertainty_stats": meta.get("uncertainty_stats"),
            "sample_debug": meta.get("sample_debug", [])[:2],
        }
    if (attend_dir / "attend_result.json").is_file():
        attend = load_json(attend_dir / "attend_result.json")
        summary["attend"] = {
            "attention_layer": attend.get("vision_layer_number"),
            "attention_top_ratio": attend.get("attention_top_ratio"),
            "uncertainty_top_ratio": attend.get("uncertainty_top_ratio"),
            "uncertainty_weight": attend.get("uncertainty_weight"),
            "patch_grid": attend.get("patch_grid"),
            "selected_patch_count": len(attend.get("selected_patch_indices", [])),
            "selected_bbox_grid_xyxy": patch_bbox(attend.get("selected_patch_indices", []), attend.get("patch_grid", [24, 24])),
            "top_selected_patch_indices": attend.get("selected_patch_indices", [])[:20],
            "score_stats": attend.get("score_stats"),
            "token_layout_meta": attend.get("token_layout_meta"),
        }
    if (respond_dir / "respond_result.json").is_file():
        respond = load_json(respond_dir / "respond_result.json")
        summary["respond"] = {
            "alpha": respond.get("alpha"),
            "use_apc": respond.get("use_apc"),
            "beta": respond.get("apc_beta"),
            "temperature": respond.get("temperature"),
            "top_p": respond.get("top_p"),
            "padding_meta": respond.get("padding_meta"),
        }
    return summary


def write_visual_impression_metrics(envision_dir: Path, output_dir: Path) -> None:
    original_path = envision_dir / "preprocessed.png"
    sample_paths = sorted((envision_dir / "samples").glob("sample_*.png"))
    if not original_path.is_file() or not sample_paths:
        return
    original = load_image_array(original_path)
    rows = []
    sample_arrays = [load_image_array(path, original.shape[:2]) for path in sample_paths]
    for idx, (path, sample) in enumerate(zip(sample_paths, sample_arrays)):
        diff = sample - original
        rows.append(
            {
                "sample_index": idx,
                "path": str(path),
                "l1_mean": float(np.abs(diff).mean()),
                "l2_mean": float(np.sqrt(np.square(diff).mean())),
                "ssim_like": float(1.0 - np.clip(np.abs(diff).mean(), 0.0, 1.0)),
                "leg_region_changed_manual_check": "inspect visual_impressions and uncertainty_overlay; no leg bbox annotation provided",
            }
        )
    pairwise = []
    for i in range(len(sample_arrays)):
        for j in range(i + 1, len(sample_arrays)):
            pairwise.append(float(np.abs(sample_arrays[i] - sample_arrays[j]).mean()))
    write_csv(output_dir / "visual_impression_metrics.csv", rows)
    (output_dir / "visual_impression_pairwise.json").write_text(
        json.dumps({"mean_pairwise_l1": float(np.mean(pairwise)) if pairwise else 0.0, "pairwise_l1": pairwise}, indent=2),
        encoding="utf-8",
    )


def write_debug_shapes(envision_dir: Path, attend_dir: Path, respond_dir: Path, output_dir: Path) -> None:
    data: dict[str, Any] = {}
    if (envision_dir / "uncertainty_map.npy").is_file():
        unc = np.load(envision_dir / "uncertainty_map.npy")
        data["uncertainty_map_shape"] = list(unc.shape)
        data["uncertainty_map_stats"] = stats(unc)
    if (attend_dir / "attend_result.json").is_file():
        attend = load_json(attend_dir / "attend_result.json")
        selected = attend.get("selected_patch_indices", [])
        data["input_ids_shape"] = nested_get(load_json(respond_dir / "respond_result.json") if (respond_dir / "respond_result.json").is_file() else {}, ["input_meta", "input_ids_shape"])
        data["image_token_positions"] = nested_get(load_json(respond_dir / "respond_result.json") if (respond_dir / "respond_result.json").is_file() else {}, ["input_meta", "image_token_positions"])
        data["image_token_start"] = min(data["image_token_positions"]) if data.get("image_token_positions") else None
        data["image_token_end"] = max(data["image_token_positions"]) if data.get("image_token_positions") else None
        data["num_vision_tokens"] = nested_get(attend, ["token_layout_meta", "num_patches"])
        data["inferred_patch_grid"] = attend.get("patch_grid")
        data["attention_tensor_shape"] = nested_get(attend, ["token_layout_meta", "raw_attention_shape"])
        data["selected_layer_index"] = nested_get(attend, ["token_layout_meta", "vision_layer_index"])
        data["selected_heads"] = "all_heads_mean"
        data["question_token_indices"] = "TODO: current Attend extracts CLIP vision self-attention, not question-token to vision-token attention."
        data["top_uncertainty_patch_indices"] = top_indices_from_array(attend_dir / "uncertainty_patch_scores.npy")
        data["top_selected_patch_indices"] = selected[:20]
        data["top_selected_patch_coords"] = patch_indices_to_coords(selected[:20], attend.get("patch_grid", [24, 24]), attend.get("patch_size", 14))
        data["fused_score_shape"] = [len(nested_get(attend, ["selected_patch_indices"]) or [])]
        data["assumption"] = "Patch coordinates are in LLaVA preprocessed square image space; inspect selected_mask_overlay.png for origin-space alignment."
    (output_dir / "debug_shapes.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_debug_config(args: argparse.Namespace, stage_dirs: dict[str, Path], output_dir: Path) -> None:
    data = {key: stringify_yaml_value(value) for key, value in vars(args).items()}
    data["stage_dirs"] = {key: str(value) for key, value in stage_dirs.items()}
    with (output_dir / "debug_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def print_core_conclusion(summary: dict[str, Any]) -> None:
    print("\n=== Five-leg wolf EnAR debug ===")
    print(f"baseline answer: {summary.get('baseline_answer')}")
    print(f"final answer: {summary.get('final_answer')}")
    print(f"answer_is_five: {summary.get('answer_is_five')}")
    print(f"answer_is_four: {summary.get('answer_is_four')}")
    print(f"best alpha/beta: {summary.get('best_alpha_beta')}")
    print(f"P(five) before / after: {summary.get('p_five_original')} / {summary.get('p_five_after_enar')}")
    print(f"P(four) before / after: {summary.get('p_four_original')} / {summary.get('p_four_after_enar')}")
    print("manual check: inspect selected_mask_overlay.png and uncertainty_overlay.png; selected mask should cover wolf legs, especially the extra leg.")


def contains_five(text: str) -> bool:
    lowered = f" {text.lower()} "
    return " five " in lowered or " 5 " in lowered or lowered.strip() == "5"


def contains_four(text: str) -> bool:
    lowered = f" {text.lower()} "
    return " four " in lowered or " 4 " in lowered or lowered.strip() == "4"


def load_image_array(path: Path, size_hw: tuple[int, int] | None = None) -> np.ndarray:
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if size_hw is not None:
        image = image.resize((size_hw[1], size_hw[0]), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def save_overlay(base_path: Path, heatmap_path: Path, out_path: Path, alpha: float = 0.45) -> None:
    base = ImageOps.exif_transpose(Image.open(base_path)).convert("RGBA")
    heat = ImageOps.exif_transpose(Image.open(heatmap_path)).convert("RGB").resize(base.size, Image.Resampling.BILINEAR).convert("RGBA")
    heat.putalpha(int(round(255 * alpha)))
    Image.alpha_composite(base, heat).convert("RGB").save(out_path)


def save_heatmap_from_npy(path: Path, out_path: Path) -> None:
    arr = np.load(path)
    side = int(round(math.sqrt(arr.size)))
    if side * side == arr.size:
        arr = arr.reshape(side, side)
    arr = normalize(arr)
    Image.fromarray((arr * 255).round().astype(np.uint8), mode="L").resize((336, 336), Image.Resampling.NEAREST).save(out_path)


def patch_indices_to_coords(indices: list[int], patch_grid: Any, patch_size: int) -> list[list[int]]:
    if not isinstance(patch_grid, (list, tuple)) or len(patch_grid) != 2:
        return []
    width = int(patch_grid[1])
    coords = []
    for idx in indices:
        y = int(idx) // width
        x = int(idx) % width
        coords.append([x * patch_size, y * patch_size, (x + 1) * patch_size, (y + 1) * patch_size])
    return coords


def patch_bbox(indices: list[int], patch_grid: Any) -> list[int] | None:
    if not indices or not isinstance(patch_grid, (list, tuple)) or len(patch_grid) != 2:
        return None
    width = int(patch_grid[1])
    ys = [int(idx) // width for idx in indices]
    xs = [int(idx) % width for idx in indices]
    return [min(xs), min(ys), max(xs), max(ys)]


def top_indices_from_array(path: Path, k: int = 20) -> list[int]:
    if not path.is_file():
        return []
    arr = np.load(path).reshape(-1)
    return [int(idx) for idx in np.argsort(-arr)[:k]]


def candidate_probs_from_rows(rows: list[dict[str, Any]], column: str) -> dict[str, float | None]:
    result = {}
    for row in rows:
        if column in row:
            value = row[column]
            result[str(row.get("candidate"))] = float(value) if value not in (None, "") else None
    return result


def table_probs(table: list[dict[str, Any]], column: str) -> dict[str, float | None]:
    return {str(row.get("candidate")): row.get(column) for row in table}


def score_alpha_beta(row: dict[str, Any]) -> float:
    p_five = float(row.get("p_five") or 0.0) + float(row.get("p_5") or 0.0)
    p_four = float(row.get("p_four") or 0.0) + float(row.get("p_4") or 0.0)
    return p_five - p_four


def stats(array: np.ndarray) -> dict[str, float]:
    arr = np.asarray(array, dtype=np.float32)
    return {"min": float(arr.min()), "max": float(arr.max()), "mean": float(arr.mean()), "std": float(arr.std())}


def normalize(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    mn = float(arr.min())
    mx = float(arr.max())
    if mx <= mn:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def nested_get(data: Any, keys: list[str], default: Any = None) -> Any:
    value = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def stringify_yaml_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [stringify_yaml_value(item) for item in value]
    if isinstance(value, dict):
        return {key: stringify_yaml_value(item) for key, item in value.items()}
    return value


def copy_named(src: Path, dst: Path) -> None:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip() if path.is_file() else ""


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
