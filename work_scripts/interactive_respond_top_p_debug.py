from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENAR_ROOT = PROJECT_ROOT / "EnAR"
if str(ENAR_ROOT) not in sys.path:
    sys.path.insert(0, str(ENAR_ROOT))

DEFAULT_SCRIPT_CONFIG = Path("EnAR/work_scripts/interactive_respond_top_p_debug.yaml")

from Respond.config import RespondConfig
from Respond.dual_branch_forwarder import forward_language_model
from Respond.embedding_merge import build_inputs_embeds
from Respond.input_encoder import MultimodalInputEncoder
from Respond.model_loader import LlavaGenerationLoader
from Respond.prompt_builder import LlavaPromptBuilder
from Respond.token_selector import NextTokenSelector, top_p_filter
from Respond.visual_embeddings import VisualEmbeddingResult, build_visual_token_layout, load_attend_result


def main() -> int:
    args = parse_args()
    config = RespondConfig.from_yaml(args.config, project_root=PROJECT_ROOT)
    apply_preset(config, args.preset)
    apply_overrides(config, args)
    config.validate()

    print("Loading Respond components...")
    attend = load_attend_result(config.attend_result_json)
    prompt = LlavaPromptBuilder().build(config.question)
    components = LlavaGenerationLoader(
        config.llava_model_dir,
        device=config.device,
        dtype=config.dtype,
        vision_feature_select_strategy=config.vision_feature_select_strategy,
        num_additional_image_tokens=config.num_additional_image_tokens,
    ).load()
    image_token_index = int(getattr(components.model.config, "image_token_index"))
    encoded = MultimodalInputEncoder(components.processor, components.device).encode(
        config.image_path,
        prompt,
        image_token_index=image_token_index,
    )
    visual_result = extract_visual_embeddings_compat(
        components.model,
        encoded.pixel_values,
        attend,
        vision_feature_select_strategy=config.vision_feature_select_strategy,
        vision_feature_layer=components.generation_meta.get("vision_feature_layer", -2),
    )

    print("Ready.")
    print(f"question: {config.question}")
    print(f"image: {config.image_path}")
    print(f"top_p: {config.top_p}, temperature: {config.temperature}, do_sample: {config.do_sample}")
    print(f"visual tokens: {visual_result.visual_token_layout['token_count']}")
    if args.print_prompt_token_map:
        print_prompt_token_map(
            tokenizer=components.tokenizer,
            input_ids=encoded.input_ids,
            image_token_positions=encoded.image_token_positions,
        )
    print("Press Enter to accept the selected token; type a token id to force it; type q to quit.")
    print()

    run_interactive_loop(
        config=config,
        args=args,
        model=components.model,
        tokenizer=components.tokenizer,
        image_token_index=image_token_index,
        input_ids=encoded.input_ids,
        attention_mask=encoded.attention_mask,
        visual_embeddings=visual_result.visual_embeddings,
    )
    return 0


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--script_config", type=Path, default=DEFAULT_SCRIPT_CONFIG)
    pre_args, _ = pre_parser.parse_known_args()
    defaults = load_yaml_defaults(pre_args.script_config)

    parser = argparse.ArgumentParser(
        parents=[pre_parser],
        description="Interactively inspect Respond top-p token probabilities step by step."
    )
    parser.add_argument("--config", type=Path, default=defaults.get("config", Path("EnAR/Respond/respond_config.yaml")))
    parser.add_argument(
        "--preset",
        choices=["enar_paper", "config"],
        default=defaults.get("preset", "enar_paper"),
        help="Use paper-style EnAR Respond defaults, or keep the YAML config unchanged.",
    )
    parser.add_argument("--question", type=str, default=defaults.get("question"))
    parser.add_argument("--image_path", type=Path, default=defaults.get("image_path"))
    parser.add_argument("--attend_result_json", type=Path, default=defaults.get("attend_result_json"))
    parser.add_argument("--max_new_tokens", type=int, default=defaults.get("max_new_tokens"))
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=defaults.get("do_sample"))
    parser.add_argument("--top_p", type=float, default=defaults.get("top_p"))
    parser.add_argument("--temperature", type=float, default=defaults.get("temperature"))
    parser.add_argument("--device", type=str, default=defaults.get("device"))
    parser.add_argument("--dtype", choices=["float16", "float32", "bfloat16"], default=defaults.get("dtype"))
    parser.add_argument(
        "--vision_feature_select_strategy",
        choices=["default", "full"],
        default=defaults.get("vision_feature_select_strategy"),
    )
    parser.add_argument("--num_additional_image_tokens", type=int, default=defaults.get("num_additional_image_tokens"))
    parser.add_argument("--limit", type=int, default=defaults.get("limit", 0), help="Max rows to print; 0 prints the whole top-p set.")
    parser.add_argument(
        "--show-special",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("show_special", False)),
        help="Show special tokens in decoded form.",
    )
    parser.add_argument(
        "--force-greedy",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("force_greedy", False)),
        help="Select argmax even if config enables sampling.",
    )
    parser.add_argument(
        "--no-print-prompt-token-map",
        dest="print_prompt_token_map",
        action="store_false",
        help="Do not print input prompt token indices before generation.",
    )
    parser.set_defaults(print_prompt_token_map=bool(defaults.get("print_prompt_token_map", True)))
    return parser.parse_args()


def load_yaml_defaults(path: Path) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required when --script_config points to a YAML file.") from exc
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"script config must be a mapping: {path}")
    return data


def apply_preset(config: RespondConfig, preset: str) -> None:
    if preset == "config":
        return
    if preset != "enar_paper":
        raise ValueError(f"Unknown preset: {preset}")
    # Paper-style LLaVA-1.5 Respond input defaults for normal-visual debugging.
    config.do_sample = False
    config.temperature = 1.0
    config.top_p = 1.0
    config.vision_feature_select_strategy = "default"
    config.num_additional_image_tokens = 1


def apply_overrides(config: RespondConfig, args: argparse.Namespace) -> None:
    for key in (
        "question",
        "image_path",
        "attend_result_json",
        "max_new_tokens",
        "do_sample",
        "top_p",
        "temperature",
        "device",
        "dtype",
        "vision_feature_select_strategy",
        "num_additional_image_tokens",
    ):
        value = getattr(args, key)
        if value is not None:
            setattr(config, key, value)
    config.__post_init__()


def run_interactive_loop(
    config: RespondConfig,
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    image_token_index: int,
    input_ids: Any,
    attention_mask: Any,
    visual_embeddings: Any,
) -> None:
    import torch

    current_ids = input_ids
    current_mask = attention_mask
    generated: list[int] = []
    selector = NextTokenSelector(
        do_sample=False if args.force_greedy else config.do_sample,
        temperature=config.temperature,
        top_p=config.top_p,
    )
    eos_ids = as_id_set(getattr(tokenizer, "eos_token_id", None), getattr(model.config, "eos_token_id", None))

    with torch.inference_mode():
        for step in range(config.max_new_tokens):
            selected_logits = forward_normal_visual(
                model,
                current_ids,
                current_mask,
                visual_embeddings,
                image_token_index,
            )
            selection = selector.select(selected_logits)
            rows = top_p_rows(
                selected_logits,
                tokenizer,
                top_p=config.top_p,
                temperature=config.temperature,
                show_special=args.show_special,
            )
            print_step(
                step=step,
                rows=rows,
                selected_token_id=selection.token_id,
                generated_ids=generated,
                tokenizer=tokenizer,
                limit=args.limit,
            )
            user_value = input("next> ").strip()
            if user_value.lower() in {"q", "quit", "exit"}:
                break
            if user_value:
                try:
                    next_token_id = int(user_value)
                except ValueError:
                    print(f"Invalid token id: {user_value!r}; using selected token.")
                    next_token_id = selection.token_id
            else:
                next_token_id = selection.token_id

            generated.append(next_token_id)
            if next_token_id in eos_ids:
                print("EOS reached.")
                break
            next_id = torch.tensor([[next_token_id]], dtype=current_ids.dtype, device=current_ids.device)
            current_ids = torch.cat([current_ids, next_id], dim=-1)
            if current_mask is not None:
                next_mask = torch.ones((current_mask.shape[0], 1), dtype=current_mask.dtype, device=current_mask.device)
                current_mask = torch.cat([current_mask, next_mask], dim=-1)

    print()
    print("generated ids:", generated)
    print("generated text:", decode_ids(tokenizer, generated, skip_special_tokens=True))


def forward_normal_visual(
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    visual_embeddings: Any,
    image_token_index: int,
) -> Any:
    inputs_embeds = build_inputs_embeds(
        model,
        input_ids,
        visual_embeddings,
        image_token_index,
    )
    outputs = forward_language_model(model, inputs_embeds, attention_mask)
    return outputs.logits[:, -1, :]


def extract_visual_embeddings_compat(
    model: Any,
    pixel_values: Any,
    attend_result: dict[str, Any],
    vision_feature_select_strategy: str,
    vision_feature_layer: int | list[int] | None,
) -> VisualEmbeddingResult:
    import torch

    with torch.inference_mode():
        image_features = get_image_features_compat(
            model,
            pixel_values,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
        )
    if isinstance(image_features, (list, tuple)):
        image_features = torch.stack(list(image_features), dim=0)
    layout = build_visual_token_layout(image_features, attend_result)
    return VisualEmbeddingResult(image_features, layout)


def get_image_features_compat(
    model: Any,
    pixel_values: Any,
    vision_feature_layer: int | list[int] | None,
    vision_feature_select_strategy: str,
) -> Any:
    get_image_features = getattr(model, "get_image_features", None)
    if get_image_features is None:
        base = getattr(model, "model", model)
        get_image_features = getattr(base, "get_image_features", None)
    if get_image_features is None:
        raise AttributeError("LLaVA model does not expose get_image_features.")

    kwargs = {
        "pixel_values": pixel_values,
        "vision_feature_layer": vision_feature_layer,
        "vision_feature_select_strategy": vision_feature_select_strategy,
    }
    try:
        out = get_image_features(**kwargs, return_dict=True)
    except TypeError as exc:
        if "return_dict" not in str(exc):
            raise
        out = get_image_features(**kwargs)
    return getattr(out, "pooler_output", out)


def top_p_rows(
    logits: Any,
    tokenizer: Any,
    top_p: float,
    temperature: float,
    show_special: bool,
) -> list[dict[str, Any]]:
    import torch

    scaled = logits.float() / temperature
    filtered = top_p_filter(scaled, top_p)
    filtered_probs = torch.softmax(filtered, dim=-1)
    full_probs = torch.softmax(scaled, dim=-1)
    keep = torch.isfinite(filtered[0])
    token_ids = keep.nonzero(as_tuple=False).flatten()
    probs = filtered_probs[0, token_ids]
    full = full_probs[0, token_ids]
    order = torch.argsort(probs, descending=True)
    rows = []
    for rank, item in enumerate(order.tolist(), start=1):
        token_id = int(token_ids[item].item())
        token_text = decode_ids(tokenizer, [token_id], skip_special_tokens=not show_special)
        rows.append(
            {
                "rank": rank,
                "token_id": token_id,
                "token": token_text,
                "prob_top_p": float(probs[item].detach().cpu()),
                "prob_full": float(full[item].detach().cpu()),
                "logit": float(logits.float()[0, token_id].detach().cpu()),
            }
        )
    return rows


def print_step(
    step: int,
    rows: list[dict[str, Any]],
    selected_token_id: int,
    generated_ids: list[int],
    tokenizer: Any,
    limit: int,
) -> None:
    print()
    print("=" * 88)
    print(f"step {step}")
    print(f"generated_so_far: {decode_ids(tokenizer, generated_ids, skip_special_tokens=True)!r}")
    print(f"top_p_token_count: {len(rows)}")
    print(f"selected_token_id: {selected_token_id}, token: {decode_ids(tokenizer, [selected_token_id], False)!r}")
    print("-" * 88)
    print(f"{'rank':>4} {'id':>8} {'p_top_p':>12} {'p_full':>12} {'logit':>12} token")
    rows_to_print = rows if limit <= 0 else rows[:limit]
    for row in rows_to_print:
        token = row["token"].replace("\n", "\\n").replace("\t", "\\t")
        print(
            f"{row['rank']:>4} {row['token_id']:>8} "
            f"{row['prob_top_p']:>12.6g} {row['prob_full']:>12.6g} {row['logit']:>12.6g} {token!r}"
        )
    if limit > 0 and len(rows) > limit:
        print(f"... {len(rows) - limit} more top-p tokens hidden by --limit")


def print_prompt_token_map(
    tokenizer: Any,
    input_ids: Any,
    image_token_positions: list[int],
) -> None:
    ids = [int(token_id) for token_id in input_ids[0].detach().cpu().tolist()]
    image_position_to_visual = {pos: idx for idx, pos in enumerate(image_token_positions)}
    print()
    print("Input prompt token map:")
    print(f"{'pos':>5} {'token_id':>8} {'visual_idx':>10} token")
    for pos, token_id in enumerate(ids):
        if pos in image_position_to_visual:
            visual_idx = image_position_to_visual[pos]
            token = "<image>"
            visual_text = str(visual_idx)
        else:
            token = decode_ids(tokenizer, [token_id], skip_special_tokens=False)
            token = token.replace("\n", "\\n").replace("\t", "\\t")
            visual_text = "-"
        print(f"{pos:>5} {token_id:>8} {visual_text:>10} {token!r}")
    print()


def decode_ids(tokenizer: Any, token_ids: list[int], skip_special_tokens: bool) -> str:
    if not token_ids:
        return ""
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=False,
    )


def as_id_set(*ids: Any) -> set[int]:
    result: set[int] = set()
    for item in ids:
        if item is None:
            continue
        if isinstance(item, (list, tuple, set)):
            result.update(int(x) for x in item if x is not None)
        else:
            result.add(int(item))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
