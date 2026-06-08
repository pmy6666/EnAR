from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_MODEL_DIR = Path("EnAR/pre_model/LLM/llava-1.5-7b-hf")
DEFAULT_IMAGE = Path("EnAR/outputs/envision/run_001/preprocessed.png")
DEFAULT_PROMPT = "How many legs does this wolf have?"
DEFAULT_SCRIPT_CONFIG = Path("EnAR/work_scripts/llava_only_interactive_top_p_debug.yaml")


def main() -> int:
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"image does not exist: {image_path}")

    print("Loading LLaVA components...")
    processor, model, tokenizer, device = load_llava(
        model_dir=model_dir,
        device_name=args.device,
        dtype_name=args.dtype,
        vision_feature_select_strategy=args.vision_feature_select_strategy,
        num_additional_image_tokens=args.num_additional_image_tokens,
    )
    prompt = args.prompt if args.prompt_is_formatted else build_llava_prompt(args.prompt)
    encoded = encode_inputs(processor, image_path, prompt, device)
    image_token_index = int(getattr(model.config, "image_token_index"))
    image_positions = find_image_token_positions(encoded["input_ids"], image_token_index)

    print("Ready.")
    print(f"image: {image_path}")
    print(f"prompt: {prompt!r}")
    print(f"top_p: {args.top_p}, temperature: {args.temperature}, do_sample: {args.do_sample}")
    print(f"input_ids_shape: {tuple(encoded['input_ids'].shape)}")
    print(f"image_token_count: {len(image_positions)}")
    if args.print_prompt_token_map:
        print_prompt_token_map(tokenizer, encoded["input_ids"], image_positions)
    print("Press Enter to accept the selected token; type a token id to force it; type q to quit.")
    print()

    run_interactive_loop(
        model=model,
        tokenizer=tokenizer,
        input_ids=encoded["input_ids"],
        attention_mask=encoded.get("attention_mask"),
        pixel_values=encoded["pixel_values"],
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        temperature=args.temperature,
        do_sample=args.do_sample,
        force_greedy=args.force_greedy,
        limit=args.limit,
        show_special=args.show_special,
    )
    return 0


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--script_config", type=Path, default=DEFAULT_SCRIPT_CONFIG)
    pre_args, _ = pre_parser.parse_known_args()
    defaults = load_yaml_defaults(pre_args.script_config)

    parser = argparse.ArgumentParser(
        parents=[pre_parser],
        description="Interactively inspect LLaVA next-token top-p probabilities from an image and prompt."
    )
    parser.add_argument("--model_dir", type=Path, default=defaults.get("model_dir", DEFAULT_MODEL_DIR))
    parser.add_argument("--image", type=Path, default=defaults.get("image", DEFAULT_IMAGE))
    parser.add_argument("--prompt", type=str, default=defaults.get("prompt", DEFAULT_PROMPT))
    parser.add_argument(
        "--prompt-is-formatted",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("prompt_is_formatted", False)),
        help="Use --prompt exactly as the model prompt instead of wrapping it in LLaVA-1.5 chat format.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=defaults.get("max_new_tokens", 64))
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=bool(defaults.get("do_sample", False)))
    parser.add_argument("--top_p", type=float, default=defaults.get("top_p", 0.9))
    parser.add_argument("--temperature", type=float, default=defaults.get("temperature", 1.0))
    parser.add_argument("--device", type=str, default=defaults.get("device", "auto"))
    parser.add_argument("--dtype", choices=["float16", "float32", "bfloat16"], default=defaults.get("dtype", "float16"))
    parser.add_argument(
        "--vision_feature_select_strategy",
        choices=["default", "full"],
        default=defaults.get("vision_feature_select_strategy", "default"),
    )
    parser.add_argument("--num_additional_image_tokens", type=int, default=defaults.get("num_additional_image_tokens", 1))
    parser.add_argument("--limit", type=int, default=defaults.get("limit", 80), help="Max rows to print; 0 prints the whole top-p set.")
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
        help="Select argmax even if --do_sample is enabled.",
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


def load_llava(
    model_dir: Path,
    device_name: str,
    dtype_name: str,
    vision_feature_select_strategy: str,
    num_additional_image_tokens: int,
) -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from transformers import AutoProcessor, LlavaForConditionalGeneration
    except Exception as exc:
        raise RuntimeError(
            "Missing LLaVA runtime dependencies: torch, transformers, pillow, accelerate, "
            "safetensors, sentencepiece, protobuf."
        ) from exc

    device = resolve_device(torch, device_name)
    dtype = resolve_dtype(torch, dtype_name, device)
    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
    patch_processor_from_config(
        processor,
        model_dir,
        vision_feature_select_strategy=vision_feature_select_strategy,
        num_additional_image_tokens=num_additional_image_tokens,
    )
    tokenizer = getattr(processor, "tokenizer", None)
    model = LlavaForConditionalGeneration.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()
    return processor, model, tokenizer, device


def resolve_device(torch: Any, device_name: str) -> Any:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def resolve_dtype(torch: Any, dtype_name: str, device: Any) -> Any:
    if dtype_name == "float32" or str(device) == "cpu":
        return torch.float32
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float16


def patch_processor_from_config(
    processor: Any,
    model_dir: Path,
    vision_feature_select_strategy: str,
    num_additional_image_tokens: int,
) -> None:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        return
    config = json.loads(config_path.read_text(encoding="utf-8"))
    vision_config = config.get("vision_config", {})
    patch_size = vision_config.get("patch_size")
    image_token_index = config.get("image_token_index")
    if getattr(processor, "patch_size", None) is None and patch_size is not None:
        processor.patch_size = patch_size
    processor.num_additional_image_tokens = num_additional_image_tokens
    if getattr(processor, "vision_feature_select_strategy", None) is None and vision_feature_select_strategy:
        processor.vision_feature_select_strategy = vision_feature_select_strategy
    if getattr(processor, "image_token", None) is None and image_token_index is not None:
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None:
            image_token = tokenizer.convert_ids_to_tokens(image_token_index)
            if image_token:
                processor.image_token = image_token


def build_llava_prompt(question: str) -> str:
    question = question.strip()
    if not question:
        raise ValueError("prompt must not be empty.")
    return f"USER: <image>\n{question}\nASSISTANT:"


def encode_inputs(processor: Any, image_path: Path, prompt: str, device: Any) -> dict[str, Any]:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    return {key: value.to(device) for key, value in inputs.items()}


def run_interactive_loop(
    model: Any,
    tokenizer: Any,
    input_ids: Any,
    attention_mask: Any,
    pixel_values: Any,
    max_new_tokens: int,
    top_p: float,
    temperature: float,
    do_sample: bool,
    force_greedy: bool,
    limit: int,
    show_special: bool,
) -> None:
    import torch

    current_ids = input_ids
    current_mask = attention_mask
    generated: list[int] = []
    eos_ids = as_id_set(getattr(tokenizer, "eos_token_id", None), getattr(model.config, "eos_token_id", None))

    with torch.inference_mode():
        for step in range(max_new_tokens):
            outputs = model(input_ids=current_ids, attention_mask=current_mask, pixel_values=pixel_values)
            logits = outputs.logits[:, -1, :]
            selected_token_id, selected_logprob = select_next_token(
                logits,
                do_sample=False if force_greedy else do_sample,
                temperature=temperature,
                top_p=top_p,
            )
            rows = top_p_rows(
                logits,
                tokenizer,
                top_p=top_p,
                temperature=temperature,
                show_special=show_special,
            )
            print_step(
                step=step,
                rows=rows,
                selected_token_id=selected_token_id,
                selected_logprob=selected_logprob,
                generated_ids=generated,
                tokenizer=tokenizer,
                limit=limit,
            )
            user_value = input("next> ").strip()
            if user_value.lower() in {"q", "quit", "exit"}:
                break
            if user_value:
                try:
                    next_token_id = int(user_value)
                except ValueError:
                    print(f"Invalid token id: {user_value!r}; using selected token.")
                    next_token_id = selected_token_id
            else:
                next_token_id = selected_token_id

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


def select_next_token(logits: Any, do_sample: bool, temperature: float, top_p: float) -> tuple[int, float]:
    import torch

    logits = logits.float()
    if not do_sample:
        token = int(torch.argmax(logits, dim=-1).item())
        logprob = float(torch.log_softmax(logits, dim=-1)[0, token].item())
        return token, logprob
    filtered = top_p_filter(logits / temperature, top_p)
    probs = torch.softmax(filtered, dim=-1)
    if not torch.isfinite(probs).all() or float(probs.sum(dim=-1).min().item()) <= 0.0:
        filtered = logits / temperature
        probs = torch.softmax(filtered, dim=-1)
    token_tensor = torch.multinomial(probs, num_samples=1)
    token = int(token_tensor.item())
    logprob = float(torch.log(probs[0, token].clamp_min(1e-20)).item())
    return token, logprob


def top_p_filter(logits: Any, top_p: float) -> Any:
    import torch

    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    filtered = logits.clone()
    filtered.scatter_(
        dim=-1,
        index=sorted_indices,
        src=torch.where(remove, torch.full_like(sorted_logits, -torch.inf), sorted_logits),
    )
    return filtered


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
    selected_logprob: float,
    generated_ids: list[int],
    tokenizer: Any,
    limit: int,
) -> None:
    print()
    print("=" * 88)
    print(f"step {step}")
    print(f"generated_so_far: {decode_ids(tokenizer, generated_ids, skip_special_tokens=True)!r}")
    print(f"top_p_token_count: {len(rows)}")
    print(
        f"selected_token_id: {selected_token_id}, "
        f"token: {decode_ids(tokenizer, [selected_token_id], False)!r}, "
        f"logprob: {selected_logprob:.6g}"
    )
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


def print_prompt_token_map(tokenizer: Any, input_ids: Any, image_token_positions: list[int]) -> None:
    ids = [int(token_id) for token_id in input_ids[0].detach().cpu().tolist()]
    image_position_to_visual = {pos: idx for idx, pos in enumerate(image_token_positions)}
    print()
    print("Input prompt token map:")
    print(f"{'pos':>5} {'token_id':>8} {'visual_idx':>10} token")
    for pos, token_id in enumerate(ids):
        if pos in image_position_to_visual:
            token = "<image>"
            visual_text = str(image_position_to_visual[pos])
        else:
            token = decode_ids(tokenizer, [token_id], skip_special_tokens=False)
            token = token.replace("\n", "\\n").replace("\t", "\\t")
            visual_text = "-"
        print(f"{pos:>5} {token_id:>8} {visual_text:>10} {token!r}")
    print()


def find_image_token_positions(input_ids: Any, image_token_index: int) -> list[int]:
    matches = (input_ids[0] == image_token_index).nonzero(as_tuple=False)
    return [int(item.item()) for item in matches.flatten()]


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
