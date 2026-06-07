#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL_DIR = Path("/home/qianustb/EnAR/pre_model/LLM/llava-1.5-7b-hf")
DEFAULT_IMAGE = Path("/home/qianustb/EnAR/Envision/image/data/wolf_5.png")
DEFAULT_QUESTION = "How many legs does this animal have?"
DEFAULT_OUTPUT_DIR = Path("/home/qianustb/EnAR/VCD/my_work/outputs")


def _ensure_vcd_on_path() -> None:
    this_file = Path(__file__).resolve()
    vcd_root = this_file.parents[1]
    for path in (vcd_root, vcd_root / "experiments"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_ensure_vcd_on_path()

from vcd_utils.vcd_add_noise import add_diffusion_noise  # noqa: E402


@dataclass
class VcdArgs:
    model_dir: Path
    image: Path
    question: str
    output_dir: Path
    noise_step: int
    cd_alpha: float
    cd_beta: float
    max_new_tokens: int
    log_first_n_tokens: int
    top_k_logit_dump: int
    do_sample: bool
    temperature: float
    top_p: float
    top_k: int | None
    seed: int
    device: str
    dtype: str
    vision_feature_select_strategy: str
    num_additional_image_tokens: int
    output_prefix: str


def build_prompt(question: str) -> str:
    return f"USER: <image>\n{question}\nASSISTANT:"


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
    if getattr(processor, "vision_feature_select_strategy", None) is None:
        processor.vision_feature_select_strategy = vision_feature_select_strategy
    if getattr(processor, "image_token", None) is None and image_token_index is not None:
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None:
            image_token = tokenizer.convert_ids_to_tokens(image_token_index)
            if image_token:
                processor.image_token = image_token


def resolve_device(torch: Any, requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def resolve_dtype(torch: Any, requested: str, device: Any):
    if requested == "float32" or str(device) == "cpu":
        return torch.float32
    if requested == "bfloat16":
        return torch.bfloat16
    if requested == "float16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {requested}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_inputs(args: VcdArgs, processor: Any, device: Any, dtype: Any):
    from PIL import Image

    if not args.image.is_file():
        raise FileNotFoundError(f"Input image does not exist: {args.image}")
    image = Image.open(args.image).convert("RGB")
    prompt = build_prompt(args.question)
    encoded = processor(text=prompt, images=image, return_tensors="pt")

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    pixel_values = encoded["pixel_values"]
    noisy_pixel = add_diffusion_noise(pixel_values[0].cpu(), args.noise_step).unsqueeze(0)
    pixel_values = pixel_values.to(device=device, dtype=dtype)
    noisy_pixel = noisy_pixel.to(device=device, dtype=dtype)
    return {
        "prompt": prompt,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "pixel_values_cd": noisy_pixel,
    }


def compute_vcd_logits(torch: Any, origin_logits: Any, distorted_logits: Any, cd_alpha: float, cd_beta: float):
    origin = origin_logits.float()
    distorted = distorted_logits.float()
    raw_vcd = (1.0 + cd_alpha) * origin - cd_alpha * distorted
    if cd_beta == 0:
        cutoff = torch.full_like(origin.max(dim=-1, keepdim=True).values, -torch.inf)
    else:
        cutoff = math.log(cd_beta) + origin.max(dim=-1, keepdim=True).values
    keep_mask = origin >= cutoff
    if not bool(keep_mask.any(dim=-1).all()):
        top_idx = origin.argmax(dim=-1, keepdim=True)
        keep_mask = keep_mask.scatter(dim=-1, index=top_idx, value=True)
    final_vcd = raw_vcd.masked_fill(~keep_mask, -torch.inf)
    return raw_vcd, final_vcd, {
        "cutoff": float(cutoff.reshape(-1)[0].detach().cpu()),
        "kept_count": int(keep_mask.sum(dim=-1).min().detach().cpu()),
        "filtered_count": int((~keep_mask).sum(dim=-1).max().detach().cpu()),
    }


def filter_top_k_top_p(torch: Any, logits: Any, top_k: int | None, top_p: float):
    filtered = logits.clone()
    if top_k is not None and top_k > 0 and top_k < filtered.shape[-1]:
        values, _ = torch.topk(filtered, top_k, dim=-1)
        cutoff = values[..., -1, None]
        filtered = filtered.masked_fill(filtered < cutoff, -torch.inf)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
        filtered = torch.full_like(filtered, -torch.inf)
        filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    return filtered


def select_next_token(torch: Any, logits: Any, args: VcdArgs, generator: Any):
    logits = logits.float()
    if not args.do_sample:
        token = int(torch.argmax(logits, dim=-1).item())
        logprob = float(torch.log_softmax(logits, dim=-1)[0, token].detach().cpu())
        return token, logprob

    sampling_logits = logits / args.temperature
    sampling_logits = filter_top_k_top_p(torch, sampling_logits, args.top_k, args.top_p)
    probs = torch.softmax(sampling_logits, dim=-1)
    if not torch.isfinite(probs).all() or float(probs.sum(dim=-1).min().detach().cpu()) <= 0.0:
        probs = torch.softmax(logits / args.temperature, dim=-1)
    token_tensor = torch.multinomial(probs, num_samples=1, generator=generator)
    token = int(token_tensor.item())
    logprob = float(torch.log(probs[0, token].clamp_min(1e-20)).detach().cpu())
    return token, logprob


def top_logits(torch: Any, logits: Any, tokenizer: Any, k: int) -> list[dict[str, Any]]:
    values, indices = torch.topk(logits.float(), k=min(k, logits.shape[-1]), dim=-1)
    items = []
    for rank, (value, idx) in enumerate(zip(values[0], indices[0]), start=1):
        token_id = int(idx.item())
        items.append(
            {
                "rank": rank,
                "token_id": token_id,
                "token": tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False),
                "logit": float(value.detach().cpu()),
            }
        )
    return items


def eos_id_set(tokenizer: Any, model: Any) -> set[int]:
    result: set[int] = set()
    for value in (getattr(tokenizer, "eos_token_id", None), getattr(model.config, "eos_token_id", None)):
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            result.update(int(v) for v in value if v is not None)
        else:
            result.add(int(value))
    return result


def run_vcd(args: VcdArgs) -> dict[str, Any]:
    import torch
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    set_seed(args.seed)
    device = resolve_device(torch, args.device)
    dtype = resolve_dtype(torch, args.dtype, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(str(args.model_dir), local_files_only=True)
    patch_processor_from_config(
        processor,
        args.model_dir,
        args.vision_feature_select_strategy,
        args.num_additional_image_tokens,
    )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("AutoProcessor did not expose a tokenizer.")

    model = LlavaForConditionalGeneration.from_pretrained(
        str(args.model_dir),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()

    inputs = prepare_inputs(args, processor, device, dtype)
    current_ids = inputs["input_ids"]
    current_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"]
    pixel_values_cd = inputs["pixel_values_cd"]
    generated: list[int] = []
    logged_steps: list[dict[str, Any]] = []
    origin_logits_dump = []
    distorted_logits_dump = []
    vcd_logits_dump = []
    raw_vcd_logits_dump = []
    eos_ids = eos_id_set(tokenizer, model)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    with torch.inference_mode():
        for step in range(args.max_new_tokens):
            forward_kwargs = {
                "input_ids": current_ids,
                "attention_mask": current_mask,
                "pixel_values": pixel_values,
                "vision_feature_select_strategy": args.vision_feature_select_strategy,
                "use_cache": False,
            }
            origin_logits = model(**forward_kwargs).logits[:, -1, :]
            forward_kwargs["pixel_values"] = pixel_values_cd
            distorted_logits = model(**forward_kwargs).logits[:, -1, :]
            raw_vcd_logits, final_vcd_logits, apc_meta = compute_vcd_logits(
                torch,
                origin_logits,
                distorted_logits,
                args.cd_alpha,
                args.cd_beta,
            )
            next_token, logprob = select_next_token(torch, final_vcd_logits, args, generator)
            generated.append(next_token)

            if step < args.log_first_n_tokens:
                origin_cpu = origin_logits.float().detach().cpu()[0]
                distorted_cpu = distorted_logits.float().detach().cpu()[0]
                raw_vcd_cpu = raw_vcd_logits.float().detach().cpu()[0]
                final_vcd_cpu = final_vcd_logits.float().detach().cpu()[0]
                origin_logits_dump.append(origin_cpu)
                distorted_logits_dump.append(distorted_cpu)
                raw_vcd_logits_dump.append(raw_vcd_cpu)
                vcd_logits_dump.append(final_vcd_cpu)
                logged_steps.append(
                    {
                        "step": step,
                        "generated_token_id": next_token,
                        "generated_token": tokenizer.decode(
                            [next_token],
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        ),
                        "selected_token_logprob": logprob,
                        "apc": apc_meta,
                        "origin_top_logits": top_logits(torch, origin_logits, tokenizer, args.top_k_logit_dump),
                        "distorted_top_logits": top_logits(torch, distorted_logits, tokenizer, args.top_k_logit_dump),
                        "vcd_raw_top_logits": top_logits(torch, raw_vcd_logits, tokenizer, args.top_k_logit_dump),
                        "vcd_top_logits": top_logits(torch, final_vcd_logits, tokenizer, args.top_k_logit_dump),
                    }
                )

            if next_token in eos_ids:
                break
            next_id = torch.tensor([[next_token]], dtype=current_ids.dtype, device=device)
            current_ids = torch.cat([current_ids, next_id], dim=-1)
            if current_mask is not None:
                next_mask = torch.ones((current_mask.shape[0], 1), dtype=current_mask.dtype, device=device)
                current_mask = torch.cat([current_mask, next_mask], dim=-1)

    answer = tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
    output_prefix = args.output_prefix or f"{args.image.stem}_llava_vcd"
    logits_path = args.output_dir / f"{output_prefix}_logits.pt"
    summary_path = args.output_dir / f"{output_prefix}_summary.json"

    metadata = {
        "image": str(args.image),
        "question": args.question,
        "prompt": inputs["prompt"],
        "model_dir": str(args.model_dir),
        "params": serializable_args(args),
        "device": str(device),
        "dtype": str(dtype),
        "prompt_len": int(inputs["input_ids"].shape[-1]),
        "vocab_size": int(model.config.vocab_size),
        "logits_note": "vcd_logits are after Adaptive Plausibility Constraints; vcd_raw_logits are before APC.",
    }
    torch.save(
        {
            "origin_logits": stack_or_empty(torch, origin_logits_dump),
            "distorted_logits": stack_or_empty(torch, distorted_logits_dump),
            "vcd_logits": stack_or_empty(torch, vcd_logits_dump),
            "vcd_raw_logits": stack_or_empty(torch, raw_vcd_logits_dump),
            "generated_token_ids": torch.tensor(generated, dtype=torch.long),
            "prompt_input_ids": inputs["input_ids"].detach().cpu()[0],
            "metadata": metadata,
        },
        logits_path,
    )
    summary = {
        "image": str(args.image),
        "question": args.question,
        "answer": answer,
        "model_dir": str(args.model_dir),
        "params": serializable_args(args),
        "generated_token_ids": generated,
        "generated_tokens": [
            tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            for token_id in generated
        ],
        "logged_steps": logged_steps,
        "logits_path": str(logits_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "answer": answer,
        "summary_path": str(summary_path),
        "logits_path": str(logits_path),
        "generated_token_count": len(generated),
        "logged_step_count": len(logged_steps),
    }


def stack_or_empty(torch: Any, tensors: list[Any]):
    if tensors:
        return torch.stack(tensors, dim=0)
    return torch.empty((0,), dtype=torch.float32)


def serializable_args(args: VcdArgs) -> dict[str, Any]:
    data = asdict(args)
    for key in ("model_dir", "image", "output_dir"):
        data[key] = str(data[key])
    return data


def parse_args() -> VcdArgs:
    parser = argparse.ArgumentParser(description="Run single-image LLaVA VCD inference with HF Transformers.")
    parser.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--cd_alpha", type=float, default=1.0)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--log_first_n_tokens", type=int, default=20)
    parser.add_argument("--top_k_logit_dump", type=int, default=20)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--vision_feature_select_strategy", choices=["full", "default"], default="full")
    parser.add_argument("--num_additional_image_tokens", type=int, default=1)
    parser.add_argument("--output_prefix", default="")
    ns = parser.parse_args()

    if not 0 <= ns.noise_step <= 999:
        parser.error("--noise_step must be in [0, 999].")
    if ns.cd_alpha < 0:
        parser.error("--cd_alpha must be non-negative.")
    if not 0 <= ns.cd_beta <= 1:
        parser.error("--cd_beta must be in [0, 1].")
    if ns.max_new_tokens <= 0:
        parser.error("--max_new_tokens must be positive.")
    if ns.log_first_n_tokens < 0:
        parser.error("--log_first_n_tokens must be non-negative.")
    if ns.top_k_logit_dump <= 0:
        parser.error("--top_k_logit_dump must be positive.")
    if ns.temperature <= 0:
        parser.error("--temperature must be positive.")
    if not 0 < ns.top_p <= 1:
        parser.error("--top_p must be in (0, 1].")
    if ns.top_k is not None and ns.top_k <= 0:
        parser.error("--top_k must be positive when provided.")
    return VcdArgs(**vars(ns))


def main() -> int:
    args = parse_args()
    try:
        result = run_vcd(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
