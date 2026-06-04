#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


REQUIRED_FILES = [
    "model_index.json",
    "scheduler/scheduler_config.json",
    "tokenizer/merges.txt",
    "tokenizer/vocab.json",
    "tokenizer/tokenizer_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "unet/config.json",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    "feature_extractor/preprocessor_config.json",
    "safety_checker/config.json",
    "safety_checker/model.safetensors",
]


def check_files(model_dir: Path) -> bool:
    print(f"Model dir: {model_dir}")
    if not model_dir.is_dir():
        print(f"ERROR: model directory does not exist: {model_dir}", file=sys.stderr)
        return False

    missing = []
    for rel_path in REQUIRED_FILES:
        path = model_dir / rel_path
        if path.is_file() and path.stat().st_size > 0:
            print(f"OK      {rel_path} ({path.stat().st_size} bytes)")
        else:
            print(f"MISSING {rel_path}")
            missing.append(rel_path)

    parts = sorted(model_dir.rglob("*.part"))
    if parts:
        print("\nFound leftover .part files:")
        for path in parts:
            print(f"PART    {path.relative_to(model_dir)} ({path.stat().st_size} bytes)")

    if missing:
        print("\nERROR: required files are missing.", file=sys.stderr)
        return False
    return True


def check_model_index(model_dir: Path) -> None:
    index_path = model_dir / "model_index.json"
    if not index_path.is_file():
        return
    try:
        data = json.loads(index_path.read_text())
    except Exception as exc:
        print(f"WARNING: failed to parse model_index.json: {exc}")
        return
    print("\nmodel_index.json components:")
    for key in ["scheduler", "tokenizer", "text_encoder", "unet", "vae", "feature_extractor", "safety_checker"]:
        print(f"  {key}: {data.get(key)}")


def load_pipeline(
    model_dir: Path,
    generate: bool,
    output: Path,
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    seed: int,
    disable_safety_checker: bool,
) -> bool:
    try:
        import torch
        from diffusers import StableDiffusionPipeline
    except Exception as exc:
        print(
            "\nERROR: missing runtime dependencies.\n"
            f"Import error: {exc}\n"
            "Install with:\n"
            "  /home/qianustb/EnAR/env/bin/python -m pip install -U torch diffusers transformers safetensors accelerate",
            file=sys.stderr,
        )
        return False

    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    device = "cuda" if use_cuda else "cpu"
    print(f"\nLoading StableDiffusionPipeline on {device} with dtype={dtype} ...")

    load_kwargs = {
        "torch_dtype": dtype,
        "use_safetensors": True,
        "local_files_only": True,
    }
    if disable_safety_checker:
        load_kwargs["safety_checker"] = None
        load_kwargs["requires_safety_checker"] = False
        print("Safety checker disabled for this local smoke test.")

    pipe = StableDiffusionPipeline.from_pretrained(str(model_dir), **load_kwargs)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=False)
    print("Pipeline loaded successfully.")

    if generate:
        try:
            import numpy as np
        except Exception:
            np = None

        print(f"Generating smoke-test image: {output}")
        print(f"Prompt: {prompt}")
        print(f"Steps: {steps}, guidance_scale: {guidance_scale}, size: {width}x{height}, seed: {seed}")
        generator = torch.Generator(device=device).manual_seed(seed)
        image = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
        ).images[0]
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output)
        print(f"Image written: {output}")
        if np is not None:
            arr = np.asarray(image)
            print(f"Image stats: min={arr.min()}, max={arr.max()}, mean={arr.mean():.2f}")

    return True


def main() -> int:
    default_model_dir = Path("/home/qianustb/EnAR/pre_model/DDIM/stable-diffusion-v1-5")
    parser = argparse.ArgumentParser(description="Test local Stable Diffusion v1.5 DDIM model loading.")
    parser.add_argument("--model_dir", type=Path, default=default_model_dir)
    parser.add_argument("--generate", action="store_true", help="Run a tiny image generation smoke test.")
    parser.add_argument(
        "--prompt",
        default="a photo of only a hand with six fingers, must six fingers and only one hand.",
    )
    parser.add_argument(
        "--negative_prompt",
        default="black image, blank image, low quality, blurry, distorted",
    )
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--disable_safety_checker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable safety checker during local smoke-test generation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/qianustb/EnAR/work_scripts/test_DDIM/outputs/sd_v1_5_hand_test.png"),
    )
    args = parser.parse_args()

    ok = check_files(args.model_dir)
    check_model_index(args.model_dir)
    if not ok:
        return 1
    return 0 if load_pipeline(
        args.model_dir,
        args.generate,
        args.output,
        args.prompt,
        args.negative_prompt,
        args.steps,
        args.guidance_scale,
        args.height,
        args.width,
        args.seed,
        args.disable_safety_checker,
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
