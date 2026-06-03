#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


REQUIRED_FILES = [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "model.safetensors.index.json",
]


def check_model_files(model_dir: Path) -> bool:
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

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        data = json.loads(index_path.read_text())
        shards = sorted(set(data.get("weight_map", {}).values()))
        print(f"\nWeight shards referenced by index: {len(shards)}")
        for rel_path in shards:
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
        print("\nERROR: required model files are missing.", file=sys.stderr)
        return False
    return True


def build_prompt(question: str) -> str:
    return f"USER: <image>\n{question}\nASSISTANT:"


def patch_processor_from_config(
    processor,
    model_dir: Path,
    vision_feature_select_strategy: str,
    num_additional_image_tokens: int,
) -> None:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        return

    config = json.loads(config_path.read_text())
    vision_config = config.get("vision_config", {})
    patch_size = vision_config.get("patch_size")
    image_token_index = config.get("image_token_index")

    if getattr(processor, "patch_size", None) is None and patch_size is not None:
        processor.patch_size = patch_size
        print(f"Patched processor.patch_size={patch_size}")

    processor.num_additional_image_tokens = num_additional_image_tokens
    print(f"Patched processor.num_additional_image_tokens={num_additional_image_tokens}")

    if (
        getattr(processor, "vision_feature_select_strategy", None) is None
        and vision_feature_select_strategy
    ):
        processor.vision_feature_select_strategy = vision_feature_select_strategy
        print(f"Patched processor.vision_feature_select_strategy={vision_feature_select_strategy}")

    if getattr(processor, "image_token", None) is None and image_token_index is not None:
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None:
            image_token = tokenizer.convert_ids_to_tokens(image_token_index)
            if image_token:
                processor.image_token = image_token
                print(f"Patched processor.image_token={image_token}")


def run_inference(
    model_dir: Path,
    image_path: Path,
    question: str,
    max_new_tokens: int,
    vision_feature_select_strategy: str,
    num_additional_image_tokens: int,
) -> bool:
    if not image_path.is_file():
        print(f"ERROR: image does not exist: {image_path}", file=sys.stderr)
        return False

    try:
        import torch
        from PIL import Image
        from transformers import AutoProcessor, LlavaForConditionalGeneration
    except Exception as exc:
        print(
            "\nERROR: missing runtime dependencies.\n"
            f"Import error: {exc}\n"
            "Install with:\n"
            "  /home/qianustb/EnAR/env/bin/python -m pip install -U transformers pillow accelerate safetensors sentencepiece protobuf",
            file=sys.stderr,
        )
        return False

    if not torch.cuda.is_available():
        print(
            "\nWARNING: CUDA is not available. LLaVA-1.5-7B CPU inference can be very slow and memory-heavy.",
            file=sys.stderr,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"\nLoading LLaVA on {device} with dtype={dtype} ...")

    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
    patch_processor_from_config(
        processor,
        model_dir,
        vision_feature_select_strategy,
        num_additional_image_tokens,
    )
    model = LlavaForConditionalGeneration.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    image = Image.open(image_path).convert("RGB")
    prompt = build_prompt(question)
    print(f"\nImage: {image_path}")
    print(f"Prompt: {question}")

    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            vision_feature_select_strategy=vision_feature_select_strategy,
        )

    prompt_len = inputs["input_ids"].shape[-1]
    answer_ids = output_ids[0][prompt_len:]
    answer = processor.decode(answer_ids, skip_special_tokens=True).strip()
    print(f"\nAnswer: {answer}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Test local LLaVA-1.5-7B image question answering.")
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=Path("/home/qianustb/EnAR/pre_model/LLM/llava-1.5-7b-hf"),
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("/home/qianustb/EnAR/work_scripts/test_DDIM/outputs/sd_v1_5_dog_test.png"),
    )
    parser.add_argument("--prompt", default="这个图片里面是什么？")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument(
        "--vision_feature_select_strategy",
        choices=["full", "default"],
        default="full",
        help="Use full for current transformers compatibility; default may produce 575/576 token mismatch.",
    )
    parser.add_argument(
        "--num_additional_image_tokens",
        type=int,
        default=1,
        help="Use 1 so the image token count includes the CLIP CLS token in current transformers.",
    )
    parser.add_argument("--check_only", action="store_true", help="Only check local files; do not load the model.")
    args = parser.parse_args()

    if not check_model_files(args.model_dir):
        return 1
    if args.check_only:
        return 0
    return 0 if run_inference(
        args.model_dir,
        args.image,
        args.prompt,
        args.max_new_tokens,
        args.vision_feature_select_strategy,
        args.num_additional_image_tokens,
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
