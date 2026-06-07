# LLaVA VCD Single-Image Inference

This directory contains a small HF Transformers implementation of VCD decoding for one LLaVA-1.5 image-question example.

It follows the plan in `docs/llava_vcd_inference_plan.md` and keeps all new code under `EnAR/VCD/my_work`.

## Files

- `run_llava_vcd_single.py`: loads local HF `LlavaForConditionalGeneration`, creates an origin and noisy image branch, runs token-by-token VCD decoding, and writes logits.
- `run_llava_vcd_single.sh`: default command for the wolf image test case.
- `outputs/`: default output directory, created automatically when the script runs.

## Default Input

- Image: `/home/qianustb/EnAR/Envision/image/data/wolf_5.png`
- Question: `How many legs does this animal have?`
- LLaVA model: `/home/qianustb/EnAR/pre_model/LLM/llava-1.5-7b-hf`

## Run

From the VCD repository root:

```bash
cd /home/qianustb/EnAR/VCD
bash my_work/run_llava_vcd_single.sh
```

Or run the Python entry directly:

```bash
cd /home/qianustb/EnAR/VCD
PYTHONPATH=/home/qianustb/EnAR/VCD:/home/qianustb/EnAR/VCD/experiments \
/home/qianustb/EnAR/env/bin/python my_work/run_llava_vcd_single.py \
  --model_dir /home/qianustb/EnAR/pre_model/LLM/llava-1.5-7b-hf \
  --image /home/qianustb/EnAR/Envision/image/data/wolf_5.png \
  --question "How many legs does this animal have?" \
  --noise_step 500 \
  --cd_alpha 1.0 \
  --cd_beta 0.1 \
  --max_new_tokens 32 \
  --log_first_n_tokens 20 \
  --output_dir /home/qianustb/EnAR/VCD/my_work/outputs
```

## Outputs

The default run writes:

- `outputs/wolf_5_llava_vcd_summary.json`
- `outputs/wolf_5_llava_vcd_logits.pt`

The JSON file contains the answer, generated token ids/tokens, and top logits for the origin, distorted, raw VCD, and APC-filtered VCD branches for the first logged steps.

The `.pt` file contains full tensors:

- `origin_logits`
- `distorted_logits`
- `vcd_raw_logits`
- `vcd_logits`
- `generated_token_ids`
- `prompt_input_ids`
- `metadata`

`vcd_raw_logits` are computed as:

```python
(1 + cd_alpha) * origin_logits - cd_alpha * distorted_logits
```

`vcd_logits` are the same logits after the Adaptive Plausibility Constraints cutoff controlled by `cd_beta`.

## Useful Options

- `--do_sample`: use multinomial sampling instead of greedy decoding.
- `--temperature 1.0`: sampling temperature.
- `--top_p 1.0`: nucleus sampling cutoff.
- `--top_k 50`: optional top-k sampling cutoff.
- `--noise_step 500`: diffusion noise step, valid range is `0..999`.
- `--log_first_n_tokens 20`: number of generated steps for which full logits are stored.
- `--vision_feature_select_strategy full --num_additional_image_tokens 1`: compatibility defaults for the local HF LLaVA weights.

## Validation

Quick syntax checks:

```bash
python3 -m py_compile my_work/run_llava_vcd_single.py
bash -n my_work/run_llava_vcd_single.sh
```

After running inference, inspect the summary:

```bash
/home/qianustb/EnAR/env/bin/python - <<'PY'
import json
from pathlib import Path

path = Path('/home/qianustb/EnAR/VCD/my_work/outputs/wolf_5_llava_vcd_summary.json')
data = json.loads(path.read_text(encoding='utf-8'))
print(data['answer'])
print(len(data['logged_steps']))
PY
```
