#!/usr/bin/env bash
set -euo pipefail

VCD_ROOT="/home/qianustb/EnAR/VCD"
PYTHON_BIN="${PYTHON_BIN:-/home/qianustb/EnAR/env/bin/python}"

cd "${VCD_ROOT}"
PYTHONPATH="${VCD_ROOT}:${VCD_ROOT}/experiments:${PYTHONPATH:-}" \
"${PYTHON_BIN}" my_work/run_llava_vcd_single.py \
  --model_dir /home/qianustb/EnAR/pre_model/LLM/llava-1.5-7b-hf \
  --image /home/qianustb/EnAR/Envision/image/data/wolf_5.png \
  --question "How many legs does this animal have?" \
  --noise_step 500 \
  --cd_alpha 1.0 \
  --cd_beta 0.1 \
  --max_new_tokens 32 \
  --log_first_n_tokens 20 \
  --top_k_logit_dump 20 \
  --output_dir /home/qianustb/EnAR/VCD/my_work/outputs
