# LLaVA VCD Inference Test Plan

## 1. Goal

在 `EnAR/VCD/my_work` 下规划一个只使用 LLaVA 的 VCD 推理测试流程，输入固定为：

- `input_image`: `/home/qianustb/EnAR/Envision/image/data/wolf_5.png`
- `question`: `How many legs does this animal have?`
- 模型权重目录：`/home/qianustb/EnAR/pre_model/LLM/llava-1.5-7b-hf`

本阶段只生成计划文档，不新增推理代码。

## 2. Current Code Findings

### VCD 代码结构

- `EnAR/VCD/vcd_utils/vcd_add_noise.py`
  - 提供 `add_diffusion_noise(image_tensor, noise_step)`。
  - 该 distorted image 生成是纯 PyTorch 前向加噪，不依赖 Stable Diffusion 或 DDIM 权重。

- `EnAR/VCD/vcd_utils/vcd_sample.py`
  - monkey patch `transformers.generation.utils.GenerationMixin.sample/_sample`。
  - 在每个 decoding step 中分别计算：
    - `next_token_logits`: origin image logits
    - `next_token_logits_cd`: distorted image logits
    - `cd_logits = (1 + cd_alpha) * origin_logits - cd_alpha * distorted_logits`
  - 使用 `cd_beta` 做 Adaptive Plausibility Constraints cutoff。

- `EnAR/VCD/experiments/eval/object_hallucination_vqa_llava.py`
  - 原始 VCD LLaVA eval 入口。
  - 依赖 `experiments/llava/model/builder.py` 和原版 LLaVA 类 `LlavaLlamaForCausalLM`。
  - 已经有 `images_cd`、`cd_alpha`、`cd_beta`、`noise_step` 参数。

### 本地模型权重情况

`/home/qianustb/EnAR/pre_model/LLM/llava-1.5-7b-hf` 已存在，且是 Hugging Face Transformers 格式：

- `config.json` 的 architecture 是 `LlavaForConditionalGeneration`
- 有 `model-00001-of-00003.safetensors` 到 `model-00003-of-00003.safetensors`
- 有 tokenizer、processor、generation config

这与 VCD 原始 eval 脚本使用的原版 LLaVA `LlavaLlamaForCausalLM` 路线不同。因此推荐后续新增脚本走 HF Transformers 路线，直接复用现有权重，避免再下载或转换原版 LLaVA 权重。

## 3. Download / Parameter Requirements

### 必需权重

当前 LLaVA HF 权重已具备，后续推理测试不需要额外训练参数。

必需文件包括：

- `config.json`
- `generation_config.json`
- `preprocessor_config.json`
- `tokenizer.model`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `model.safetensors.index.json`
- `model-00001-of-00003.safetensors`
- `model-00002-of-00003.safetensors`
- `model-00003-of-00003.safetensors`

### 如需重新下载 LLaVA HF 权重

项目已有下载脚本：

```bash
cd /home/qianustb/EnAR
/home/qianustb/EnAR/work_scripts/download_llava_1_5_7b_modelscope.sh
```

参数文件：

```bash
/home/qianustb/EnAR/pre_model/LLM/llava_1_5_7b_modelscope.params.env
```

当前参数指向：

```bash
MODELSCOPE_MODEL_ID="swift/llava-1.5-7b-hf"
MODEL_LOCAL_DIR="llava-1.5-7b-hf"
```

如果环境缺少依赖，先安装：

```bash
/home/qianustb/EnAR/env/bin/python -m pip install -U modelscope requests
```

### VCD distorted image 是否需要 DDIM / SD 训练参数

按当前 `vcd_add_noise.py` 实现，不需要 DDIM 或 Stable Diffusion 训练参数。`noise_step` 只是一个 0-999 区间内的加噪步数，默认可沿用 VCD 脚本的 `500`。

如果未来改成论文里更复杂的 diffusion distortion 或 DDIM inversion，再考虑复用：

```bash
/home/qianustb/EnAR/work_scripts/download_sd_v1_5_ddim.sh
```

但本次 LLaVA VCD 推理测试不需要。

## 4. Recommended Implementation Plan

### 4.1 新增脚本位置

后续代码阶段建议新增：

```text
EnAR/VCD/my_work/run_llava_vcd_single.py
EnAR/VCD/my_work/run_llava_vcd_single.sh
EnAR/VCD/my_work/outputs/
```

其中：

- `.py` 负责模型加载、origin/distorted 前向、VCD decoding、logits 记录。
- `.sh` 封装默认路径和参数，方便一条命令运行。

### 4.2 使用 HF Transformers 路线

建议直接使用：

```python
from transformers import AutoProcessor, LlavaForConditionalGeneration
```

原因：

- 本地权重就是 HF `LlavaForConditionalGeneration` 格式。
- 可直接加载 safetensors shard。
- 无需改动 `experiments/llava/model/language_model/llava_llama.py`。
- 方便手写 decoding loop 并记录 origin/distorted logits。

### 4.3 Prompt 构造

保持 LLaVA 1.5 HF 常见格式：

```text
USER: <image>
How many legs does this animal have?
ASSISTANT:
```

可参考已有测试脚本：

```text
/home/qianustb/EnAR/work_scripts/test_LLaVA/test_load_llava_1_5_7b.py
```

该脚本已有 processor patch 逻辑：

- `processor.patch_size = 14`
- `processor.num_additional_image_tokens = 1`
- `processor.vision_feature_select_strategy = "full"`
- `processor.image_token = tokenizer.convert_ids_to_tokens(image_token_index)`

后续实现应复用这段兼容逻辑，避免 575/576 image token mismatch。

### 4.4 Distorted Image 构造

加载 origin image 后：

1. 用 `AutoProcessor` 得到 `pixel_values`
2. 对 `pixel_values[0]` 调用：

```python
from vcd_utils.vcd_add_noise import add_diffusion_noise

distorted_pixel_values = add_diffusion_noise(origin_pixel_values[0].cpu(), noise_step)
```

3. 放回 batch 维度并移动到模型 device/dtype。

推荐默认参数：

```text
noise_step = 500
cd_alpha = 1.0
cd_beta = 0.1
max_new_tokens = 32
log_first_n_tokens = 20
temperature = 1.0
top_p = 1.0
top_k = None
seed = 42
```

### 4.5 手写 VCD Decoding Loop

因为需要同时输出 origin 和 distorted 的前 20 token logits，建议不要只调用 `model.generate()`。更稳妥的方案是手写逐 token decoding：

每一步：

1. 使用当前 `input_ids` + origin `pixel_values` 前向，得到 `origin_logits = outputs.logits[:, -1, :]`
2. 使用当前 `input_ids` + distorted `pixel_values` 前向，得到 `distorted_logits = outputs_cd.logits[:, -1, :]`
3. 计算：

```python
cutoff = log(cd_beta) + origin_logits.max(dim=-1, keepdim=True).values
vcd_logits = (1 + cd_alpha) * origin_logits - cd_alpha * distorted_logits
vcd_logits = vcd_logits.masked_fill(origin_logits < cutoff, -inf)
```

4. 根据 decoding 参数选择下一个 token：
   - 可复现实验优先：`do_sample=False` 时取 `argmax(vcd_logits)`
   - 与 VCD 原始脚本一致：`do_sample=True` 时 multinomial sampling
5. 将 next token append 到 `input_ids`
6. 若 step < 20，保存 origin/distorted logits

注意：HF LLaVA 每次完整前向会重复 vision encoding，单张图测试可接受。后续如需加速，再考虑 cache/past_key_values。

## 5. Logits Output Design

完整 vocab logits 大约是：

```text
20 steps * 2 branches * 32064 vocab
```

建议同时输出两类文件：

### 5.1 Full logits tensor

保存完整 logits，便于后续分析：

```text
EnAR/VCD/my_work/outputs/wolf_5_llava_vcd_logits.pt
```

结构建议：

```python
{
    "origin_logits": FloatTensor[num_steps_logged, vocab_size],
    "distorted_logits": FloatTensor[num_steps_logged, vocab_size],
    "vcd_logits": FloatTensor[num_steps_logged, vocab_size],
    "generated_token_ids": LongTensor[num_generated_tokens],
    "prompt_input_ids": LongTensor[prompt_len],
    "metadata": {...}
}
```

### 5.2 Human-readable summary

保存 JSON，包含每一步生成 token 以及 top-k logits，避免文本文件过大：

```text
EnAR/VCD/my_work/outputs/wolf_5_llava_vcd_summary.json
```

结构建议：

```json
{
  "image": "/home/qianustb/EnAR/Envision/image/data/wolf_5.png",
  "question": "How many legs does this animal have?",
  "answer": "...",
  "model_dir": "/home/qianustb/EnAR/pre_model/LLM/llava-1.5-7b-hf",
  "params": {
    "noise_step": 500,
    "cd_alpha": 1.0,
    "cd_beta": 0.1,
    "seed": 42
  },
  "logged_steps": [
    {
      "step": 0,
      "generated_token_id": 123,
      "generated_token": "...",
      "origin_top_logits": [
        {"token_id": 1, "token": "...", "logit": 12.3}
      ],
      "distorted_top_logits": [
        {"token_id": 2, "token": "...", "logit": 11.8}
      ],
      "vcd_top_logits": [
        {"token_id": 3, "token": "...", "logit": 13.1}
      ]
    }
  ]
}
```

建议 `top_k_logit_dump = 20`，即每个 step 对 origin/distorted/vcd 各保存 top 20 的可读摘要；完整 logits 仍保存在 `.pt` 中。

## 6. Proposed Commands For Code Stage

后续实现完成后的默认运行命令规划为：

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

封装脚本运行：

```bash
cd /home/qianustb/EnAR/VCD
bash my_work/run_llava_vcd_single.sh
```

## 7. Validation Plan

代码阶段完成后建议依次验证：

1. 权重文件检查

```bash
/home/qianustb/EnAR/work_scripts/test_LLaVA/run_test_llava_1_5_7b.sh \
  --model_dir /home/qianustb/EnAR/pre_model/LLM/llava-1.5-7b-hf \
  --check_only
```

2. HF LLaVA 普通推理 smoke test

```bash
/home/qianustb/EnAR/work_scripts/test_LLaVA/run_test_llava_1_5_7b.sh \
  --model_dir /home/qianustb/EnAR/pre_model/LLM/llava-1.5-7b-hf \
  --image /home/qianustb/EnAR/Envision/image/data/wolf_5.png \
  --prompt "How many legs does this animal have?" \
  --max_new_tokens 32
```

3. VCD 推理输出检查

确认生成：

```text
EnAR/VCD/my_work/outputs/wolf_5_llava_vcd_logits.pt
EnAR/VCD/my_work/outputs/wolf_5_llava_vcd_summary.json
```

检查内容：

- `origin_logits.shape[0] <= 20`
- `distorted_logits.shape[0] <= 20`
- `origin_logits.shape[-1] == distorted_logits.shape[-1]`
- JSON 中包含 answer、generated tokens、origin/distorted top logits

## 8. Open Decisions For Code Stage

默认建议：

- 使用 `do_sample=False` 做可复现单例测试。
- 同时保留 `--do_sample` 参数，以便复刻 VCD 原始 sampling。
- 完整 logits 用 `.pt` 保存，JSON 只放 top-k 摘要。
- 不改动 VCD 原仓库已有 eval 脚本，新增代码全部放在 `EnAR/VCD/my_work`。

如后续必须严格复用原始 `vcd_sample.py` monkey patch，则需要额外适配 HF `LlavaForConditionalGeneration.prepare_inputs_for_generation_cd`，风险和改动面都比手写 decoding loop 更大，不作为本次推荐路线。
