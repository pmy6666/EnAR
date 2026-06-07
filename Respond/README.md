# EnAR Respond Stage

Respond 阶段基于原图、问题和 Attend 输出的反事实 token indices，运行 LLaVA-1.5 regular decoding 与 EnAR contrastive decoding，并保存最终回答与逐步解码日志。EnAR 这里复用 VCD 的 logits 公式，但 contrastive input 是 token-level padded visual input `v_pad`，不是 VCD 原始代码里的 Gaussian noisy image `v'`。

## 目录结构

```text
Respond/
  config.py                 # RespondConfig 与 YAML 字段校验
  prompt_builder.py         # LLaVA-1.5 对话 prompt 构造
  model_loader.py           # LLaVA processor/model/tokenizer 加载
  input_encoder.py          # 图像和 prompt 编码
  visual_embeddings.py      # projected visual embeddings 提取与 Attend index 对齐
  padded_visual_builder.py  # v' padded visual embeddings 构造
  embedding_merge.py        # inputs_embeds 级别 image placeholder 替换
  dual_branch_forwarder.py  # 原始分支与 padded 分支 forward
  logits_processor.py       # Eq.7 contrastive logits 与可选 APC
  token_selector.py         # greedy / sampling next-token 选择
  generation_loop.py        # 无 KV cache 的对比生成循环
  regular_generation.py     # LLaVA 原生 generate baseline
  output_writer.py          # JSON、trace、答案文本输出
  pipeline.py               # RespondPipeline 主流程
  cli.py                    # 命令行入口
  respond_config.yaml       # 示例配置
  tests/                    # 模块单元测试
```

## 依赖

统一使用项目环境 `EnAR/env/`。如果需要补装：

```bash
/home/qianustb/EnAR/env/bin/python -m pip install -U \
  torch torchvision transformers accelerate safetensors sentencepiece protobuf \
  pillow numpy PyYAML pytest
```

只运行单元测试通常需要：

```bash
/home/qianustb/EnAR/env/bin/python -m pip install -U torch pillow PyYAML pytest
```

## 配置

示例配置在 `Respond/respond_config.yaml`：

```yaml
paths:
  llava_model_dir: EnAR/pre_model/LLM/llava-1.5-7b-hf
  image_path: EnAR/outputs/envision/run_001/reconstruction_no_perturb.png
  attend_result_json: EnAR/outputs/attend/run_001/attend_result.json
  output_dir: EnAR/outputs/respond/run_001
generation:
  question: What is shown in the image?
  max_new_tokens: 64
  do_sample: true
  temperature: 1.0
  top_p: 1.0
  seed: 42
contrastive:
  alpha: 1.0
  use_apc: true
  apc_beta: 0.1
  padding_strategy: pad_token_embedding
model:
  device: auto
  dtype: float16
  vision_feature_select_strategy: default
```

`vision_feature_select_strategy: default` 会使用 576 个 patch embeddings，与当前 LLaVA-1.5 prompt 中 `<image>` 展开的 token 数匹配。若改成 `full`，需要同步 processor 的 image token 设置，否则会触发 placeholder 数量校验。

论文式 VCD 复现配置使用 `do_sample: true, temperature: 1.0, top_p: 1.0, alpha: 1.0, use_apc: true, apc_beta: 0.1`。快速调试可以改为 `do_sample: false`，此时输出 meta 中会标为 `decode_mode: greedy_debug`。

## 运行

在 `/home/qianustb/EnAR` 下运行：

```bash
PYTHONPATH=/home/qianustb/EnAR \
/home/qianustb/EnAR/env/bin/python -m Respond.cli \
  --config /home/qianustb/EnAR/Respond/respond_config.yaml
```

也可以用命令行覆盖关键参数：

```bash
PYTHONPATH=/home/qianustb/EnAR \
/home/qianustb/EnAR/env/bin/python -m Respond.cli \
  --config /home/qianustb/EnAR/Respond/respond_config.yaml \
  --question "What animal is in the image?" \
  --alpha 0.8 \
  --max_new_tokens 32 \
  --padding_strategy zero_embedding
```

## 输出

默认输出到 `paths.output_dir`：

```text
resolved_config.yaml
respond_result.json
decode_trace.json
answer_regular.txt
answer_enar.txt
```

`respond_result.json` 包含输入路径、问题、Attend token indices、padding 策略、regular 答案、EnAR 答案、prompt 长度、visual token layout 和运行参数。`decode_trace.json` 每步记录 original branch、padded contrastive branch、VCD logits 和 APC 后 final logits 的 top tokens，以及 selected token 在四路 logits/probs 下的数值。

## 测试

单元测试不加载 LLaVA-1.5-7B：

```bash
cd /home/qianustb/EnAR
PYTHONPATH=/home/qianustb/EnAR \
/home/qianustb/EnAR/env/bin/python -m pytest Respond/tests
```

当前测试覆盖配置解析、prompt 格式、Attend token index 对齐、padding 策略、contrastive logits、APC、token 选择、inputs_embeds 替换和输出写入。

## 实现说明

- Regular baseline 使用 LLaVA 原生 `model.generate()`。
- EnAR decoding 使用自定义无 KV cache loop，每步分别 forward 原始 visual embeddings 与 padded visual embeddings；首版为了可读性和正确性暂不维护双分支 KV cache，生成耗时通常超过 regular decoding 的两倍。
- logits 合并公式为 `(1 + alpha) * logits_original - alpha * logits_contrastive_input`，其中 `logits_contrastive_input == logit_theta(y | x, v_pad, y_<t)`。
- APC 使用 VCD 的 original-logits cutoff：`logits_original >= log(beta) + max(logits_original)`；`beta = 0` 时保留全部 token。
- `pad_token_embedding` 会优先使用 LLM embedding table 的 pad token；如果 hidden size 或 pad id 不兼容，会自动退化为 `zero_embedding`，并在 `padding_meta` 记录 requested/actual strategy、fallback reason、pad token id 来源和 ignored indices。
- `alpha = 0` 时，对比 logits 等价于原始分支 logits，适合做调试基线。
