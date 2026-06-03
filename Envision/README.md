# Envision 阶段

本目录实现 EnAR 第 4.1 阶段：将输入图像转换为扩散模型视觉先验下的视觉印象，并生成像素级不确定性图。

## 模块说明

- `config.py`：读取分组式 YAML 配置，并生成统一的 `EnvisionConfig` 对象。
- `preprocessor.py`：读取 RGB 图像，执行中心裁剪、resize，并归一化到 `[-1, 1]`。
- `model_loader.py`：从本地加载 Stable Diffusion v1.5，并显式替换为 `DDIMScheduler`。
- `prompt_conditioner.py`：生成空 prompt 或固定 prompt 的 text embedding。
- `latent_codec.py`：使用 VAE 完成图像 latent 编码和 latent 解码，并处理 SD latent scaling。
- `ddim_inverter.py`：执行确定性 DDIM inversion，从 `z0` 推到指定扰动步 `zT`。
- `gradient_estimator.py`：根据 UNet 噪声预测估计 Tweedie 风格的 latent score direction。
- `langevin.py`：执行退火 Langevin latent 扰动。
- `ddim_reconstructor.py`：执行 DDIM reverse，并通过 VAE 解码回图像。
- `uncertainty.py`：根据多张视觉印象计算像素方差、不确定性灰度图和热力图。
- `representative.py`：从 K 个 visual impressions 中选择与原始输入图像 L2 距离最大的样本作为代表性视觉印象。
- `output_writer.py`：保存图片、`.npy` 不确定性图和 `metadata.json`。
- `pipeline.py`：串联完整单图 Envision 流程。
- `cli.py`：命令行入口。
- `tests/`：每个模块的轻量测试代码，使用 fake 组件，不加载 SD 大模型。

## 依赖安装

当前 `EnAR/env/` 环境里已经包含大部分运行依赖。如果需要安装或刷新依赖，使用：

```bash
/home/qianustb/EnAR/env/bin/python -m pip install -U \
  torch torchvision diffusers transformers accelerate safetensors \
  pillow numpy pyyaml pytest
```

Stable Diffusion v1.5 本地模型默认路径：

```text
EnAR/pre_model/DDIM/stable-diffusion-v1-5/
```

## YAML 配置

默认配置文件：

```text
EnAR/Envision/envision_config.yaml
```

运行前至少需要修改：

```yaml
paths:
  input_image: /path/to/input.png
  output_dir: /home/qianustb/EnAR/outputs/envision/run_001
```

完整配置结构如下：

```yaml
paths:
  sd_model_dir: /home/qianustb/EnAR/pre_model/DDIM/stable-diffusion-v1-5
  input_image: /path/to/input.png
  output_dir: /home/qianustb/EnAR/outputs/envision/run_001

image:
  image_size: 512

ddim:
  num_ddim_steps: 50
  inversion_step_T: 30
  guidance_scale: 1.0

langevin:
  langevin_steps_M: 10
  sample_count_K: 4
  eta_start: 1.0e-2
  eta_end: 1.0e-4
  temperature_tau: 1.0

prompt:
  prompt: ""
  negative_prompt: ""

runtime:
  seed: 42
  dtype: float16
  device: null
  debug: false
```

说明：

- `inversion_step_T` 表示 DDIM 推理 schedule 中的第几个 step，不是训练时间步 `30`。
- CUDA 下建议使用 `dtype: float16`。
- CPU 下请使用 `dtype: float32`。
- `sample_count_K` 越大，不确定性估计越稳定，但显存和耗时也会增加。

## 运行方式

从 `/home/qianustb` 执行：

```bash
PYTHONPATH=/home/qianustb/EnAR \
/home/qianustb/EnAR/env/bin/python -m Envision.cli \
  --config /home/qianustb/EnAR/Envision/envision_config.yaml
```

命令行参数可以覆盖 YAML 中的配置。比如开发期低成本 smoke run：

```bash
PYTHONPATH=/home/qianustb/EnAR \
/home/qianustb/EnAR/env/bin/python -m Envision.cli \
  --config /home/qianustb/EnAR/Envision/envision_config.yaml \
  --output_dir /home/qianustb/EnAR/outputs/envision/smoke \
  --num_ddim_steps 10 \
  --inversion_step_T 5 \
  --langevin_steps_M 2 \
  --sample_count_K 2 \
  --dtype float16
```

## 输出结果

运行完成后，输出目录结构如下：

```text
output_dir/
  original.png
  preprocessed.png
  impression.png
  difference.png
  reconstruction_no_perturb.png
  uncertainty_gray.png
  uncertainty_heatmap.png
  uncertainty_map.npy
  samples/
    sample_000.png
    sample_001.png
  metadata.json
```

其中：

- `original.png`：原始输入图像。
- `preprocessed.png`：进入 SD 前的 512 尺寸预处理图像。
- `impression.png`：代表性视觉印象。
- `difference.png`：预处理图像与代表性视觉印象的差异图。
- `reconstruction_no_perturb.png`：不加 Langevin 扰动的 DDIM 重建图，用于 sanity check。
- `uncertainty_map.npy`：float 类型像素级不确定性图。
- `uncertainty_gray.png`：灰度不确定性图。
- `uncertainty_heatmap.png`：伪彩色不确定性热力图。
- `samples/`：所有采样视觉印象。
- `metadata.json`：记录配置、尺寸变换、扰动 timestep、代表样本 index、L2 差异分数、Langevin 调试 norm 和输出路径。

Attend 阶段建议直接读取 `metadata.json` 中的：

- `outputs.original_image`
- `outputs.preprocessed_image`
- `outputs.impression_image`
- `outputs.uncertainty_map`
- `outputs.uncertainty_heatmap`
- `transform_meta`

## 测试

运行轻量模块测试：

```bash
PYTHONPATH=/home/qianustb/EnAR \
/home/qianustb/EnAR/env/bin/python -m pytest /home/qianustb/EnAR/Envision/tests
```

这些测试不会加载 Stable Diffusion 大模型。真实模型验证请使用 CLI 对单张图片运行。

## 调试建议

- 如果 `impression.png` 整体被重绘，优先降低 `eta_start` 或 `temperature_tau`，并检查 `reconstruction_no_perturb.png`。
- 如果输出几乎不变化，可以增大 `eta_start` 或 `langevin_steps_M`。
- 如果 `uncertainty_heatmap.png` 几乎全黑，通常说明 K 个样本太接近，可以增大 `sample_count_K`、`langevin_steps_M` 或 `eta_start`。
- 如果不确定性图全亮或图像漂移严重，降低 `temperature_tau` 或缩短 Langevin 扰动步数。
