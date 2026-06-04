# 4.1 Envision 阶段模块化实现设计

## 1. 阶段目标

Envision 阶段负责把输入图像 `V` 转换为扩散模型视觉先验下的视觉印象 `V_hat`，并生成像素级不确定性图 `U`。该阶段对应论文 Algorithm 1，也是当前复现任务中最优先落地的部分。

阶段输入：

- 原始图像路径。
- Stable Diffusion v1.5 本地模型路径：`EnAR/pre_model/DDIM/stable-diffusion-v1-5/`。
- DDIM 与 Langevin 超参数。
- 可选 prompt 或 caption。

阶段输出：

- 代表性视觉印象 `impression.png`。
- 不确定性热力图 `uncertainty_heatmap.png`。
- 多个采样视觉印象 `samples/*.png`。
- 原图与视觉印象差异图 `difference.png`。
- 运行元数据 `metadata.json`。

## 2. 总体流程

```text
输入图像
  -> 图像预处理
  -> 加载 SD v1.5 与 DDIM scheduler
  -> VAE 编码得到 z0
  -> DDIM forward/inversion 得到 zT
  -> K 次 Langevin latent perturbation
  -> K 次 DDIM reverse + VAE decode
  -> 选择代表性视觉印象
  -> 计算不确定性图
  -> 保存可视化与元数据
```

## 3. 模块设计

### 3.1 配置模块 `EnvisionConfig`

职责：

- 集中管理 Envision 阶段所有路径、模型、采样和输出参数。
- 为后续命令行参数、配置文件和实验记录提供统一入口。

建议字段：

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `sd_model_dir` | path | `EnAR/pre_model/DDIM/stable-diffusion-v1-5` | 本地 SD v1.5 模型 |
| `input_image` | path | 必填 | 原始图像 |
| `output_dir` | path | 必填 | 输出目录 |
| `image_size` | int | 512 | SD 输入尺寸 |
| `num_ddim_steps` | int | 50 | 论文 `Tmax` |
| `inversion_step_T` | int | 30 | 论文扰动时间步 |
| `langevin_steps_M` | int | 10 | Langevin 扰动步数 |
| `sample_count_K` | int | 4 | 开发期先用 4，正式可增大 |
| `eta_start` | float | `1e-2` | 初始步长 |
| `eta_end` | float | `1e-4` | 终止步长 |
| `temperature_tau` | float | 0.1 | 随机扰动温度 |
| `prompt` | str | `""` | 初版可为空 |
| `negative_prompt` | str | `""` | 初版可为空 |
| `seed` | int | 42 | 随机种子 |
| `dtype` | str | `float16` | CUDA 下使用 fp16 |

输入：

- 命令行参数或 JSON/YAML 配置。

输出：

- 规范化后的配置对象。

### 3.2 图像预处理模块 `ImagePreprocessor`

职责：

- 读取输入图像并转换为 RGB。
- 将图像 resize/crop 到 SD v1.5 的工作尺寸。
- 保存原始尺寸和 resize 变换信息，便于后续结果回映射。

输入：

- `input_image`。
- `image_size`。

输出：

- `image_pil`: 预处理后的 PIL 图像。
- `image_tensor`: 归一化到 SD VAE 输入范围的 tensor。
- `transform_meta`: 原始尺寸、目标尺寸、crop/resize 方式。

实现要点：

- 初版建议采用 center crop + resize，保证复现简单。
- 输出图像范围需要与 diffusers VAE 约定一致，通常为 `[-1, 1]`。
- 后续如果接入 Attend，需要记录 SD 的 512 尺寸与 LLaVA 的 336 尺寸之间的映射关系。

### 3.3 模型加载模块 `StableDiffusionLoader`

职责：

- 从本地加载 SD v1.5 的 VAE、UNet、tokenizer、text encoder。
- 显式构造 `DDIMScheduler`，替换模型目录默认的 `PNDMScheduler`。
- 设置 dtype、device、eval 模式。

输入：

- `sd_model_dir`。
- `dtype`。
- `device`。

输出：

- `vae`。
- `unet`。
- `tokenizer`。
- `text_encoder`。
- `ddim_scheduler`。

实现要点：

- 当前本地 `model_index.json` 中默认 scheduler 是 `PNDMScheduler`，正式实现必须显式替换为 `DDIMScheduler`。
- 建议先复用 `StableDiffusionPipeline.from_pretrained(..., local_files_only=True)` 加载组件，再替换 scheduler。
- 加载后关闭梯度：`requires_grad_(False)`，全流程不训练。

### 3.4 文本条件模块 `PromptConditioner`

职责：

- 生成扩散模型 UNet 所需的 text condition embedding。
- 支持空 prompt、固定 prompt、外部 caption 三种模式。

输入：

- `prompt`。
- `negative_prompt`。
- `tokenizer`。
- `text_encoder`。

输出：

- `text_embeddings`。
- 可选 `negative_text_embeddings`。

实现策略：

- 第一版使用空 prompt，减少额外变量。
- 如果空 prompt 使视觉印象过于不稳定，可改成通用 prompt，例如 `a realistic photo`。
- 如果后续引入 LLaVA caption，应把 caption 生成结果写入 `metadata.json`，避免实验不可追踪。

### 3.5 VAE 编解码模块 `LatentCodec`

职责：

- 把预处理图像编码为 latent `z0`。
- 把 DDIM reverse 后的 latent 解码为图像。

输入：

- `image_tensor`。
- `vae`。

输出：

- `z0`。
- decoded PIL image 或 tensor image。

实现要点：

- SD v1.5 VAE latent 通常带有 scaling factor，diffusers 中常见值为 `0.18215`。实现时必须与 pipeline 的 encode/decode 逻辑一致。
- 编码阶段使用 VAE posterior 的 mode 或 mean，避免引入额外随机性；本阶段随机性应主要来自 Langevin noise。

### 3.6 DDIM Inversion 模块 `DDIMInverter`

职责：

- 使用确定性 DDIM forward/inversion 将 `z0` 推到指定扩散步 `zT`。
- 提供无扰动重建测试能力，验证 forward/reverse 是否一致。

输入：

- `z0`。
- `text_embeddings`。
- `num_ddim_steps`。
- `inversion_step_T`。
- `unet`。
- `ddim_scheduler`。

输出：

- `zT`。
- `timestep_T`。
- 可选完整 latent trajectory。

实现要点：

- 需要明确论文中的 `T = 30` 是 50 个 DDIM 推理步中的第 30 个，而不是训练时间步 30。
- diffusers scheduler 的 timesteps 通常是从大到小排列；inversion 需要按相反方向推进。
- 验收时应做 reconstruction sanity check：不加 Langevin，直接 reverse 回图像，确认与输入结构接近。

### 3.7 梯度场估计模块 `TweedieGradientEstimator`

职责：

- 根据 UNet 的噪声预测近似论文 Eq.3 中的梯度场 `G = ∇zT log p(zT)`。
- 为 Langevin 模块提供每一步的更新方向。

输入：

- 当前 latent `z`。
- 当前 timestep `t`。
- `text_embeddings`。
- `unet`。
- `scheduler` 中对应 alpha/beta 参数。

输出：

- `gradient_G`。
- 可选 `noise_pred`。

实现要点：

- 论文近似形式可理解为由噪声预测反推 score direction。
- 实现时需要保证 `gradient_G` 的尺度不会过大，建议记录 norm，并在 debug 模式输出每步 norm。
- 初版不做复杂归一化，先严格按论文公式；若出现全图漂移，再增加 gradient clipping 作为可控实验项。

### 3.8 Langevin 扰动模块 `LangevinPerturber`

职责：

- 对 `zT` 做 `M` 步 annealed Langevin sampling。
- 为每个样本生成独立扰动 latent `zT_hat(k)`。

输入：

- `zT`。
- `langevin_steps_M`。
- `eta_start`、`eta_end`。
- `temperature_tau`。
- `TweedieGradientEstimator`。
- 随机种子。

输出：

- `zT_hat`。
- 每步调试信息：`eta`、`gradient_norm`、`noise_norm`、`latent_delta_norm`。

实现要点：

- `eta` 建议采用线性或对数退火，从 `1e-2` 到 `1e-4`。
- 每个 `k` 使用不同随机子种子，例如 `seed + k`。
- 如果显存紧张，K 个样本串行生成，避免 batch K 同时占用显存。

### 3.9 DDIM Reverse 模块 `DDIMReconstructor`

职责：

- 将扰动后的 `zT_hat(k)` 通过确定性 DDIM reverse 解码回 `z0_hat(k)`。
- 调用 VAE decode 得到视觉印象图像。

输入：

- `zT_hat(k)`。
- `timestep_T`。
- `text_embeddings`。
- `unet`。
- `ddim_scheduler`。
- `vae`。

输出：

- `sample_images`: `{V_hat(k)}`。
- `sample_latents`: 可选 `{z0_hat(k)}`。

实现要点：

- reverse 起点必须与 inversion 的 `T` 对齐。
- 保存每张 sample，便于人工检查不同采样的稳定性。

### 3.10 不确定性估计模块 `UncertaintyEstimator`

职责：

- 根据 K 张视觉印象计算像素级方差。
- 生成灰度不确定性图和伪彩色热力图。

输入：

- `sample_images`。

输出：

- `uncertainty_map.npy`: float map。
- `uncertainty_gray.png`。
- `uncertainty_heatmap.png`。

实现要点：

- 对 RGB 三通道可先计算每个像素三通道方差均值。
- 输出前做 min-max normalization。
- 若 K 太小，不确定性会偏噪，开发期可接受，正式评估应增大 K。

### 3.11 代表性视觉印象选择模块 `RepresentativeSelector`

职责：

- 按论文思路选择“与输入偏离最大”的视觉印象作为代表性 `V_hat`。

输入：

- 原始预处理图像。
- `sample_images`。

输出：

- `representative_image`。
- `representative_index`。
- 每个 sample 的差异分数。

差异分数建议：

- 初版使用像素级 L1 或 L2 均值。
- 后续可增加 LPIPS 或 CLIP feature distance，但要记录为扩展设置。

### 3.12 输出管理模块 `EnvisionOutputWriter`

职责：

- 保存所有阶段产物。
- 写入 `metadata.json`，保证复现实验可追踪。

输出目录建议：

```text
outputs/envision/{run_id}/
  original.png
  preprocessed.png
  impression.png
  difference.png
  uncertainty_gray.png
  uncertainty_heatmap.png
  uncertainty_map.npy
  samples/
    sample_000.png
    sample_001.png
  metadata.json
```

`metadata.json` 建议包含：

- 输入图路径。
- 模型路径。
- prompt。
- 所有超参数。
- device、dtype。
- seed。
- 每个样本差异分数。
- 代表性样本 index。
- 耗时信息。

## 4. 主控流程模块 `EnvisionPipeline`

职责：

- 串联上述模块，形成单图 Envision 处理闭环。

输入：

- `EnvisionConfig`。

输出：

- `EnvisionResult`，包含：
  - `original_image_path`
  - `impression_image_path`
  - `uncertainty_map_path`
  - `uncertainty_heatmap_path`
  - `metadata_path`

执行顺序：

1. 读取配置。
2. 预处理图像。
3. 加载 SD 与 DDIM。
4. 构造 text embeddings。
5. VAE encode 得到 `z0`。
6. DDIM inversion 得到 `zT`。
7. 循环 K 次执行 Langevin + DDIM reverse + VAE decode。
8. 计算不确定性图。
9. 选择代表性视觉印象。
10. 保存输出。

## 5. 验收与调试

最小验收：

- 对一张测试图输出完整目录。
- `impression.png` 非空、结构清晰。
- `uncertainty_heatmap.png` 非空，数值范围合理。
- `metadata.json` 可追踪全部参数。

关键调试项：

- `reconstruction_no_perturb.png`: 不加 Langevin 的重建结果。
- `latent_delta_norm`: 扰动前后 latent 差异。
- `gradient_norm`: 每步梯度尺度。
- `sample_diff_scores`: K 个样本与原图的差异。

失败现象与处理：

| 现象 | 可能原因 | 处理 |
| --- | --- | --- |
| 视觉印象整体重绘 | `eta` 过大或 DDIM inversion 不一致 | 降低 `eta_start`，先检查无扰动重建 |
| 输出几乎不变化 | `eta` 太小或梯度方向尺度过低 | 增大 `eta_start` 或检查 Tweedie 公式实现 |
| 不确定性图全黑 | K 个样本太接近 | 增大 K 或检查随机噪声是否生效 |
| 不确定性图全亮 | 采样过度随机 | 降低 `tau` 或缩短 M |

## 6. 与后续阶段的接口

Envision 阶段需要为 Attend 阶段提供稳定接口：

```text
{
  "original_image": ".../original.png",
  "preprocessed_image": ".../preprocessed.png",
  "impression_image": ".../impression.png",
  "uncertainty_map": ".../uncertainty_map.npy",
  "uncertainty_heatmap": ".../uncertainty_heatmap.png",
  "transform_meta": {...}
}
```

Attend 阶段只依赖 `original_image`、`impression_image`、`uncertainty_map` 和尺寸变换信息，不应重新运行 Envision。这样可以降低显存压力，也便于独立调试。