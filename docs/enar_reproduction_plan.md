# EnAR 论文方法复现与本地实施方案计划书

## 1. 目标与范围

本文档基于 `Liang_Envision_Attend_Then_Respond_Counterfactual_Hallucination_Mitigation_in_Large_Vision-Language_CVPR_2026_paper.pdf` 的方法部分，以及 `Envision, Attend, Then Respond_ Counterfactual Hallucination Mitigation in Large Vision-Language Models (CVPR 2026)论文复现与延伸思考.pdf` 中 3.1 的复现要求，结合当前 `EnAR/pre_model/` 下已有模型，制定 EnAR 的本地复现与扩展实施计划。

本阶段只输出方案计划书，不进行代码实现。后续编码目标优先覆盖复现思考文档 3.1 的 Envision 阶段，再逐步扩展到 Attend 与 Respond，形成完整 training-free 的反事实幻觉缓解流程。

## 2. 论文方法要点

EnAR 是一个 training-free 框架，核心思想是利用外部扩散模型中的真实世界视觉先验，构造输入图像的“视觉印象”，再让 LVLM 比较原图与视觉印象之间的注意力差异，定位反事实区域，最后通过原始视觉输入与 masked/padded 视觉输入的对比解码抑制幻觉。

### 2.1 Envision: 视觉印象生成

输入为原始图像 `V`。论文将图像经 VAE 编码到扩散模型 latent `z0`，使用确定性 DDIM forward 到指定时间步 `T` 得到 `zT`。随后在 `zT` 上做 `M` 步 annealed Langevin 扰动：

```text
zT <- zT + eta * G + sqrt(2 * eta * tau) * noise
```

其中梯度场 `G = ∇zT log p(zT)` 由 Tweedie 估计近似得到，方向指向扩散模型视觉先验下更高似然的区域。直观上，它会把不符合常识的区域往更常见、更真实的形态推，同时尽量保留图像正常区域。论文设置包括 `T = 30`、总 DDIM 步数 `Tmax = 50`、`M = 10`、步长 `eta` 从 `1e-2` 退火到 `1e-4`、温度 `tau = 0.1`。

为了估计视觉印象的不确定性，EnAR 对同一个 `zT` 做 `K` 次扰动采样，得到多张视觉印象 `{V_hat(k)}`，按像素方差构造不确定性图 `U`。代表性视觉印象按论文思路选择与输入偏离最大的样本，用于后续注意力对比。

### 2.2 Attend: 反事实 token 定位

输入为原图 `V`、视觉印象 `V_hat` 和不确定性图 `U`。论文从 LVLM vision encoder 的第 `L` 层提取注意力。对于含 `cls` token 的 CLIP 类视觉编码器，取 `cls` token 对 patch token 的注意力；对于无 `cls` token 的视觉编码器，则统计各 token 的 incoming attention 并按 head 平均。

原图与视觉印象的注意力差异为：

```text
DeltaA = abs(Attn_L(V) - Attn_L(V_hat))
```

随后取 `DeltaA` 分数最高的 top-K% token 作为 `Hattn`，同时将像素级不确定性图 `U` 映射到视觉 patch 网格，取 top-5% 不确定 token 作为 `Hunc`，二者并集为候选反事实 token：

```text
H = Hattn union Hunc
```

论文默认使用第 6 层 vision encoder 和 10% padding token 比例。对于当前本地 LLaVA-1.5-7B，其视觉编码器为 CLIP ViT，配置中 `image_size = 336`、`patch_size = 14`、`num_hidden_layers = 24`，理论 patch 网格为 `24 x 24 = 576`，若使用 `full` 策略还会包含额外 `cls` token。

### 2.3 Respond: 对比解码

Attend 阶段将候选反事实视觉 token 替换为模型内置 pad token 或对应 pad embedding，构造 padded visual input `v'`。Respond 阶段对同一问题 `x` 分别使用原始视觉输入 `v` 和 padded 视觉输入 `v'` 前向，得到两路 next-token logits，并执行对比解码：

```text
p(y | x, v, v') = (1 + alpha) * p(y | x, v) - alpha * p(y | x, v')
```

这里 `alpha` 控制抑制强度。直觉是：如果某些 token 更依赖被遮蔽后的偏置分支，则在合并 logits 时被压低；真正由原始视觉证据支撑的 token 会被保留或增强。

## 3. 当前本地模型与资源

当前 `EnAR/pre_model/` 已具备复现 EnAR 的核心模型资产。

| 模块 | 本地路径 | 用途 |
| --- | --- | --- |
| LLaVA-1.5-7B | `EnAR/pre_model/LLM/llava-1.5-7b-hf/` | LVLM 主干，负责视觉编码、问答生成、注意力提取和对比解码 |
| Stable Diffusion v1.5 | `EnAR/pre_model/DDIM/stable-diffusion-v1-5/` | Envision 阶段的扩散视觉先验，负责 DDIM inversion、Langevin latent perturbation 和视觉印象生成 |
| LLaVA 加载测试脚本 | `EnAR/work_scripts/test_LLaVA/test_load_llava_1_5_7b.py` | 已包含本地模型文件检查、processor patch、单图问答 smoke test |
| SD 加载测试脚本 | `EnAR/work_scripts/test_DDIM/test_load_sd_v1_5.py` | 已包含本地 Stable Diffusion pipeline 加载和生成 smoke test |

需要注意：Stable Diffusion 目录的 `model_index.json` 默认 scheduler 是 `PNDMScheduler`，但论文和复现要求需要确定性 DDIM。因此正式实现时应显式替换为 `DDIMScheduler`，并使用一致的 timestep 设置完成 forward/inversion 与 reverse。

## 4. 实施路线

### 4.1 第一阶段：Envision 复现

目标是满足复现思考文档 3.1：输入一张图像，输出视觉印象和不确定性热力图，并能说明扰动效果是否符合预期。

计划步骤：

1. 加载本地 Stable Diffusion v1.5 的 VAE、UNet、tokenizer、text encoder，并将 scheduler 替换为 DDIM。
2. 将输入图像 resize/crop 到 SD v1.5 支持的尺寸，建议默认 `512 x 512`，同时保留原图尺寸用于结果回写。
3. 使用 VAE encoder 得到 latent `z0`。
4. 实现确定性 DDIM forward/inversion，将 `z0` 推到论文设定的 `T = 30`。
5. 构造条件文本。初版可采用空 prompt 或通用 prompt；若后续希望更贴近图像内容，可先用 LLaVA 生成 caption，再作为 SD 条件。
6. 由 UNet 噪声预测近似 Tweedie 梯度场 `G`，对 `zT` 做 `M = 10` 步 annealed Langevin 扰动。
7. 重复 `K` 次扰动采样，得到多张视觉印象。
8. 计算逐像素方差，归一化并生成不确定性热力图。
9. 选择与输入图像像素差异最大的视觉印象作为代表性 `V_hat`。
10. 输出原图、代表性视觉印象、差异图、不确定性热力图和元数据 JSON。

建议默认参数：

| 参数 | 建议值 | 说明 |
| --- | --- | --- |
| `num_ddim_steps` | 50 | 对齐论文 `Tmax = 50` |
| `inversion_step_T` | 30 | 对齐论文设置 |
| `langevin_steps_M` | 10 | 对齐论文设置 |
| `sample_count_K` | 4 或 8 起步 | 资源受限时先用 4；正式实验建议扩大 |
| `eta_start` | `1e-2` | Langevin 初始步长 |
| `eta_end` | `1e-4` | Langevin 末端步长 |
| `temperature_tau` | 0.1 | 控制随机扰动强度 |
| `dtype` | float16 on CUDA | 降低显存占用 |

第一阶段验收标准：

- 对单张测试图能稳定输出视觉印象和不确定性热力图。
- 正常区域结构基本保留，不应整体重绘或语义漂移。
- 不确定性热力图能在局部变化区域呈现更高响应。
- 记录运行耗时、显存峰值、参数配置和随机种子。

### 4.2 第二阶段：Attend 复现

目标是基于 LLaVA-1.5-7B 的 CLIP vision encoder，提取原图与视觉印象的第 6 层注意力差异，并输出反事实 token 索引及 patch 可视化。

计划步骤：

1. 加载 `LlavaForConditionalGeneration` 和 `AutoProcessor`，沿用现有测试脚本中的 processor patch 逻辑。
2. 对原图和视觉印象执行同样的 image preprocessing，确保 patch 对齐。
3. 对 `model.vision_tower` 或对应 CLIP vision model 打开 `output_attentions=True`。
4. 提取第 6 层 attention，取 `cls -> patch` 注意力并按 head 平均。
5. 将 `DeltaA` reshape 到 `24 x 24` patch 网格。
6. 将 Envision 输出的不确定性图 resize/average pool 到 `24 x 24`。
7. 取 attention top-K% 与 uncertainty top-5% 的并集，并受 10% padding ratio 上限约束。
8. 输出 token index 列表、patch mask、原图叠加可视化。

关键风险：

- LLaVA 的 `vision_feature_select_strategy` 可能导致 575/576 token 差异。当前本地测试脚本默认使用 `full` 并设置 `num_additional_image_tokens=1`，后续实现要明确 `cls` token 是否保留。
- 不同 transformers 版本中 LLaVA 内部模块命名可能不同，应先用最小 introspection 脚本确认 attention 输出位置。

### 4.3 第三阶段：Respond 复现

目标是在 LLaVA-1.5-7B 上实现原始视觉输入与 padded 视觉输入的两路 logits 合并。

计划步骤：

1. 明确 LLaVA 图像特征进入语言模型前的张量形态，即 projected image embeddings。
2. 根据 Attend 输出的 patch indices，在 projected visual embeddings 中将对应位置替换为 pad embedding 或零化/learned pad 近似。
3. 对同一文本问题分别构造原始分支和 padded 分支。
4. 在 generation loop 中每步分别前向得到两路 logits。
5. 使用 `(1 + alpha) * logits_orig - alpha * logits_pad` 合并，再执行 greedy 或 sampling。
6. 可选加入 VCD 的 adaptive plausibility constraint，避免低可信 token 被对比项异常放大。

初版建议先做 greedy decoding，减少采样噪声；`alpha` 可从 VCD 常用配置开始网格搜索，例如 `0.5, 1.0, 1.5`。

## 5. 实验验证设计

### 5.1 Envision 单阶段验证

由于当前任务重点是 3.1，第一轮验证以可视化和数值 sanity check 为主。

验证内容：

- 输入图、视觉印象、差异图、不确定性热力图四联图。
- 多个 `K` 样本之间的方差是否集中在变化/不稳定区域。
- 不同 `T`、`M`、`eta` 下的扰动强度对比。
- 与原图的 LPIPS、L1/SSIM 或简单像素差异，作为“保留正常区域”的粗略指标。

### 5.2 Attend 定位验证

在没有完整 VLMBias 数据集时，可先构造少量人工样例或使用论文示例类图像，检查 patch mask 是否覆盖异常区域。

验证内容：

- attention-only、uncertainty-only、attention+uncertainty 三种 mask 对比。
- 第 3、6、9、12 层 attention 可视化对比，验证第 6 层是否适合本地 LLaVA。
- padding ratio 从 5%、10%、15% 扫描，观察覆盖范围和误伤区域。

### 5.3 端到端验证

完整流程跑通后，优先选择小规模问题集做 A/B 测试：

- Baseline: LLaVA regular decoding。
- Envision-only sanity: 仅查看视觉印象，不改解码。
- Attend+Respond: 完整 EnAR。
- 可选 baseline: VCD 风格图像扰动对比解码。

指标：

- 反事实问答准确率。
- yes/no 问题的准确率、precision、recall、F1。
- 输出是否出现常识优先而忽视图像证据的错误。
- 单样本耗时和显存峰值。

## 6. 资源与工程约束

1. LLaVA-1.5-7B 和 SD v1.5 同时加载显存压力较大。建议分进程或分阶段缓存：先离线生成视觉印象与不确定性图，再加载 LLaVA 做 Attend/Respond。
2. Envision 的 `K` 次采样成本最高。开发期建议 `K=4`，正式结果再扩大。
3. SD v1.5 使用 512 分辨率，LLaVA vision encoder 使用 336 分辨率，必须在输出中记录尺寸变换，并保证 uncertainty map 映射到 patch 网格时坐标一致。
4. 当前本地 SD 配置默认不是 DDIM scheduler，实现时必须显式构造 `DDIMScheduler`，否则不满足论文的确定性 forward/reverse 要求。
5. 如果空 prompt 导致视觉印象不稳定，可引入 LLaVA caption 或 BLIP 风格 caption 作为条件，但这会让 pipeline 多一个依赖分支，需要在实验中报告。

## 7. 风险点与应对

| 风险 | 影响 | 应对 |
| --- | --- | --- |
| DDIM inversion 实现与 diffusers scheduler 细节不一致 | 视觉印象不能保留原图结构 | 先做 inversion-reconstruction 测试，确保无 Langevin 时能近似重建 |
| Langevin 步长过大 | 全图语义漂移、正常区域被重绘 | 保持论文退火设置，并增加差异阈值与可视化检查 |
| 不确定性图过噪 | Attend 阶段误选背景 token | 使用 patch-level average pooling 和 top 比例约束 |
| 第 6 层不适配本地 transformers/LLaVA 输出 | 定位效果差 | 做多层可视化和小样本 IoU/人工覆盖评估 |
| padded embedding 集成困难 | Respond 无法直接接入 generate | 先实现核心 logits merge 函数，再改写最小 generation loop |
| 显存不足 | 无法端到端同时运行 | Envision 离线缓存，Attend/Respond 单独加载 LLaVA |

## 8. 里程碑

### M1: Envision 可视化闭环

交付物：

- 单图视觉印象生成脚本。
- 输出目录包含 `original.png`、`impression.png`、`uncertainty_heatmap.png`、`difference.png`、`metadata.json`。
- 一页简短观察记录，说明扰动是否符合预期。

### M2: Attend token 定位

交付物：

- 第 6 层 contrastive attention 提取。
- uncertainty map 到 LLaVA patch 网格的映射。
- 反事实 token index 输出与 patch overlay 可视化。

### M3: Respond 对比解码

交付物：

- 两路 logits 合并函数。
- LLaVA 最小 greedy generation loop。
- 单样例 regular vs EnAR 输出对比。

### M4: 小规模评测与消融

交付物：

- 至少 20 到 50 个样例的小规模评测结果。
- `w/o uncertainty map`、`w/o visual impression`、不同 layer、不同 padding ratio 的消融。
- 耗时和显存统计。

## 9. 推荐目录结构

后续如进入代码实现，可采用以下结构，便于隔离阶段产物：

```text
EnAR/
  scripts/
    envision_generate.py
    attend_localize.py
    respond_decode.py
    run_enar_pipeline.py
  outputs/
    envision/
    attend/
    end_to_end/
  docs/
    enar_reproduction_plan.md
```

## 10. 结论

当前本地模型资产已经覆盖 EnAR 复现所需的两个关键先验来源：`Stable Diffusion v1.5` 提供 Envision 阶段的视觉先验，`LLaVA-1.5-7B` 提供 Attend/Respond 阶段的 LVLM 视觉编码和语言生成能力。最稳妥的实施方式是先将 Envision 阶段作为独立可视化模块跑通并缓存结果，再接入 LLaVA 提取注意力，最后实现对比解码。这样既符合论文的三阶段数据依赖，也能在显存和调试复杂度上保持可控。
