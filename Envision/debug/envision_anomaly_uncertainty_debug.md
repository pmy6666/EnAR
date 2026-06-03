# Envision 异常恢复与不确定性图不高亮排查

生成时间：2026-06-03

本次排查只阅读代码、配置、已有输出和本地设计文档，未修改 `EnAR/Envision/` 的业务代码。

## 1. 现象复述

你观察到：

- 复现出的视觉印象会把异常位置恢复成正常。
- 多张恢复结果看起来都差不多。
- `uncertainty_map` / `uncertainty_heatmap` 很难在异常位置形成明显高亮。

从 EnAR 的方法设定看，“异常位置被恢复成正常”本身不一定是错的。根据 `EnAR/docs/enar_reproduction_plan.md`，Langevin 方向会把不符合视觉先验的区域推向更常见、更真实的形态；后续 Attend 阶段依赖原图与视觉印象的 attention 差异，以及 uncertainty top 区域共同定位反事实 token。

真正需要警惕的是：如果 K 次视觉印象几乎一致，那么“按样本方差”得到的不确定性图天然不会突出异常区域。

## 2. 当前运行结果的数值证据

检查已有输出目录：

```text
EnAR/outputs/envision/run_001/
```

配置来自 `metadata.json`：

```text
input_image        = wolf_5.png
num_ddim_steps     = 50
inversion_step_T   = 30
langevin_steps_M   = 10
sample_count_K     = 10
eta_start/end      = 0.01 -> 0.0001
temperature_tau    = 0.1
guidance_scale     = 1.0
dtype/device       = float16 / cuda
timestep_T         = 601
```

`uncertainty_map.npy` 分布：

```text
shape = (512, 512)
min   = 0.00000164
max   = 0.04415913
mean  = 0.00021195
std   = 0.00069596

percentiles:
0%   0.00000164
50%  0.00005311
75%  0.00016865
95%  0.00085529
99%  0.00243219
100% 0.04415913
```

这个分布说明：极少数像素有很高值，99% 的像素都低于 0.00243。当前可视化用全局 min-max，会被极端最大值拉伸，导致大多数区域很暗。

样本差异统计：

```text
reconstruction_no_perturb vs preprocessed mean abs = [13.884, 13.273, 12.871] / 255
10 samples pairwise mean abs:
  min  = 2.812 / 255
  max  = 3.630 / 255
  mean = 3.153 / 255
```

这和现象一致：无扰动重建已经和输入有明显差异，但 K 个扰动样本彼此非常接近。因此 uncertainty 主要反映“小范围随机采样差异”，而不是“原图异常被恢复的位置”。

## 3. 高概率问题 1：DDIM reverse 起点疑似多跑了一步

相关代码：

- `EnAR/Envision/ddim_inverter.py:33-53`
- `EnAR/Envision/ddim_reconstructor.py:33-40`

inversion 逻辑：

```python
timesteps = list(self.scheduler.timesteps)
reverse_timesteps = list(reversed(timesteps))
...
for source_t, target_t in zip(reverse_timesteps[:-1], reverse_timesteps[1:]):
    ...
    if completed_steps >= inversion_step_T:
        break

step_index_T = min(inversion_step_T, len(timesteps) - 1)
timestep_T = int(timesteps[-1 - step_index_T].item())
```

以 diffusers DDIM 50 steps 为例，`scheduler.timesteps` 通常类似：

```text
[981, 961, ..., 21, 1]
```

因此：

- `reverse_timesteps` 从低噪声到高噪声，例如 `[1, 21, ..., 981]`。
- `inversion_step_T = 30` 后，latent 应该位于 `timesteps[-1 - 30]`，即当前 metadata 中的 `601`。

reconstructor 逻辑：

```python
start_index = len(timesteps) - 1 - step_index_T
selected = timesteps[start_index:]

for timestep in selected:
    latent = self.scheduler.step(...).prev_sample
```

当 `step_index_T=30` 时，`selected` 会包含当前 timestep `601`，然后从 `601` 做一次 scheduler.step 到上一个状态。

问题点：如果 `zT` 已经是 timestep 601 对应的 latent，那么 reverse 从 `601` 开始是合理的；但要和 inversion 的 `_forward_step(source_t, target_t)` 精确互逆，需要确认 diffusers `scheduler.step` 的 `prev_timestep` 是否恰好回到 `581` 或 schedule 中下一个低噪声点，并且 inversion 第 30 次 forward 是否确实生成的是 `x_601` 而不是“准备从 601 继续 forward 的状态”。当前无扰动重建平均误差约 13/255，不算灾难，但足够提示 forward/reverse 不是严格闭环。

本地设计文档 `EnAR/docs/enar_stage_4_1_envision_module_design.md` 明确要求：

```text
reverse 起点必须与 inversion 的 T 对齐。
验收时应做 reconstruction sanity check：不加 Langevin，直接 reverse 回图像，确认与输入结构接近。
```

建议后续验证：

- 打印 50-step `scheduler.timesteps`。
- 对 `inversion_step_T` 分别取 0, 1, 5, 10, 30，记录无扰动重建误差。
- 保存 inversion trajectory，然后尝试从 trajectory 中每个中间 latent reverse，检查哪一种索引最接近输入。
- 在 fake scheduler 测试中加入“forward 后 reverse 应回到原 latent”的强约束；当前 `tests/test_ddim_modules.py` 只检查 shape 和可 decode，不能发现索引错位。

## 4. 高概率问题 2：Langevin 噪声系数少了 `sqrt(2)`

本地复现计划写的是：

```text
zT <- zT + eta * G + sqrt(2 * eta * tau) * noise
```

当前 `EnAR/Envision/langevin.py:46`：

```python
delta = eta * estimate.gradient + (eta * temperature_tau) ** 0.5 * noise
```

这里随机项是 `sqrt(eta * tau)`，比计划/常见 Langevin 形式的 `sqrt(2 * eta * tau)` 小 `sqrt(2)` 倍。

影响：

- K 次采样之间的随机扩散更弱。
- 采样结果更容易被同一个梯度方向拉到相似的正常视觉先验。
- 按 K 样本方差计算的 uncertainty 更不明显。

这不是唯一原因，但它直接解释“都一样”的一部分。

## 5. 高概率问题 3：uncertainty 只度量样本间方差，不度量“被修复差异”

当前 `EnAR/Envision/uncertainty.py:21-29`：

```python
stack = np.stack(arrays, axis=0)
var_map = stack.var(axis=0).mean(axis=2)
normalized = self._normalize(var_map)
```

也就是说，uncertainty 是：

```text
U(x, y) = mean_rgb variance_k( V_hat_k(x, y) )
```

如果异常区域每次都被稳定恢复成同一个正常形态，则：

```text
V_hat_1(x, y) ~= V_hat_2(x, y) ~= ... ~= V_hat_K(x, y)
```

那么该区域样本方差会很低，即使它和原始异常输入差异巨大。

所以“异常被恢复成正常，但 uncertainty 不亮”并不矛盾。当前 `difference.png` 更可能显示“异常修复位置”，而 `uncertainty_map` 只显示“多次采样不一致的位置”。

这和 EnAR 后续 Attend 阶段的设定有关：论文并非只靠 uncertainty 单独定位异常，而是还比较原图和视觉印象在 LVLM vision encoder 中的 attention 差异。

建议后续验证：

- 对比 `difference.png` 是否比 `uncertainty_heatmap.png` 更能覆盖异常位置。
- 后续 Attend 阶段不要只使用 uncertainty；应结合 `DeltaA = abs(Attn(V) - Attn(V_hat))`。
- 如果要让 Envision 单阶段就显示异常修复位置，可以额外输出 `mean_abs(preprocessed, mean(samples))` 或 `mean_abs(preprocessed, representative)` 作为 debug map，但这属于新增诊断，不等价于论文中的 uncertainty。

## 6. 中概率问题 4：min-max 可视化被极端值压缩

当前 `EnAR/Envision/uncertainty.py:31-37`：

```python
normalized = (array - min) / (max - min)
```

已有输出中：

```text
99% percentile = 0.002432
max            = 0.044159
```

max 是 99% 分位的约 18 倍。因此 min-max 会让绝大多数像素落在很低亮度，视觉上接近全黑。即使 top-5% 在数值上可用，heatmap 也不一定“看起来高亮”。

建议后续验证：

- 用 95% 或 99% percentile clipping 重新可视化同一个 `uncertainty_map.npy`。
- 输出 top-5% mask，而不是只看连续热力图。
- 检查 Attend 阶段使用原始 `.npy` top-k，而不是依赖 heatmap 的视觉亮度。

## 7. 中概率问题 5：空 prompt + CFG=1 可能让样本更趋同

当前配置：

```text
prompt          = ""
negative_prompt = ""
guidance_scale  = 1.0
```

代码行为：

- `PromptConditioner.encode()` 对空 prompt 生成 unconditional text embedding。
- `DDIMInverter._predict_noise()` 中 `guidance_scale == 1.0` 时不会使用 negative branch。

这基本是无文本条件的 SD 视觉先验。它可能更偏向把异常区域恢复成训练分布中的常见形态，也可能让多次采样落到相似吸引盆地。论文是否使用空 prompt 要结合原文细节再确认；仅从当前本地实现看，这会降低条件多样性。

建议后续做参数扫：

```text
prompt = ""
prompt = "a natural photo"
prompt = 与图像类别相关的弱 prompt，例如 "a photo of a wolf"
guidance_scale = 1.0, 3.0, 5.0
```

但这属于实验策略，不能直接断定当前为空 prompt 就是 bug。

## 8. 中概率问题 6：只在单一 timestep 上做 M 步 Langevin，梯度可能收敛到同一模式

当前 `LangevinPerturber.perturb()` 在固定 `inversion.timestep_T` 上循环 M 次：

```python
estimate = self.gradient_estimator.estimate(latent, timestep, ...)
latent = latent + delta
```

这符合“在 zT 上扰动”的实现理解，但副作用是所有样本共享同一个起点、同一个 timestep、同一个确定性梯度场，只是噪声 seed 不同。在较强 score drift 下，不同样本会被拉回相近区域，方差降低。

已有 metadata 中第一步：

```text
gradient_norm ~= 130
noise_norm    ~= 128
latent_delta_norm ~= 4.2
```

虽然 noise norm 和 gradient norm 数值接近，但 delta 中梯度项是 `eta * gradient`，噪声项是 `sqrt(eta*tau)*noise`。在 `eta=0.01,tau=0.1` 时，噪声系数约 0.0316，噪声项范数约 4.0；梯度项范数约 1.3。早期噪声并不小，但 reverse 解码和视觉先验可能仍把样本压到相似图像。

建议后续检查：

- 保存扰动后 latent 之间的 pairwise L2。
- 保存 reverse 前后的 latent pairwise L2，判断是 Langevin 已趋同，还是 DDIM reverse/VAE decode 后趋同。

## 9. 测试覆盖缺口

当前测试比较轻量，不能发现本次现象：

- `tests/test_ddim_modules.py` 只断言 inversion 输出 shape 和 step index，以及 reconstructor 能 decode。
- 没有测试 DDIM forward/reverse 的闭环误差。
- 没有测试 `inversion_step_T` 与 `timestep_T`、reverse selected timesteps 的精确对应。
- `tests/test_uncertainty_representative.py` 只用极端纯色图片测试方差输出，没有覆盖“样本彼此相似但都与原图差异大”的场景。

建议后续新增测试：

```text
1. scheduler timesteps ordering test
2. inversion trajectory index alignment test
3. no-perturb reconstruction regression test
4. uncertainty vs difference semantic distinction test
5. heatmap percentile visualization test
```

## 10. 优先级结论

按排查优先级排序：

1. 先验证 DDIM inversion/reverse 是否严格对齐。
   无扰动重建是整个 Envision 的地基；当前约 13/255 的平均差异值得继续追。

2. 检查 Langevin 噪声公式。
   当前实现缺少 `sqrt(2)`，会降低 K 次采样差异，直接影响 uncertainty。

3. 不要期待当前 uncertainty 单独高亮“稳定被修复”的异常。
   当前 uncertainty 是样本间方差；异常如果每次都被同样修复，方差会低。需要结合 difference map 或 Attend 阶段 attention difference。

4. 对 uncertainty heatmap 做 percentile clipping/top-k mask 可视化。
   当前 min-max 可视化容易被极端值压暗，不代表 `.npy` 在 top-k 使用时完全无效。

5. 做 prompt/guidance/T/eta/K 的小规模 sweep。
   重点观察：样本间差异、异常区域 difference、uncertainty top-5% 是否落在异常附近。

## 11. 建议的下一步 debug 实验

不改核心代码的情况下，可以先做这些离线检查：

```text
Experiment A: DDIM sanity
- inversion_step_T = 0, 1, 5, 10, 20, 30
- langevin_steps_M = 0
- 记录 reconstruction_no_perturb 与 preprocessed 的 mean abs / PSNR

Experiment B: sample diversity
- 固定 T=30
- eta_start = 0.003, 0.01, 0.03
- temperature_tau = 0.05, 0.1, 0.2
- K=10
- 记录 samples pairwise mean abs 与 uncertainty 分位数

Experiment C: map comparison
- uncertainty_map
- difference(preprocessed, representative)
- difference(preprocessed, mean(samples))
- top-5% uncertainty mask
- top-5% difference mask

Experiment D: visualization only
- 对同一个 uncertainty_map.npy 做 95% / 99% percentile clipping heatmap
- 判断是数值无信号，还是可视化压缩造成看不出来
```

这些实验能把问题拆成三类：DDIM 闭环问题、采样多样性问题、可视化/语义解释问题。
