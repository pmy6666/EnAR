# DDIM reverse 起点/步数对齐专项排查

生成时间：2026-06-03

本次只排查 `debug/envision_anomaly_uncertainty_debug.md` 中的第 1 个问题：`DDIM reverse 起点疑似多跑了一步`。本次未修改任何业务代码。

## 1. 结论

有 bug，但更准确地说：

```text
reverse 从 timestep_T 开始不是错；
真正的问题是 reverse 的 selected timesteps 包含了最后的 t=1，
导致 inversion 做了 N 次 forward transition，reverse 却做了 N+1 次 denoise transition。
```

对当前配置 `num_ddim_steps=50, inversion_step_T=30`：

```text
scheduler.timesteps = [981, 961, 941, ..., 41, 21, 1]

inversion 第 30 步：
  581 -> 601
  因此 zT 确实处在 timestep 601。

当前 reverse：
  601 -> 581 -> ... -> 21 -> 1 -> final_alpha/x0
  一共 31 步。

和 inversion 对齐的 reverse 应该是：
  601 -> 581 -> ... -> 21 -> 1
  一共 30 步。
```

所以旧 debug 文档里“从 601 开始可能多跑一步”的表述不够精确。`601` 这个起点本身是对的；多出来的是最后 `t=1` 的那次 `scheduler.step()`。

## 2. 关键证据

使用本地 SD v1.5 scheduler 检查 50-step DDIM timesteps：

```text
前 5 个: [981, 961, 941, 921, 901]
后 5 个: [81, 61, 41, 21, 1]
len = 50
step_offset = 1
timestep_spacing = leading
```

反向 inversion 的前 35 个 transition 是：

```text
1  (1, 21)
2  (21, 41)
3  (41, 61)
...
28 (541, 561)
29 (561, 581)
30 (581, 601)
31 (601, 621)
```

因此 `inversion_step_T=30` 后，latent 处在 `601`。当前代码返回：

```text
step_index_T = 30
timestep_T   = timesteps[-1 - 30] = 601
```

这部分是对的。

## 3. 问题代码在哪里

问题在 `EnAR/Envision/ddim_reconstructor.py`：

```python
timesteps = list(self.scheduler.timesteps)
start_index = len(timesteps) - 1 - step_index_T
selected = timesteps[start_index:]

latent = zT
for timestep in selected:
    noise_pred = self._noise_helper._predict_noise(...)
    latent = self.scheduler.step(noise_pred, timestep, latent, eta=0.0).prev_sample
```

当 `step_index_T=30`：

```text
start_index = 49 - 30 = 19
selected    = timesteps[19:]
            = [601, 581, ..., 41, 21, 1]
len         = 31
```

但 inversion 只做了 30 个 transition，所以 reverse 不应该再执行最后的 `t=1` step。

更明显的边界案例：

```text
step_index_T = 0
```

当前逻辑：

```text
start_index = 49
selected = [1]
```

也就是说，即使没有做任何 inversion，`reconstruct(z0, step_index_T=0)` 仍然会对原始 latent 执行一次 `scheduler.step(..., t=1)`。这显然不是“无扰动重建应该等于输入 latent”的行为。

## 4. 为什么会出现这个 off-by-one

当前 `DDIMInverter` 从 VAE latent `z0` 开始：

```python
reverse_timesteps = list(reversed(timesteps))

for source_t, target_t in zip(reverse_timesteps[:-1], reverse_timesteps[1:]):
    noise_pred = self._predict_noise(latent, source_t, ...)
    latent = self._forward_step(latent, noise_pred, source_t, target_t)
```

对于 50-step leading schedule，`reverse_timesteps` 从 `1` 开始：

```text
[1, 21, 41, ..., 981]
```

所以 inversion 的第 1 步实际是：

```text
把 z0 当作 x_1，用 UNet(t=1) 预测噪声，然后 forward 到 x_21。
```

这是一种常见近似，因为 schedule 没有显式包含训练 timestep 0，而 `t=1` 和真正干净 latent 很接近。

在这个定义下，reverse 的目标应该是回到同一个近似状态 `x_1`，然后直接 VAE decode。当前代码又执行了：

```text
x_1 -> final_alpha/x0
```

于是多了一步。

## 5. 一维代数 sanity check

我做了一个不加载 UNet/VAE 的一维检查：固定同一个 `eps`，用当前 `_forward_step()` 公式 forward，再用 diffusers `scheduler.step()` reverse。

结果：

```text
n=0:
  当前 selected=[1]，误差约 0.00106585
  排除 t=1，不执行 reverse step，误差 0.0

n=1:
  当前 reverse 2 步，误差约 0.00106585
  排除 t=1，reverse 1 步，误差 0.0

n=2:
  当前 reverse 3 步，误差约 0.00106585
  排除 t=1，reverse 2 步，误差 0.0

n=30:
  当前 reverse 31 步，误差约 0.00106627
  排除 t=1，reverse 30 步，误差约 0.00000042
```

这个检查去掉了模型预测误差，说明 off-by-one 是调度步数问题，而不是 UNet 或 VAE 造成的。

## 6. 我打算怎么修改

如果后续允许修改代码，我建议优先改 `DDIMReconstructor.reconstruct()` 的 timestep 选择：

```python
timesteps = list(self.scheduler.timesteps)
start_index = len(timesteps) - 1 - step_index_T
selected = timesteps[start_index:-1]
```

含义：

```text
从当前 zT 所在 timestep 开始 reverse；
但排除最后一个 timestep，也就是 t=1；
让 reverse transition 数量等于 inversion_step_T。
```

边界行为：

```text
step_index_T = 0:
  selected = []
  直接 decode z0

step_index_T = 1:
  selected = [21]
  x_21 -> x_1

step_index_T = 30:
  selected = [601, 581, ..., 21]
  x_601 -> ... -> x_1
```

这和当前 inversion 的定义一致：把原始 VAE latent 视作 schedule 中的 `x_1` 近似。

## 7. 是否需要同时改 DDIMInverter

短期可以只改 `DDIMReconstructor`。

更严格的长期方案有两个：

### 方案 A：保持当前 inversion 定义，只修 reverse

这是最小修改：

```text
inversion:
  z0 ~= x_1
  x_1 -> x_21 -> ... -> x_T

reverse:
  x_T -> ... -> x_21 -> x_1
  decode x_1
```

优点：

- 改动小。
- 和当前 `step_offset=1` 的 diffusers schedule 兼容。
- 可以消除明显的 N vs N+1 步数不一致。

缺点：

- 严格说仍然把 clean latent 近似成 `x_1`，不是显式的 `x_0`。
- 但 `alpha_cumprod[1]` 非常接近 1，这个近似通常可接受。

### 方案 B：显式引入 clean timestep 0/final_alpha

让 inversion 从真正 clean latent 先到 `x_1`，再继续到 `x_21` 等：

```text
x0 -> x_1 -> x_21 -> ... -> x_T
```

reverse 则保留：

```text
x_T -> ... -> x_21 -> x_1 -> x0
```

优点：

- 数学定义更完整。

缺点：

- 需要更细地处理 `final_alpha_cumprod`、UNet 在 `t=0/1` 的调用语义，以及 `inversion_step_T` 的含义。
- 改动更大，容易影响后续配置解释。

我建议先采用方案 A。

## 8. 建议同步增加的测试

如果后续改代码，建议补这些测试，避免同类问题复发：

```text
1. step_index_T=0 时，reconstructor 不应调用 scheduler.step。
2. step_index_T=N 时，reconstructor 调用 scheduler.step 的次数应等于 N。
3. 用真实 DDIMScheduler + 固定 eps 做 algebra roundtrip：
   forward N steps，再 reverse N steps，误差应接近 0。
4. 检查 selected timesteps：
   N=30 时应为 [601, 581, ..., 21]，不应包含最后的 1。
```

当前 `tests/test_ddim_modules.py` 只检查 shape 和能 decode，无法发现这个 off-by-one。

## 9. 对现象的影响

这个 bug 会影响 `reconstruction_no_perturb.png` 和所有 sample 的最终视觉印象。

不过它不一定是 uncertainty 不高亮的唯一原因：

- 最后 `t=1` 这一步数值上很小，但会破坏严格闭环。
- 已有输出中无扰动重建平均差约 `13/255`，其中一部分可能来自这个 off-by-one，另一部分来自真实 UNet inversion 近似、VAE 编解码、float16 等因素。
- K 个 sample 彼此接近、uncertainty 只统计样本间方差，仍然是 uncertainty 不高亮的重要原因。

因此修完这个问题后，建议重新跑：

```text
inversion_step_T = 0, 1, 5, 10, 30
langevin_steps_M = 0
```

观察 `reconstruction_no_perturb.png` 与 `preprocessed.png` 的 mean absolute difference 是否明显下降。
