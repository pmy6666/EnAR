# EnAR 三阶段串联排查报告

生成日期：2026-06-04
复查更新：2026-06-05

本报告只基于当前工作区代码与已有输出进行排查，未修改 `Attend/`、`Envision/`、`Respond/` 源码。重点检查三阶段产物如何串联，以及当前 `outputs/*/run_*` 中能解释“实际测试效果不理想”的信号。

2026-06-05 补充：用户已使用统一的 `original.png` 重新运行 `outputs/*/run_001`。新 run 中，`Attend` 的 `original_image` 与 `Respond` 的 `image_path` 均已指向 `outputs/envision/run_001/original.png`。因此上一版报告中的“最终问答使用 SD 重建图”问题已经修正，当前主要问题转为：视觉 token 策略仍未统一、padded 分支仍明显异常、Attend mask 与“数腿”问题语义仍不够对齐。

## 1. 当前串联链路

当前配置实际形成的链路如下：

1. `Envision` 输入：`EnAR/Envision/image/data/wolf_5.png`
2. `Envision` 输出：`outputs/envision/run_001/`
   - `reconstruction_no_perturb.png`
   - `impression.png`
   - `uncertainty_map.npy`
   - `metadata.json`
3. `Attend` 输入：
   - original image: `outputs/envision/run_001/original.png`
   - impression image: `outputs/envision/run_001/impression.png`
   - uncertainty map: `outputs/envision/run_001/uncertainty_map.npy`
4. `Attend` 输出：`outputs/attend/run_001/attend_result.json`
5. `Respond` 输入：
   - image: `outputs/envision/run_001/original.png`
   - attend result: `outputs/attend/run_001/attend_result.json`
   - question: `How many legs does this wolf have?`
6. `Respond` 输出：
   - regular answer: `The wolf in the image has four legs.`
   - EnAR answer: `The wolf has four legs.`

结论：三阶段在文件层面可以串起来，且当前 run 已经统一到 `original.png` 上评估。当前效果不理想已经不能主要归因于“Respond 使用了 reconstruction 图”。

## 2. 已确认的关键问题

### 2.1 Attend 与 Respond 的视觉 token 布局配置不一致

`Attend/attend_config.yaml` 使用：

```yaml
vision_feature_select_strategy: full
num_additional_image_tokens: 1
```

`Respond/respond_config.yaml` 使用：

```yaml
vision_feature_select_strategy: default
num_additional_image_tokens: 1
```

这会造成两个阶段对视觉 token 是否包含 CLS token 的理解不同。

当前 `attend_result.json` 中记录：

- `patch_grid`: `[24, 24]`
- patch 数量：576
- `has_cls_token`: `true`
- `selected_patch_indices`: 58 个
- `selected_vision_token_indices`: 58 个，且相对 patch index 整体 `+1`

但 `respond_result.json` 中实际视觉 embedding 为：

- `token_count`: 576
- `has_cls_token`: `false`
- `selected_vision_token_indices` 被修正回 patch index

也就是说，`Respond/visual_embeddings.py` 里有一层防护逻辑：当实际视觉 embedding 长度等于 patch 数量 576 时，会忽略 Attend 保存的 `+1` 后索引，改用 `selected_patch_indices`。这避免了直接越位或错位一格，但也暴露了一个根本问题：三阶段配置对视觉 token 布局没有统一。

影响：

- 当前 run 没有因为这个问题崩溃。
- 但这会让后续实验非常脆弱；一旦 Respond 改成 `full`、模型版本变化、processor patch 策略变化，padding 位置可能立刻错位。
- `attend_result.json` 中保存的 `selected_vision_token_indices` 与 Respond 实际使用的索引不一致，容易误导排查。

优先级：高。

### 2.2 padded 分支的输出分布明显异常

`outputs/respond/run_001/decode_trace.json` 显示，原图分支 `top_orig` 是正常语言分布，但 padded 分支 `top_pad` 每一步都强烈偏向异常碎片 token，例如：

- step 0: `ms`, `s`, `m`
- step 1: `ms`, `m`, `ss`
- step 2: `ms`, `as`, `ss`
- step 4: `ms`, `ss`, `as`

这说明 padded visual embeddings 不是一个合理的“去掉高风险视觉区域后”的反事实视觉输入，而更像是把多模态输入扰乱到了模型不熟悉的状态。

当前 padding 策略是 `pad_token_embedding`。统计结果显示，被替换视觉 token 的 norm 从大约 `18-80` 变成统一的 `0.1509`。这对 LLaVA 的视觉 embedding 分布来说是非常强的 out-of-distribution 替换。

影响：

- contrastive logits 当前公式是 `(1 + alpha) * logits_orig - alpha * logits_pad`。
- 如果 `logits_pad` 本身是异常分布，那么对比解码实际是在减去一个失真的语言分布，而不是减去“关键视觉证据缺失后的回答倾向”。
- 当前问题简单，所以 EnAR 结果仍然回答了 four legs；但这不代表方法稳定，复杂问题上很可能出现怪异、重复、过短或错误答案。

优先级：最高。


### 2.4 Envision 的 impression 选择策略可能过于激进

`Envision/representative.py` 当前选择与原图 L2 距离最大的 sample 作为 `impression`：

```python
index = int(np.argmax(scores))
```

当前 `run_001` 的 sample diff scores 约为：

```text
84.84 - 89.47
representative_index = 6
```

`run_003` 的 sample diff scores 约为：

```text
53.63 - 55.14
representative_index = 5
```

这说明不同 run 的扰动强度差异明显。选择“离原图最远”的 sample 可以制造强 counterfactual，但也可能让 impression 偏离原图语义太多，导致 Attend 关注到生成伪影、背景变化、纹理漂移，而不是真正与幻觉相关的区域。

优先级：中高。

### 2.5 uncertainty map 的动态范围在不同 run 间差异很大

统计结果：

```text
run_001 uncertainty_map:
  min 0.0000196, max 0.076798, mean 0.001567, std 0.002760

run_003 uncertainty_map:
  min 0.0000029, max 0.015095, mean 0.000065, std 0.000177
```

虽然 Attend 选 top ratio，绝对值不直接决定数量，但这说明 Envision 的采样不确定性稳定性不足。对于后续融合 attention/uncertainty 的选择，run 间波动会导致 mask 区域变化较大。

优先级：中。

## 3. 当前 Attend 选择结果

`Attend` 当前选择：

- patch grid: `24 x 24`
- 总 patch 数：576
- `attention_top_ratio = 0.10`，理论约 58 个 patch
- `uncertainty_top_ratio = 0.05`，理论约 29 个 patch
- `padding_ratio_limit = 0.10`，最多约 58 个 patch
- 实际 selected_total: 58

source counts：

```json
{
  "attention_only": 33,
  "uncertainty_only": 18,
  "attention_and_uncertainty": 7,
  "selected_total": 58
}
```

这说明最终 mask 基本被 padding limit 截断到上限。注意 attention 与 uncertainty 的重叠只有 7 个 patch，二者一致性较弱。若 mask 可视化没有落在目标物关键区域，这个低重叠可能是原因之一。

## 4. 代码层面的可疑点

### 4.1 `Respond` 手动构造 `inputs_embeds`，绕开了模型原生 multimodal prepare 逻辑

`Respond/dual_branch_forwarder.py` 使用 `build_inputs_embeds()` 手动把 image features scatter 到 `<image>` token 位置，然后直接调用 language model：

```python
outputs = language_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
```

这条路径与 `model.generate(input_ids=..., pixel_values=...)` 的原生路径不完全等价。LLaVA/HF 版本里通常还会有 position ids、cache、special image token expansion、feature select strategy 等内部处理。当前代码能跑通，说明 placeholder 数量匹配，但不能证明分布等价。

结合 padded 分支异常 token 分布，这一块需要优先验证。

### 4.2 `pad_token_embedding` 不适合作为视觉 embedding 替换向量

视觉 embedding hidden size 与文本 embedding hidden size 都是 4096，所以 shape 上可以替换。但语义上，文本 pad token embedding 并不是视觉 encoder 输出分布中的“空视觉 patch”。当前 norm 统计也表明它和真实视觉 token 差距很大。

更合理的候选包括：

- 全图视觉 token 均值
- 非选中视觉 token 均值
- learned null visual token
- 同分布噪声或 blur/crop 后重新走 vision tower 得到的视觉 token

当前已有 `mean_visual_embedding` 策略，建议优先与 `pad_token_embedding` 对比。

### 4.3 `Attend` 的 mask-origin 映射基于 center crop 假设

`Attend/preprocessor.py` 的 `build_center_crop_preprocess_meta()` 假设 LLaVA/CLIP 预处理为 shortest-edge resize 加 center crop，并在 `mask_mapper.py` 中用该假设映射回原图。

当前输入给 Attend 的图是 512x512，所以这个问题暂时不明显。但如果后续使用非正方形原图，mask 回原图的位置可能出现偏差。

## 5. 建议的排查顺序

1. 先统一 Attend 与 Respond 的视觉 feature 策略。
   - 推荐先固定为 Respond 实际使用的 `default`，使两个阶段都以 576 个 patch token 为准。
   - 或者全链路固定为 `full`，但必须确认 Respond 的 image placeholder 数量、visual embedding 数量、selected index offset 全部一致。

2. 对比三种 padding 策略的 decode trace。
   - `pad_token_embedding`
   - `zero_embedding`
   - `mean_visual_embedding`
   重点看 `top_pad` 是否仍然大量出现 `ms/s/as/ss` 这类异常碎片 token。

3. 做一个 alpha sweep。
   - 建议测试 `alpha = 0, 0.1, 0.3, 0.5, 1.0`
   - `alpha=0` 应接近 regular 分支，可作为 sanity check。

4. 明确 Respond 输入图。
   - 若目标是论文式原图问答，建议用原始图或 `outputs/envision/run_001/original.png`。
   - 若目标是评估 diffusion reconstruction 后的链路，则报告中要单独说明。

5. 检查 mask 可视化与问题语义是否一致。
   - 对 “How many legs...” 这类计数问题，mask 应覆盖腿/身体边界等关键区域。
   - 如果 mask 大量落在背景、伪影或非目标区域，应回到 Envision impression 与 Attend layer/top ratio 调参。

6. 复跑 Envision 多个 seed，观察 uncertainty 与 impression 稳定性。
   - 当前 run_001 与 run_003 uncertainty 动态范围差异较大。
   - 建议固定输入问题后记录 mask IoU、selected patch 分布、最终答案稳定性。

## 6. 总体结论

当前三阶段文件链路是通的，现有 run 也没有明显的路径缺失或 shape 崩溃。但“效果不理想”的核心原因很可能不在单个阶段是否能运行，而在三处结构性偏差：

1. Attend 与 Respond 对视觉 token 布局的配置不一致。
2. Respond padded 分支分布异常，`pad_token_embedding` 替换视觉 token 后产生了明显不自然的 logits。
3. Attend mask 空间分布过散，仍有不少 patch 落在天空、地面和图像边缘等与“数腿”弱相关区域。

优先建议先验证 Respond padded 分支是否合理，再统一视觉 token 布局。否则后续即使调整 Envision 或 Attend 的 top ratio，也可能只是围绕一个失真的 contrastive branch 调参。

## 7. 2026-06-05 复查补充

本次复查基于新的 `outputs/envision/run_001`、`outputs/attend/run_001`、`outputs/respond/run_001`。用户已经将 Attend/Respond 统一到 Envision 的 `original.png`。

### 7.1 已修正：Respond 使用重建图的问题

新 run 的配置显示：

```text
Attend original_image: outputs/envision/run_001/original.png
Respond image_path:    outputs/envision/run_001/original.png
Envision input_image:  EnAR/Envision/image/data/wolf_5.png
```

因此上一版报告中“最终问答使用 `reconstruction_no_perturb.png`”的问题已经不再成立。现在 regular answer 与 EnAR answer 都是在同一张 `original.png` 上产生的：

```text
regular_answer: The wolf in the image has four legs.
enar_answer:    The wolf has four legs.
```

这说明当前“效果不好”的根因需要继续往 Attend mask 与 Respond contrastive branch 查。

### 7.2 仍未修正：Attend/Respond 视觉 token 策略不一致

新 run 中：

```yaml
Attend:
  vision_feature_select_strategy: full
  num_additional_image_tokens: 1

Respond:
  vision_feature_select_strategy: default
  num_additional_image_tokens: 1
```

`Attend` 仍记录 `has_cls_token: true`，并保存了相对 patch index `+1` 后的 `selected_vision_token_indices`。但 `Respond` 实际提取到的视觉 embedding 为 576 个 token，`visual_token_layout.has_cls_token` 被判定为 `false`，最终使用的是未加 offset 的 patch index。

这次没有直接出错，是因为 `Respond/visual_embeddings.py` 自动把索引修回 patch index。但它仍然说明两个阶段对同一批视觉 token 的定义不一致。这个问题优先级仍然是高。

### 7.3 仍未修正：padded 分支 logits 明显异常

新 run 的 padded 替换统计：

```text
replaced_count: 58 / 576
before_norm: min 19.7031, max 73.8750, mean 26.7893
after_norm:  min 0.1509,  max 0.1509,  mean 0.1509
padding_strategy: pad_token_embedding
```

也就是说，真实视觉 token 被替换成了 norm 极小且完全一致的 pad token embedding。decode trace 中 padded 分支依旧反复偏向异常碎片 token：

```text
step 0 top_pad: ms, s, m, </s>, ss
step 1 top_pad: ms, m, s, ss, </s>
step 2 top_pad: ms, ss, m, </s>, ass
step 4 top_pad: ms, m, ss, </s>, as
```

这与上一版排查一致：当前 padded branch 并不像“遮挡关键视觉证据后的 VLM”，而更像一个被 OOD visual embedding 扰乱后的语言分布。只要这个分支不正常，contrastive decoding 就很难稳定改善效果。

### 7.4 新发现：Attend mask 与“数腿”问题语义对齐不足

新 run 的 Attend 选择结果：

```text
selected_total: 58
attention_only: 30
uncertainty_only: 14
attention_and_uncertainty: 14
```

相比上一版，attention 与 uncertainty 的重叠从 7 个 patch 增加到 14 个 patch，说明统一 original 后一致性有所改善。但可视化与坐标统计显示，mask 仍然过散：

```text
selected bbox: rows 0-23, cols 0-23
selected rows:
  row 20: 8 patches
  row 21: 12 patches
  row 22: 13 patches
```

mask 的确覆盖了一部分脚和腿，但也明显覆盖了鼻子、耳朵、天空、地面、图像右侧边界等区域。对于问题 `How many legs does this wolf have?`，理想 mask 应该更集中在腿部、脚部、身体下沿与遮挡边界，而不是跨整张图。

这说明 Attend 当前选出的 patch 并不完全是问题相关的 counterfactual evidence，而是混合了：

- 原图与 impression 的全局生成差异；
- diffusion 造成的背景/边缘变化；
- 不确定性图中的边界/纹理噪声；
- 与问题无关但 attention 或 uncertainty 较高的区域。

### 7.5 新发现：impression 与 original 的语义/背景变化仍偏大

当前 `impression.png` 中狼的姿态和背景比 original 更“重绘化”：背景变得更干净、颜色和地平线明显变化，狼的纹理与轮廓也发生了全局变化。`difference.png` 和 `uncertainty_heatmap.png` 显示差异并不只集中在腿部，而是沿轮廓、背景和局部纹理广泛出现。

这会导致 Attend 的 contrastive attention 学到“原图 vs 生成图”的差异，而不是“数腿问题所需证据 vs 反事实缺失证据”的差异。当前 impression 的生成目标没有绑定 question，因此对计数类问题不一定能产生合适的反事实。

### 7.6 更新后的建议顺序

1. 先修 Respond padded branch。
   - 优先把 `padding_strategy` 从 `pad_token_embedding` 改为 `mean_visual_embedding` 做对照。
   - 判断标准不是只看最终答案，而是看 `decode_trace.json` 中 `top_pad` 是否恢复成自然 token 分布。

2. 再统一视觉 token 策略。
   - 推荐先让 Attend 与 Respond 都使用 `default`，全链路以 576 个 patch token 为准。
   - 同时让 Attend 输出中的 `selected_vision_token_indices` 与 Respond 实际使用值一致，避免日志误导。

3. 调整 Attend mask 的选择方式。
   - 降低 `padding_ratio_limit`，例如从 `0.10` 测到 `0.05`。
   - 对 `attention_top_ratio` 和 `uncertainty_top_ratio` 做小范围 sweep。
   - 每次以 `mask_origin_three_color_overlay.png` 为准确认是否集中在腿部/脚部。

4. 检查 Envision impression 的可用性。
   - 如果 impression 主要改变背景和纹理，而不是问题相关结构，Attend 会被带偏。
   - 对计数问题，可能需要问题条件化的 impression 或更保守的 representative 选择，而不是简单选择 L2 最远样本。

5. 做一个最小 sanity check。
   - `alpha=0` 应基本复现 regular answer。
   - 小 `alpha` 如 `0.1/0.3` 应比 `alpha=1.0` 更稳定。
   - 若 `alpha=0` 与 regular 不一致，说明手动 `inputs_embeds` 路径与 `model.generate(pixel_values=...)` 路径仍有实现差异，需要优先排查 `DualBranchForwarder`。
