# 4.2 Attend 阶段模块化实现设计

## 1. 阶段目标

Attend 阶段负责比较原始图像 `V` 与视觉印象 `V_hat` 在 LVLM vision encoder 中的注意力分布差异，并结合 Envision 阶段产生的不确定性图 `U`，定位可能导致反事实幻觉的视觉 token。

阶段输入：

- YAML 配置文件，统一管理原始图像、视觉印象、不确定性图、模型路径、输出路径和 token 选择参数。
- 原始图像。
- Envision 阶段生成的视觉印象。
- Envision 阶段生成的不确定性图。
- LLaVA-1.5-7B 本地模型路径：`EnAR/pre_model/LLM/llava-1.5-7b-hf/`。
- 目标 vision encoder 层数，默认第 6 层。
- padding ratio，默认 10%。

阶段输出：

- 反事实 token index 列表。
- attention 差异图。
- uncertainty patch 分数图。
- 最终 patch mask。
- 映射回原图尺寸的并集 mask：`mask_origin.png`。
- 原图 overlay 可视化。
- `attend_result.json`。

## 2. 总体流程

```text
Envision 输出
  -> 加载 LLaVA processor 与 vision encoder
  -> 原图和视觉印象使用同一预处理
  -> 提取第 L 层 vision attention
  -> 计算 contrastive attention DeltaA
  -> 将 uncertainty map 映射到 patch 网格
  -> 选择 attention top-K% token
  -> 选择 uncertainty top-5% token
  -> 合并并受 padding ratio 限制
  -> 将并集 mask 映射回原图尺寸
  -> 输出 token index、mask_origin 与 patch overlay
```

## 3. 模块设计

### 3.1 配置模块 `AttendConfig`

职责：

- 使用 YAML 文件管理 Attend 阶段的输入输出路径、模型路径、token 选择参数和可视化参数。
- 对 YAML 中的相对路径进行规范化，推荐相对 `EnAR/` 项目根目录解析。
- 校验必填字段是否存在，避免运行到中间阶段才发现路径缺失。

建议字段：

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `llava_model_dir` | path | `EnAR/pre_model/LLM/llava-1.5-7b-hf` | 本地 LLaVA 模型 |
| `original_image` | path | 必填 | 原图路径 |
| `impression_image` | path | 必填 | 视觉印象路径 |
| `uncertainty_map` | path | 必填 | `.npy` 或图像格式的不确定性图 |
| `output_dir` | path | 必填 | 输出目录 |
| `vision_layer` | int | 6 | 提取注意力的层 |
| `attention_top_ratio` | float | 0.10 | attention 候选比例 |
| `uncertainty_top_ratio` | float | 0.05 | uncertainty 候选比例 |
| `padding_ratio_limit` | float | 0.10 | 最终 token 数上限 |
| `vision_feature_select_strategy` | str | `full` | 当前本地脚本建议使用 full |
| `num_additional_image_tokens` | int | 1 | 与 full 策略匹配 |
| `mask_origin_mode` | str | `binary` | 原图 mask 输出模式，可选 `binary` 或 `overlay` |
| `mask_origin_alpha` | float | 0.45 | overlay 模式下 mask 透明度 |

输入：

- YAML 配置文件路径，例如 `EnAR/Attend/attend_config.yaml`。
- 可选命令行 override，仅用于临时实验，不作为主要配置来源。

输出：

- 规范化后的配置对象。

建议 YAML 结构：

```yaml
paths:
  llava_model_dir: EnAR/pre_model/LLM/llava-1.5-7b-hf
  original_image: EnAR/outputs/envision/demo/original.png
  impression_image: EnAR/outputs/envision/demo/impression.png
  uncertainty_map: EnAR/outputs/envision/demo/uncertainty_map.npy
  envision_metadata: EnAR/outputs/envision/demo/metadata.json
  output_dir: EnAR/outputs/attend/demo

model:
  vision_feature_select_strategy: full
  num_additional_image_tokens: 1
  device: cuda
  dtype: float16

attention:
  vision_layer_number: 6
  attention_top_ratio: 0.10
  uncertainty_top_ratio: 0.05
  padding_ratio_limit: 0.10
  uncertainty_weight: 1.0

visualization:
  save_raw_arrays: true
  save_heatmaps: true
  save_patch_overlay: true
  save_mask_origin: true
  mask_origin_mode: binary
  mask_origin_alpha: 0.45
```

字段约定：

- `vision_layer_number` 使用论文中的自然层号，第 6 层写作 `6`；实现时内部转换为 0-based index `5`。
- `envision_metadata` 推荐填写，用于把 LLaVA 336 坐标系下的 patch mask 更准确地映射回原图坐标。
- 如果 `device` 写 `auto`，实现时可自动选择 CUDA 或 CPU。

### 3.2 YAML 配置加载模块 `AttendYamlConfigLoader`

职责：

- 读取 YAML 配置文件并生成 `AttendConfig`。
- 校验路径、数值范围和字段依赖关系。
- 将配置快照复制到输出目录，保证实验可复现。

输入：

- `config_yaml` 路径。
- 可选 `project_root`，默认 `/home/qianustb` 或运行时推断出的项目根目录。

输出：

- `AttendConfig`。
- `resolved_config.yaml`，保存到输出目录。

实现要点：

- 使用 `yaml.safe_load` 读取配置。
- 对 `paths.*` 中的相对路径统一解析为绝对路径。
- 校验 `attention_top_ratio`、`uncertainty_top_ratio`、`padding_ratio_limit` 均在 `(0, 1]`。
- 校验 `vision_layer_number >= 1`，且不超过 vision encoder 层数。
- 校验 `original_image`、`impression_image`、`uncertainty_map` 必须存在。
- 输出目录不存在时由 pipeline 创建。

### 3.3 LLaVA 加载模块 `LlavaVisionLoader`

职责：

- 加载 LLaVA processor 和模型。
- patch processor 配置，避免图像 token 数不匹配。
- 暴露 vision tower 和图像预处理接口。

输入：

- `llava_model_dir`。
- `vision_feature_select_strategy`。
- `num_additional_image_tokens`。

输出：

- `processor`。
- `model`。
- `vision_tower`。
- `vision_config`。

实现要点：

- 可复用 `EnAR/work_scripts/test_LLaVA/test_load_llava_1_5_7b.py` 中的 `patch_processor_from_config` 思路。
- 从 `config.json` 读取：
  - `vision_config.image_size = 336`
  - `vision_config.patch_size = 14`
  - `vision_config.num_hidden_layers = 24`
  - `pad_token_id = 32001`
- 推导 patch 网格为 `24 x 24`。

### 3.4 图像对齐预处理模块 `LlavaImagePreprocessor`

职责：

- 对原图和视觉印象执行完全一致的 LLaVA image preprocessing。
- 确保两个输入进入 vision encoder 后 token 位置可逐一对齐。

输入：

- `original_image`。
- `impression_image`。
- `processor`。

输出：

- `pixel_values_original`。
- `pixel_values_impression`。
- `preprocess_meta`。

实现要点：

- 不要分别使用不同 resize/crop 策略。
- 如果 Envision 的视觉印象已经是 512 尺寸，进入 LLaVA 前仍由 LLaVA processor 转为 336。
- 记录实际输入 vision encoder 的尺寸和归一化配置。
- `preprocess_meta` 需要包含从原图到 LLaVA 输入图的 resize/crop/pad 信息，供 `MaskOriginMapper` 将 patch mask 回映射到原图。

### 3.5 Attention 提取模块 `VisionAttentionExtractor`

职责：

- 对图像输入执行 vision encoder 前向。
- 提取指定层 `L` 的 self-attention。
- 将 attention 转为 patch token 分数。

输入：

- `pixel_values`。
- `vision_tower`。
- `vision_layer`。

输出：

- `attention_scores`: shape 约为 `[num_patches]`。
- `raw_attention`: 原始 attention tensor，可选保存。
- `token_layout_meta`: 是否含 cls token、patch 数、网格尺寸。

实现策略：

- 对 LLaVA-1.5-7B 的 CLIP vision encoder，优先使用 `cls -> patch` attention。
- 如果输出 token 数是 577，则第 0 个 token 视为 cls，后 576 个为 patch。
- 如果输出 token 数是 576，则说明 cls 已被移除，需要改用 incoming attention 或检查 `vision_feature_select_strategy`。

注意：

- 论文中的第 6 层需要确认 indexing 方式。实现中建议用 0-based index `5` 表示第 6 层，并在配置中明确字段语义，例如 `vision_layer_index=5` 或 `vision_layer_number=6`，避免混淆。

### 3.6 对比注意力模块 `ContrastiveAttentionComputer`

职责：

- 计算原图与视觉印象 attention map 的绝对差异。

输入：

- `attention_scores_original`。
- `attention_scores_impression`。

输出：

- `delta_attention_scores`。
- `delta_attention_grid`: shape `[24, 24]`。
- 归一化可视化图。

实现公式：

```text
DeltaA = abs(Attn_L(V) - Attn_L(V_hat))
```

实现要点：

- 两个 attention score 必须 token 数一致。
- 建议对 `DeltaA` 做 min-max normalization 仅用于可视化；token 排序使用原始差值或稳定归一化后的值均可，但要写入 metadata。

### 3.7 不确定性映射模块 `UncertaintyPatchMapper`

职责：

- 将 Envision 阶段的像素级不确定性图映射到 LLaVA patch 网格。

输入：

- `uncertainty_map`。
- LLaVA vision 输入尺寸 `336 x 336`。
- patch size `14`。
- 可选 Envision transform meta。

输出：

- `uncertainty_patch_scores`: shape `[576]`。
- `uncertainty_patch_grid`: shape `[24, 24]`。

实现策略：

- 将 uncertainty map resize 到 LLaVA image size，即 `336 x 336`。
- 按 `14 x 14` patch 分块平均池化，得到 `24 x 24` patch score。
- 如果输入 uncertainty map 来自非方形原图，应优先按照 Envision metadata 的 crop/resize 信息对齐。

### 3.8 Token 选择模块 `CounterfactualTokenSelector`

职责：

- 根据 contrastive attention 和 uncertainty patch scores 选择候选反事实 token。
- 按论文合并 `Hattn` 与 `Hunc`，并受 padding 比例上限约束。

输入：

- `delta_attention_scores`。
- `uncertainty_patch_scores`。
- `attention_top_ratio`。
- `uncertainty_top_ratio`。
- `padding_ratio_limit`。

输出：

- `h_attn`。
- `h_unc`。
- `h_union_raw`。
- `h_final`。
- `union_patch_mask_grid`: shape `[24, 24]`，最终并集 mask。

实现细节：

- `num_patches = 576`。
- `attention_top_k = ceil(num_patches * attention_top_ratio)`。
- `uncertainty_top_k = ceil(num_patches * uncertainty_top_ratio)`。
- `padding_limit = ceil(num_patches * padding_ratio_limit)`。
- 合并后如果超过 `padding_limit`，建议按综合分数截断：

```text
score = normalize(DeltaA) + lambda_u * normalize(U_patch)
```

初版 `lambda_u = 1.0`。

token index 约定：

- `patch_index` 范围为 `[0, 575]`，表示纯 patch index。
- 如果后续要写入包含 cls 的 vision token 序列，则实际 vision token index 可能是 `patch_index + 1`。
- 输出 JSON 中必须同时保存 `patch_indices` 和 `vision_token_indices`。

### 3.9 原图 Mask 映射工具模块 `MaskOriginMapper`

职责：

- 将 `CounterfactualTokenSelector` 得到的并集 patch mask 从 LLaVA patch 网格映射回原始图像坐标。
- 输出一张与原图同尺寸的二值 mask 图片 `mask_origin.png`。
- 可选输出叠加原图的 `mask_origin_overlay.png`，便于人工检查反事实区域是否落在正确位置。

输入：

- `union_patch_mask_grid`: shape `[24, 24]`。
- 原始图像路径或原始图像尺寸。
- `preprocess_meta`。
- 可选 `envision_metadata`，用于补充 Envision 阶段的 resize/crop 信息。
- `mask_origin_mode`。
- `mask_origin_alpha`。

输出：

- `mask_origin`: 与原图尺寸一致的二值 mask array。
- `mask_origin.png`: 白色表示被选中的并集区域，黑色表示未选中区域。
- `mask_origin_overlay.png`: 可选，将 mask 半透明叠加到原图。
- `mask_origin_meta`: 回映射使用的尺寸、缩放比例、裁剪偏移、是否存在近似映射。

实现流程：

1. 将 `[24, 24]` 的 patch mask 按 patch size `14` 展开为 LLaVA 输入坐标系下的 `336 x 336` mask。
2. 根据 `LlavaImagePreprocessor` 记录的 `preprocess_meta`，把 `336 x 336` mask 逆变换回原图坐标。
3. 如果 LLaVA processor 使用 center crop，需要先把 crop 区域放回 resize 后画布，再逆 resize 到原图尺寸。
4. 如果 processor 使用 padding 或 shortest-edge resize，需要按 metadata 中的 pad offset 和 scale 还原。
5. 保存 `mask_origin.png`，并在 metadata 中记录该映射是精确逆变换还是基于默认 center-crop 假设的近似映射。

实现要点：

- `mask_origin.png` 必须和原始输入图像宽高完全一致。
- 二值 mask 建议使用 `uint8`，取值 `0` 或 `255`。
- 最近邻插值用于 mask resize，避免产生灰边。
- 如果无法从 processor 自动拿到完整几何变换，初版可明确采用与 LLaVA 默认预处理一致的 center-crop 假设，并在 `attend_result.json` 中写入 `mask_origin_mapping_assumption`。
- 该模块属于工具模块，不参与 token 选择，只负责把已选并集 mask 转换为便于观察和后续处理的原图空间 mask。

### 3.10 可视化模块 `AttendVisualizer`

职责：

- 将 attention 差异、不确定性 patch 分数和最终 mask 可视化。
- 在原图上叠加被选中的 patch 区域。

输入：

- 原图。
- `delta_attention_grid`。
- `uncertainty_patch_grid`。
- `h_final`。
- patch grid size。

输出：

- `contrastive_attention_heatmap.png`。
- `uncertainty_patch_heatmap.png`。
- `selected_patch_mask.png`。
- `mask_origin.png`。
- `mask_origin_overlay.png`。
- `patch_overlay.png`。

实现要点：

- patch overlay 需要明确每个 patch 在 LLaVA 336 输入上的位置。
- 为了便于人工检查，可以在 overlay 上用红框标出最终 selected patches。
- `mask_origin.png` 由 `MaskOriginMapper` 产生，`AttendVisualizer` 只负责把它纳入可视化报告或生成 overlay。

### 3.11 输出管理模块 `AttendOutputWriter`

职责：

- 保存 Attend 阶段所有产物。
- 提供 Respond 阶段可以直接读取的标准 JSON。

输出目录建议：

```text
outputs/attend/{run_id}/
  resolved_config.yaml
  contrastive_attention.npy
  contrastive_attention_heatmap.png
  uncertainty_patch_scores.npy
  uncertainty_patch_heatmap.png
  selected_patch_mask.png
  mask_origin.png
  mask_origin_overlay.png
  patch_overlay.png
  attend_result.json
```

`attend_result.json` 建议包含：

```json
{
  "original_image": "...",
  "impression_image": "...",
  "uncertainty_map": "...",
  "vision_layer_number": 6,
  "patch_grid": [24, 24],
  "patch_size": 14,
  "has_cls_token": true,
  "h_attn_patch_indices": [],
  "h_unc_patch_indices": [],
  "selected_patch_indices": [],
  "selected_vision_token_indices": [],
  "mask_origin_path": "...",
  "mask_origin_overlay_path": "...",
  "mask_origin_mapping_meta": {},
  "attention_top_ratio": 0.1,
  "uncertainty_top_ratio": 0.05,
  "padding_ratio_limit": 0.1
}
```

## 4. 主控流程模块 `AttendPipeline`

职责：

- 串联 YAML 配置加载、模型加载、图像预处理、attention 提取、uncertainty 映射、token 选择、原图 mask 映射和可视化。

输入：

- YAML 配置文件路径。
- `AttendConfig`，由 `AttendYamlConfigLoader` 生成。

输出：

- `AttendResult`：
  - `selected_patch_indices`
  - `selected_vision_token_indices`
  - `mask_origin_path`
  - `patch_overlay_path`
  - `attend_result_json`

执行顺序：

1. 读取 YAML 配置文件，生成并校验 `AttendConfig`。
2. 将解析后的 `resolved_config.yaml` 保存到输出目录。
3. 加载 LLaVA processor 与 vision tower。
4. 预处理原图与视觉印象，并记录 `preprocess_meta`。
5. 分别提取第 6 层 attention。
6. 计算 `DeltaA`。
7. 映射 uncertainty 到 patch grid。
8. 选择候选 token，得到最终并集 patch mask。
9. 使用 `MaskOriginMapper` 将并集 mask 映射回原图，生成 `mask_origin.png`。
10. 生成 heatmap、patch overlay 和 mask overlay 可视化。
11. 保存 `attend_result.json`。

## 5. 验收与调试

最小验收：

- 原图和视觉印象均能得到 shape 一致的 attention scores。
- `DeltaA` 可 reshape 为 `24 x 24`。
- uncertainty map 可映射为 `24 x 24`。
- 输出非空 selected token indices。
- `mask_origin.png` 与原图尺寸完全一致。
- `patch_overlay.png` 能显示选中 patch。

关键调试项：

- 打印 vision token 数：576 或 577。
- 打印是否含 cls token。
- 保存第 3、6、9、12 层 attention heatmap，用于验证第 6 层是否合理。
- 检查 selected patch 是否集中在图像主体或变化区域，而不是全在边缘背景。
- 检查 `mask_origin.png` 的宽高是否等于原图宽高。
- 检查 `mask_origin_overlay.png` 是否与 `patch_overlay.png` 在空间位置上大致一致。

失败现象与处理：

| 现象 | 可能原因 | 处理 |
| --- | --- | --- |
| token 数为 575 | processor 配置不匹配 | 沿用现有测试脚本的 `full` 策略与 additional image token patch |
| `DeltaA` 全零 | 原图和视觉印象过于接近，或 attention 提取错误 | 检查输入图是否不同，检查 layer index |
| mask 大量落在背景 | 层数过深或 uncertainty 过噪 | 尝试第 3/6/9 层对比，降低 uncertainty 权重 |
| patch overlay 坐标偏移 | resize/crop 元数据未对齐 | 统一使用 LLaVA processor 后的 336 坐标系 |
| `mask_origin.png` 尺寸不等于原图 | 逆变换流程缺失原图尺寸 | 从原图重新读取尺寸，并在 `preprocess_meta` 中强制记录 |
| `mask_origin` 与 overlay 位置不一致 | LLaVA 预处理几何假设错误 | 保存 processor resize/crop 参数，必要时采用显式自定义预处理 |

## 6. 与前后阶段的接口

来自 Envision 的输入：

```text
original_image
impression_image
uncertainty_map
transform_meta
```

配置入口：

```text
attend_config.yaml
  -> paths.original_image
  -> paths.impression_image
  -> paths.uncertainty_map
  -> paths.output_dir
  -> attention.*
  -> visualization.*
```

提供给 Respond 的输出：

```text
selected_patch_indices
selected_vision_token_indices
patch_grid
has_cls_token
padding_ratio_limit
```

提供给人工检查或后续图像级分析的输出：

```text
mask_origin.png
mask_origin_overlay.png
mask_origin_mapping_meta
```

Respond 阶段不应重新计算 attention，只读取 Attend 输出的 token indices 并构造 padded visual input。
