# Attend 三色 Pad Mask 可视化改造计划

## 1. 背景与目标

当前 Attend 实现已经能分别计算：

- `h_attn_patch_indices`: 来自 `attention_top_ratio` 的 attention top patch。
- `h_unc_patch_indices`: 来自 `uncertainty_top_ratio` 的 uncertainty top patch。
- `h_union_raw_patch_indices`: 二者并集。
- `selected_patch_indices`: 经过 `padding_ratio_limit` 截断后的最终 patch。

但当前可视化只使用 `selection.union_patch_mask_grid`，因此 `selected_patch_mask.png`、`patch_overlay.png`、`mask_origin.png`、`mask_origin_overlay.png` 都是单一 union mask，无法区分某个 patch 是由 attention 命中、uncertainty 命中，还是两者共同命中。

本次需求是：对 pad 回原图的 mask 做三色区分。

颜色语义建议：

| 类别 | label | 含义 | 建议颜色 |
| --- | --- | --- | --- |
| attention only | 1 | 只来自 `attention_top_ratio` | 红色 `#FF3030` |
| uncertainty only | 2 | 只来自 `uncertainty_top_ratio` | 蓝色 `#2F80FF` |
| attention and uncertainty | 3 | 同时来自 attention 与 uncertainty | 黄色 `#FFD23F` |

保持兼容原则：

- 现有二值 union mask 仍然保留，用于 Respond 的 pad token 选择。
- 新增三值 label mask 只用于 debug、可视化和原图回映射。
- `selected_patch_indices` 的语义不变，仍表示最终会被 pad 的 patch。

## 2. 当前代码路径梳理

### 2.1 Token 选择

文件：`EnAR/Attend/token_selector.py`

当前 `CounterfactualTokenSelector.select()` 做了：

1. 通过 `attention_top_ratio` 得到 `h_attn`。
2. 通过 `uncertainty_top_ratio` 得到 `h_unc`。
3. 求并集 `union_raw`。
4. 若超过 `padding_ratio_limit`，按 combined score 截断为 `h_final`。
5. 只生成一个 bool 类型的 `union_patch_mask_grid`。

问题：

- 虽然 dataclass 已保留 `h_attn` 和 `h_unc`，但没有产出三类 label grid。
- 若 `padding_ratio_limit < len(union_raw)`，需要明确三色图到底展示 raw union 还是 final selected union。建议展示 final selected，因为它对应真正 pad 的区域。

### 2.2 原图映射

文件：`EnAR/Attend/mask_mapper.py`

当前 `MaskOriginMapper.map_and_save()` 输入是 bool/binary 的 `union_patch_mask_grid`，输出：

- `mask_origin.png`: 单通道二值 mask。
- `mask_origin_overlay.png`: 红色半透明 overlay。

问题：

- `map_to_origin()` 默认把输入转成 `uint8` 后乘 255，无法保留 1/2/3 类别。
- `make_overlay()` 写死为红色，无法按类别着色。

### 2.3 Patch overlay

文件：`EnAR/Attend/visualizer.py`

当前 `AttendVisualizer.save_patch_overlay()` 只接受 binary patch mask，所有选中 patch 都是红色。

问题：

- 336 坐标系下的 patch overlay 也无法区分来源。
- 如果只改 `mask_origin`，调试时会出现 patch overlay 和 origin overlay 颜色语义不一致。

### 2.4 Pipeline 与结果输出

文件：`EnAR/Attend/pipeline.py`

当前 pipeline 只传：

```python
selection.union_patch_mask_grid
```

给 visualizer 和 mask mapper。

问题：

- 没有把三色 label mask 输出到 `attend_result.json`。
- `image_paths` 中没有三色文件路径。

## 3. 推荐改造方案

### 3.1 新增 label mask 约定

新增 patch-level label grid：

```text
0 = background
1 = attention only
2 = uncertainty only
3 = attention and uncertainty
```

生成规则：

```text
attn_final = h_attn ∩ h_final
unc_final = h_unc ∩ h_final

for each patch in h_final:
  if patch in attn_final and patch in unc_final: label = 3
  elif patch in attn_final: label = 1
  elif patch in unc_final: label = 2
```

说明：

- 使用 `h_final` 作为显示范围，确保三色 mask 和真正 pad 的 token 一致。
- 如果未来希望同时查看未被 padding ratio 保留的 raw union，可额外输出 `raw_source_label_grid`，但第一版不需要。

## 4. 模块级修改计划

### 4.1 修改 `TokenSelectionResult`

文件：`EnAR/Attend/token_selector.py`

新增字段：

```python
source_label_grid: np.ndarray
source_label_flat: np.ndarray
source_counts: dict[str, int]
```

字段含义：

- `source_label_grid`: shape `[24, 24]`，值为 `0/1/2/3`。
- `source_label_flat`: shape `[576]`，便于保存 `.npy` 或 JSON。
- `source_counts`: 统计三类 patch 数，例如：

```json
{
  "attention_only": 52,
  "uncertainty_only": 23,
  "attention_and_uncertainty": 6,
  "selected_total": 81
}
```

推荐辅助函数：

```python
def build_source_label_mask(
    h_attn: list[int],
    h_unc: list[int],
    h_final: list[int],
    num_patches: int,
) -> tuple[np.ndarray, dict]:
    ...
```

注意：

- 如果 `padding_ratio_limit` 触发截断，某些 `h_attn` 或 `h_unc` 中的 patch 可能不在 `h_final`，这些 patch 不应该显示在最终三色 pad mask 中。
- `union_patch_mask_grid` 继续保留，值等价于 `source_label_grid > 0`。

### 4.2 扩展 `MaskOriginMapper`

文件：`EnAR/Attend/mask_mapper.py`

新增能力：

1. 保留现有 `map_and_save()` 二值接口，避免破坏旧调用。
2. 新增 `map_label_and_save()`，用于三值 label mask 回映射。

建议接口：

```python
@dataclass
class LabelMaskOriginResult:
    label_mask_origin: np.ndarray
    label_mask_origin_path: str
    label_mask_origin_color_path: str
    label_mask_origin_overlay_path: str | None
    meta: dict

def map_label_and_save(
    self,
    source_label_grid: np.ndarray,
    original_image_path: str | Path,
    preprocess_meta: dict,
    output_dir: str | Path,
    save_overlay: bool = True,
) -> LabelMaskOriginResult:
    ...
```

输出建议：

- `mask_origin_label.png`: 单通道 label mask，像素值为 `0/1/2/3`。
- `mask_origin_color.png`: RGB 纯色 mask，按类别上色。
- `mask_origin_three_color_overlay.png`: 三色半透明叠加原图。

实现要点：

- label mask resize 必须使用 nearest neighbor，避免 1/2/3 被插值污染。
- 原有 `mask_origin.png` 继续输出二值 mask，供旧流程和简单检查使用。
- `mask_origin_color.png` 不能只保存 label 值，否则肉眼看几乎全黑；需要真正 RGB 着色。

颜色映射建议集中定义：

```python
SOURCE_LABEL_COLORS = {
    0: (0, 0, 0, 0),
    1: (255, 48, 48, alpha),
    2: (47, 128, 255, alpha),
    3: (255, 210, 63, alpha),
}
```

### 4.3 扩展 `AttendVisualizer`

文件：`EnAR/Attend/visualizer.py`

新增方法：

```python
def save_source_label_patch_mask(
    self,
    source_label_grid: np.ndarray,
    filename: str = "selected_patch_source_mask.png",
) -> str:
    ...

def save_source_label_patch_overlay(
    self,
    original_image_path: str | Path,
    source_label_grid: np.ndarray,
    filename: str = "patch_source_overlay.png",
) -> str:
    ...
```

输出建议：

- `selected_patch_source_mask.png`: 336 尺寸三色 patch mask。
- `patch_source_overlay.png`: 336 坐标系下三色 patch overlay。

注意：

- 旧的 `selected_patch_mask.png` 和 `patch_overlay.png` 保留，仍为二值红色 union mask。
- 新增三色图命名中带 `source`，避免和旧文件混淆。

### 4.4 扩展配置

文件：

- `EnAR/Attend/config.py`
- `EnAR/Attend/attend_config.yaml`
- `EnAR/Attend/README.md`

建议新增 YAML 字段：

```yaml
visualization:
  save_source_masks: true
  source_label_colors:
    attention_only: [255, 48, 48]
    uncertainty_only: [47, 128, 255]
    attention_and_uncertainty: [255, 210, 63]
```

对应 `AttendConfig` 新增：

```python
save_source_masks: bool = True
source_label_colors: dict | None = None
```

如果想减少配置复杂度，第一版也可以只新增 `save_source_masks`，颜色写成常量。

### 4.5 修改 `AttendPipeline`

文件：`EnAR/Attend/pipeline.py`

在 token selection 后，新增三色输出逻辑：

```python
selection = CounterfactualTokenSelector().select(...)

image_paths["selected_patch_source_mask"] = visualizer.save_source_label_patch_mask(
    selection.source_label_grid
)
image_paths["patch_source_overlay"] = visualizer.save_source_label_patch_overlay(
    self.config.original_image,
    selection.source_label_grid,
)

label_mask_result = MaskOriginMapper(...).map_label_and_save(
    selection.source_label_grid,
    self.config.original_image,
    prep.preprocess_meta,
    self.config.output_dir,
    save_overlay=True,
)
```

`result_data` 增加：

```json
{
  "source_label_encoding": {
    "0": "background",
    "1": "attention_only",
    "2": "uncertainty_only",
    "3": "attention_and_uncertainty"
  },
  "source_counts": {},
  "source_label_paths": {
    "patch_source_mask": "...",
    "patch_source_overlay": "...",
    "mask_origin_label": "...",
    "mask_origin_color": "...",
    "mask_origin_three_color_overlay": "..."
  }
}
```

保留旧字段：

- `mask_origin_path`
- `mask_origin_overlay_path`
- `patch_overlay`
- `selected_patch_mask`

这样旧的调试文档和 Respond 阶段不会受影响。

## 5. 输出文件规划

现有输出继续保留：

```text
selected_patch_mask.png
patch_overlay.png
mask_origin.png
mask_origin_overlay.png
```

新增输出：

```text
selected_patch_source_mask.png
patch_source_overlay.png
mask_origin_label.png
mask_origin_color.png
mask_origin_three_color_overlay.png
source_label_grid.npy
```

文件含义：

| 文件 | 坐标系 | 类型 | 用途 |
| --- | --- | --- | --- |
| `selected_patch_source_mask.png` | LLaVA 336 | RGB | 三色 patch mask |
| `patch_source_overlay.png` | LLaVA 336 | RGB | 三色 patch overlay |
| `mask_origin_label.png` | 原图 | L/uint8 | 原图尺寸 label mask，值为 0/1/2/3 |
| `mask_origin_color.png` | 原图 | RGB | 原图尺寸纯三色 mask |
| `mask_origin_three_color_overlay.png` | 原图 | RGB | 原图三色半透明 overlay |
| `source_label_grid.npy` | 24x24 | int | 调试和测试使用 |

## 6. 测试计划

### 6.1 Token selector 单元测试

文件：`EnAR/Attend/tests/test_uncertainty_selector.py`

新增测试：

- 构造 `delta` top 为 `[1, 2, 3, 4]`。
- 构造 `unc` top 为 `[3, 4, 5, 6]`。
- 验证：
  - patch 1/2 label 为 1。
  - patch 5/6 label 为 2。
  - patch 3/4 label 为 3。
  - label > 0 的位置等于 `h_final`。

还要覆盖 padding 截断：

- `padding_ratio_limit` 设置较小。
- 验证被截断掉的 patch 不出现在 `source_label_grid` 中。

### 6.2 Mask mapper 单元测试

文件：`EnAR/Attend/tests/test_mask_visualizer.py`

新增测试：

- 输入一个 `2 x 2` label grid：

```text
[[1, 2],
 [3, 0]]
```

- 使用 `patch_size=4`、`vision_input_size=(8, 8)`。
- 验证 `mask_origin_label.png` 存在。
- 验证输出 label mask 只有 `{0, 1, 2, 3}`。
- 验证 RGB color mask 存在。
- 验证 overlay 存在。

### 6.3 Visualizer 单元测试

新增测试：

- `save_source_label_patch_mask()` 能输出 RGB 图。
- `save_source_label_patch_overlay()` 能输出 RGB overlay。
- 三类颜色都能在输出图中找到。

### 6.4 端到端 smoke test

使用当前配置：

```bash
cd /home/qianustb/EnAR
PYTHONPATH=/home/qianustb/EnAR \
/home/qianustb/EnAR/env/bin/python -m Attend.cli \
  --config /home/qianustb/EnAR/Attend/attend_config.yaml
```

检查输出：

- `attend_result.json` 中 `h_attn_patch_indices`、`h_unc_patch_indices` 仍存在。
- `source_counts` 中三类数量之和等于 `len(selected_patch_indices)`。
- `mask_origin_label.png` 尺寸等于原图尺寸。
- `mask_origin_three_color_overlay.png` 能看到三色区域。

## 7. 风险与注意事项

1. `padding_ratio_limit` 当前配置为 `1.0`，因此 `h_final` 基本等于 raw union。若后续改回 `0.10`，三色图应只展示最终被保留的 pad 区域。
2. 三色 label mask 不应替代二值 mask。Respond 只需要知道哪些 patch 被 pad，不需要来源颜色。
3. `mask_origin_label.png` 使用像素值 1/2/3，肉眼可能接近黑色，因此必须同时输出 `mask_origin_color.png` 或 overlay。
4. 最近邻插值是硬要求，不能用 bilinear/bicubic，否则 label 会出现非法中间值。
5. 如果 attention 与 uncertainty 大量重叠，黄色区域会比较多，这是正常结果；应通过 `source_counts` 辅助判断。
6. 如果 `patch_source_overlay.png` 与 `mask_origin_three_color_overlay.png` 空间位置不一致，优先检查 `preprocess_meta` 的 crop/resize 逆变换。

## 8. 推荐实施顺序

1. 在 `token_selector.py` 中新增 `source_label_grid` 和 `source_counts`。
2. 为 token selector 增加单元测试，先确保 label 逻辑正确。
3. 在 `visualizer.py` 中增加 336 patch 坐标系三色 mask 和 overlay。
4. 在 `mask_mapper.py` 中增加 label mask 回原图与三色 overlay。
5. 修改 `pipeline.py` 连接新增输出。
6. 修改 `config.py` 和 `attend_config.yaml`，增加 `save_source_masks`。
7. 修改 `README.md` 和输出说明。
8. 跑单元测试和一次当前 `run_001` 配置的 smoke test。

## 9. 最小改动版本

如果希望先快速验证效果，可以采用最小改动：

1. 不改 YAML 配置。
2. 不改旧二值输出。
3. 只在 `TokenSelectionResult` 增加 `source_label_grid`。
4. 只在 `MaskOriginMapper` 增加 `map_label_and_save()`。
5. Pipeline 无条件输出：

```text
mask_origin_label.png
mask_origin_color.png
mask_origin_three_color_overlay.png
```

这个版本已经能满足“pad 回原图分三种颜色”的核心需求。后续再补配置开关和 336 patch source overlay。
