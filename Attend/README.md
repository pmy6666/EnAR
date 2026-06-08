# EnAR Attend Stage

Attend 阶段用于比较原图 `V` 与 Envision 视觉印象 `V_hat` 在 LLaVA vision encoder 中的注意力差异，并结合 Envision 产生的不确定性图 `U`，定位可能导致反事实幻觉的视觉 patch/token。

## 目录结构

```text
Attend/
  config.py                 # AttendConfig 数据结构与 YAML 字段校验
  yaml_loader.py            # YAML 加载、路径解析、resolved_config 保存
  model_loader.py           # LLaVA processor/model/vision tower 加载
  preprocessor.py           # 原图与视觉印象的一致 LLaVA 预处理
  attention_extractor.py    # 指定 vision 层 self-attention 提取
  contrastive.py            # DeltaA = abs(Attn(V) - Attn(V_hat))
  uncertainty_mapper.py     # uncertainty map -> 24x24 patch scores
  token_selector.py         # Hattn/Hunc 合并与 padding ratio 截断
  mask_mapper.py            # patch mask 回映射到原图 mask_origin.png
  visualizer.py             # heatmap、patch mask、overlay 输出
  output_writer.py          # npy/json 输出管理
  pipeline.py               # AttendPipeline 主流程
  cli.py                    # 命令行入口
  tests/                    # 每个核心模块的单元测试
```

## 依赖

当前 `EnAR/env/` 里已经能看到多数依赖：`torch`、`transformers`、`accelerate`、`safetensors`、`sentencepiece`、`protobuf`、`pillow`、`numpy`、`PyYAML`、`pytest`。

如需补装，使用统一环境：

```bash
/home/qianustb/EnAR/env/bin/python -m pip install -U \
  torch torchvision transformers accelerate safetensors sentencepiece protobuf \
  pillow numpy PyYAML pytest
```

如果只运行单元测试，不加载 LLaVA 大模型，核心依赖为：

```bash
/home/qianustb/EnAR/env/bin/python -m pip install -U numpy pillow PyYAML pytest
```

## 配置

示例配置在 `Attend/attend_config.yaml`：

```yaml
paths:
  llava_model_dir: EnAR/pre_model/LLM/llava-1.5-7b-hf
  original_image: EnAR/outputs/envision/demo/preprocessed.png
  impression_image: EnAR/outputs/envision/demo/impression.png
  uncertainty_map: EnAR/outputs/envision/demo/uncertainty_map.npy
  envision_metadata: EnAR/outputs/envision/demo/metadata.json
  output_dir: EnAR/outputs/attend/demo
model:
  vision_feature_select_strategy: full
  num_additional_image_tokens: 1
  device: auto
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
  save_source_masks: true
  mask_origin_mode: binary
  mask_origin_alpha: 0.45
```

相对路径默认按项目根目录的上一级工作目录解析，例如 `/home/qianustb` 下的 `EnAR/...`。

## 运行

在 `/home/qianustb/EnAR` 下运行：

```bash
PYTHONPATH=/home/qianustb/EnAR \
/home/qianustb/EnAR/env/bin/python -m Attend.cli \
  --config /home/qianustb/EnAR/Attend/attend_config.yaml
```

也可以临时覆盖配置：

```bash
PYTHONPATH=/home/qianustb/EnAR \
/home/qianustb/EnAR/env/bin/python -m Attend.cli \
  --config /home/qianustb/EnAR/Attend/attend_config.yaml \
  --vision_layer_number 6 \
  --attention_top_ratio 0.10 \
  --uncertainty_top_ratio 0.05 \
  --padding_ratio_limit 0.10
```

## 输出

默认输出到 `paths.output_dir`：

```text
resolved_config.yaml
contrastive_attention.npy
contrastive_attention_heatmap.png
uncertainty_patch_scores.npy
uncertainty_patch_heatmap.png
selected_patch_mask.png
selected_patch_source_mask.png
mask_origin.png
mask_origin_label.png
mask_origin_color.png
mask_origin_overlay.png
mask_origin_three_color_overlay.png
patch_overlay.png
patch_source_overlay.png
source_label_grid.npy
attend_result.json
```

三色 source mask 的 label 约定：

```text
0 = background
1 = attention only，红色 #FF3030
2 = uncertainty only，蓝色 #2F80FF
3 = attention and uncertainty，黄色 #FFD23F
```

旧的二值输出仍保留：`selected_patch_mask.png`、`patch_overlay.png`、`mask_origin.png` 和 `mask_origin_overlay.png` 继续表示最终被 pad 的 union 区域。三色输出只用于 debug 和人工检查来源。

`attend_result.json` 中同时保存纯 patch index 和包含 CLS 偏移的 vision token index：

```json
{
  "selected_patch_indices": [0, 1],
  "selected_vision_token_indices": [1, 2],
  "source_counts": {
    "attention_only": 1,
    "uncertainty_only": 0,
    "attention_and_uncertainty": 1,
    "selected_total": 2
  },
  "mask_origin_path": ".../mask_origin.png",
  "patch_grid": [24, 24],
  "patch_size": 14
}
```

## 测试

单元测试不需要加载 LLaVA-1.5-7B：

```bash
cd /home/qianustb/EnAR
PYTHONPATH=/home/qianustb/EnAR \
/home/qianustb/EnAR/env/bin/python -m pytest Attend/tests
```

这些测试覆盖配置解析、attention tensor 转 patch score、contrastive attention、不确定性 patch pooling、token 选择、三色 source label、mask 回映射和可视化文件输出。

## 注意事项

- `vision_layer_number` 使用论文自然层号，`6` 表示第 6 层；代码内部转换为 0-based index `5`。
- LLaVA/CLIP vision encoder 常见 token 数为 `577 = 1 CLS + 576 patch`，输出 JSON 中的 `selected_vision_token_indices` 默认是 `patch_index + 1`。
- 当前 Attend 的基准图像建议使用 Envision 输出的 `preprocessed.png`，与 `impression.png` 和 `uncertainty_map.npy` 保持同一视觉坐标系。`mask_origin.png` 回映射采用 LLaVA/CLIP 默认 center-crop 近似假设，并在 `attend_result.json` 的 `mask_origin_mapping_meta` 中记录。
