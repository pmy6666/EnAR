# EnAR LLaVA 重构计划

## 目标

在 `EnAR/merge/` 下重新组织一个完整的 EnAR 实现，复用 `EnAR/VCD/experiments/llava/` 中的 LLaVA 代码与权重加载逻辑，把当前 `Envision`、`Attend`、`Respond` 三阶段串成一个可运行、可验证、可复用的整体流程。

本阶段只做计划，不改动现有三阶段代码。后续实现应尽量通过适配层复用现有模块，避免直接破坏 `Envision/`、`Attend/`、`Respond/` 的既有实验脚本。

## 当前代码观察

### LLaVA 代码来源

`EnAR/VCD/experiments/llava/` 是 LLaVA 原始风格实现，核心入口包括：

- `model/builder.py`
  - `load_pretrained_model(model_path, model_base, model_name, load_8bit, load_4bit, device_map, device)`
  - 支持完整 LLaVA、LoRA、base model + `mm_projector.bin` 等加载方式。
  - 返回 `tokenizer, model, image_processor, context_len`。
- `model/llava_arch.py`
  - `encode_images(images)`：vision tower + mm projector。
  - `prepare_inputs_labels_for_multimodal(...)`：将 `<image>` token 替换成视觉 embedding。
- `mm_utils.py`
  - `process_images(...)`：使用 LLaVA image processor 预处理图像。
  - `tokenizer_image_token(...)`：把 prompt 中的 `<image>` 替换为 `IMAGE_TOKEN_INDEX`。
- `conversation.py`、`constants.py`
  - prompt 模板、`IMAGE_TOKEN_INDEX`、默认 image token 等常量。

### 现有 EnAR 三阶段

`Envision/` 当前负责基于 Stable Diffusion 的 impression image 与 uncertainty map：

- `Envision/pipeline.py`
  - 输入原图、prompt、SD 权重。
  - 输出 original image、impression image、uncertainty map、heatmap、metadata。

`Attend/` 当前负责从原图/impression 图中提取视觉 attention 差异，并与 uncertainty 融合：

- `Attend/pipeline.py`
  - 当前使用 Hugging Face `LlavaForConditionalGeneration`。
  - 读取原图、impression image、uncertainty map。
  - 输出 selected patch indices、selected vision token indices、mask、可视化、result json。
- `Attend/model_loader.py`
  - 当前直接加载 HF LLaVA，并暴露 vision tower。

`Respond/` 当前负责常规生成与 EnAR/VCD 式对比解码：

- `Respond/pipeline.py`
  - 当前同样使用 HF `LlavaForConditionalGeneration`。
  - 使用 Attend 输出选择视觉 token。
  - 构造 padded visual embeddings，与原始 visual embeddings 双分支前向。
  - 输出 regular answer、EnAR answer、decode trace。
- `Respond/dual_branch_forwarder.py`
  - 用 `inputs_embeds` 分别跑原始视觉分支和 padded 视觉分支。
- `Respond/generation_loop.py`
  - 执行 `(1 + alpha) * logits_original - alpha * logits_padded`，可选 APC。

### 需要重构的核心矛盾

当前 `Attend/Respond` 依赖 Hugging Face 新版 `LlavaForConditionalGeneration` 接口；用户要求使用 `VCD/experiments/llava/` 中的 LLaVA 代码，并加载该实现所支持的权重。因此重构主轴不是简单移动文件，而是建立一个统一的 LLaVA 适配层：

- 对外提供三阶段需要的稳定接口。
- 对内调用 `VCD/experiments/llava/model/builder.py::load_pretrained_model`。
- 屏蔽原始 LLaVA 和 HF LLaVA 在 processor、image token、visual embedding、language forward 接口上的差异。

## 目标目录设计

建议在 `EnAR/merge/` 下建立如下结构：

```text
merge/
  README.md
  llava_enar_refactor_plan.md
  config.py
  merge_config.yaml
  cli.py
  pipeline.py
  schemas.py
  llava_runtime.py
  llava_adapter.py
  envision_stage.py
  attend_stage.py
  respond_stage.py
  output_writer.py
  debug_tools.py
  tests/
    test_config.py
    test_llava_adapter_contract.py
    test_stage_contracts.py
    test_pipeline_dry_run.py
```

职责划分：

- `config.py`
  - 定义一个总配置 `MergeEnARConfig`。
  - 包含 `paths`、`llava`、`envision`、`attend`、`respond`、`runtime` 等 section。
- `schemas.py`
  - 定义三阶段之间传递的数据结构。
  - 例如 `EnvisionArtifacts`、`AttendArtifacts`、`RespondArtifacts`、`LlavaRuntimeComponents`。
- `llava_runtime.py`
  - 设置 import path，使 `from llava...` 可以解析到 `EnAR/VCD/experiments/llava`。
  - 调用 VCD LLaVA builder 加载 tokenizer、model、image_processor。
  - 统一 dtype、device、context length、special token 信息。
- `llava_adapter.py`
  - 提供 EnAR 所需的 LLaVA 稳定接口。
  - 包括图像预处理、prompt 编码、vision attention 提取、visual embeddings 提取、双分支 language forward、regular generate。
- `envision_stage.py`
  - 轻包装现有 `EnvisionPipeline`。
  - 把总配置转换为 `EnvisionConfig`。
- `attend_stage.py`
  - 复用现有 Attend 的 attention、contrastive、uncertainty、selector、visualizer 等纯逻辑模块。
  - 模型和图像编码改为走 `LlavaAdapter`。
- `respond_stage.py`
  - 复用现有 Respond 的 padding、logits processor、token selector、generation loop 思路。
  - 对模型前向和 embedding 构造改为走 `LlavaAdapter`。
- `pipeline.py`
  - 编排完整 EnAR：Envision -> Attend -> Respond。
  - 支持跳过已有阶段产物，便于调试。
- `cli.py`
  - 提供 `python -m merge.cli --config ...` 或 `python EnAR/merge/cli.py --config ...` 入口。
- `output_writer.py`
  - 统一保存 resolved config、stage artifacts、manifest、最终 answer。

## 总体数据流

```text
input image + question + prompts + weights
        |
        v
Envision
  - original_image
  - impression_image
  - uncertainty_map
  - uncertainty_heatmap
  - envision_metadata
        |
        v
Attend
  - original vision attention
  - impression vision attention
  - contrastive attention score
  - uncertainty patch score
  - selected patch indices
  - selected vision token indices
  - masks and overlays
        |
        v
Respond
  - regular LLaVA answer
  - original visual embeddings
  - padded visual embeddings
  - contrastive decoding
  - EnAR answer
  - decode trace
```

## LLaVA 适配层设计

### 加载接口

`llava_runtime.py` 中建议提供：

```python
@dataclass
class LlavaRuntimeConfig:
    model_path: Path
    model_base: Path | None
    model_name: str | None
    load_8bit: bool
    load_4bit: bool
    device: str
    device_map: str
    dtype: str

def load_llava_runtime(config: LlavaRuntimeConfig) -> LlavaRuntimeComponents:
    ...
```

实现要点：

- 在导入前把 `EnAR/VCD/experiments` 加入 `sys.path`，保证 `from llava.model import *` 指向本仓库的 LLaVA。
- `model_name` 缺省时用 `llava.mm_utils.get_model_name_from_path(model_path)`。
- 调用 `load_pretrained_model(...)` 加载权重。
- 加载完成后执行 `model.eval()`。
- 统一记录：
  - `image_token_index`
  - `mm_use_im_start_end`
  - `mm_use_im_patch_token`
  - `context_len`
  - `vision tower image_size / patch_size / select_layer / select_feature`

### Adapter 对外能力

`LlavaAdapter` 建议提供这些方法：

```python
class LlavaAdapter:
    def preprocess_image(self, image_path: Path) -> torch.Tensor: ...
    def encode_prompt(self, prompt: str) -> EncodedInput: ...
    def encode_multimodal_input(self, image_path: Path, prompt: str) -> EncodedMultimodalInput: ...
    def extract_vision_attention(self, pixel_values: torch.Tensor, layer: int) -> VisionAttentionResult: ...
    def extract_visual_embeddings(self, pixel_values: torch.Tensor) -> torch.Tensor: ...
    def build_inputs_embeds(self, input_ids: torch.Tensor, visual_embeddings: torch.Tensor) -> torch.Tensor: ...
    def forward_language(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor | None): ...
    def generate_regular(...): ...
```

### 兼容 VCD LLaVA 的关键点

1. 图像预处理
   - 使用 `llava.mm_utils.process_images([image], image_processor, model.config)`。
   - 注意 `image_aspect_ratio == "pad"` 时会 pad 成正方形。

2. Prompt 编码
   - 使用 `llava.mm_utils.tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")`。
   - Prompt 模板建议优先复用 `llava.conversation.conv_templates`，而不是硬编码 `USER: <image>`。
   - 配置中保留 `conv_mode`，默认根据模型名推断，如 `llava_v1`。

3. Visual embeddings
   - 原始 LLaVA 中可调用 `model.encode_images(pixel_values)` 得到投影后的视觉 token embedding。
   - 该 embedding 应直接对应 `<image>` token 替换后的 hidden size。
   - 需要记录 token 数量是否包含 CLS token，以及 patch grid 映射。

4. Attention 提取
   - 当前 `Attend/attention_extractor.py` 针对 HF vision tower 的输出结构。
   - VCD LLaVA vision tower 包装在 `model.get_vision_tower()`，底层多为 CLIPVisionTower。
   - 需要检查 `vision_tower.vision_tower(...)` 或 `vision_tower(...)` 是否可传 `output_attentions=True`。
   - 如果 wrapper 不返回 attentions，优先在 adapter 内访问底层 CLIP 模型，而不是修改原始 LLaVA 文件。

5. 双分支前向
   - 先用 `build_inputs_embeds(input_ids, visual_embeddings)` 替换 `IMAGE_TOKEN_INDEX` 位置。
   - 对原始 visual embeddings 与 padded visual embeddings 分别 forward。
   - 对 VCD LLaVA Llama 模型，通常可调用：
     - `model(input_ids=None, inputs_embeds=..., attention_mask=...)`
     - 或 `model.model(...); model.lm_head(...)`
   - Adapter 内做兼容分支，Respond 阶段不直接碰模型内部结构。

## 三阶段重构策略

### Phase 1：配置与产物契约

新增 `merge/config.py`、`merge/schemas.py`、`merge/merge_config.yaml`。

总配置建议：

```yaml
paths:
  input_image: EnAR/Envision/image/data/horse_6.png
  output_dir: EnAR/merge/outputs/demo

llava:
  model_path: EnAR/pre_model/LLM/llava-v1.5-7b
  model_base:
  model_name:
  conv_mode: llava_v1
  load_8bit: false
  load_4bit: false
  device: cuda
  device_map: auto
  dtype: float16

envision:
  sd_model_dir: EnAR/pre_model/DDIM/stable-diffusion-v1-5
  image_size: 512
  prompt: ""
  negative_prompt: ""
  num_ddim_steps: 50
  inversion_step_T: 30
  langevin_steps_M: 10
  sample_count_K: 4
  guidance_scale: 1.0

attend:
  vision_layer_number: 6
  attention_top_ratio: 0.10
  uncertainty_top_ratio: 0.05
  padding_ratio_limit: 0.10
  uncertainty_weight: 1.0
  save_heatmaps: true
  save_source_masks: true

respond:
  question: "Describe the image."
  alpha: 1.0
  max_new_tokens: 64
  do_sample: true
  temperature: 1.0
  top_p: 1.0
  seed: 42
  use_apc: true
  apc_beta: 0.1
  padding_strategy: pad_token_embedding
  save_decode_trace: true

runtime:
  skip_envision_if_exists: false
  skip_attend_if_exists: false
  save_intermediate: true
  debug: false
```

### Phase 2：LLaVA runtime 和 adapter

实现并单测以下最小契约：

- 能从 `VCD/experiments/llava/model/builder.py` 成功加载 tokenizer/model/image_processor。
- 能把一张图片预处理成 LLaVA 所需 pixel tensor。
- 能把包含 `<image>` 的 prompt 编成 `input_ids`，并定位 image token 位置。
- 能调用 `model.encode_images(pixel_values)` 得到 `[batch, num_image_tokens, hidden]`。
- 能使用 visual embeddings 替换 image token 并完成一次 language forward。
- 能 regular generate 一个 baseline answer。

验收标准：

- 不依赖 HF `LlavaForConditionalGeneration`。
- Attend 和 Respond 不再各自加载一份 LLaVA。
- 单次完整 pipeline 中 LLaVA 只加载一次。

### Phase 3：Envision stage 包装

先薄封装现有 `EnvisionPipeline`：

- 输入来自总配置。
- 输出转换为 `EnvisionArtifacts`。
- 保存到 `output_dir/envision/`。
- 支持复用已有产物。

该阶段暂不改变 Stable Diffusion 逻辑。

### Phase 4：Attend stage 重构

复用以下现有模块：

- `Attend/contrastive.py`
- `Attend/token_selector.py`
- `Attend/uncertainty_mapper.py`
- `Attend/mask_mapper.py`
- `Attend/visualizer.py`
- `Attend/output_writer.py` 可部分复用或改成 merge writer。

替换以下部分：

- `Attend/model_loader.py` 的 HF 加载逻辑不用在 merge 中复用。
- `Attend/preprocessor.py` 可参考，但图像预处理应统一走 `LlavaAdapter`。
- `Attend/attention_extractor.py` 可复用算法，但输入输出结构可能需要 adapter 做兼容。

重点验证：

- original image 与 impression image 的 preprocess meta 一致。
- attention token 数量和 patch grid 一致。
- `selected_patch_indices` 到 `selected_vision_token_indices` 的 offset 正确。
- 如果 visual embeddings 不含 CLS token，selected vision token 应等于 patch index。
- 如果 visual embeddings 含 CLS token，selected vision token 应加 1。

### Phase 5：Respond stage 重构

复用以下现有模块：

- `Respond/padded_visual_builder.py`
- `Respond/logits_processor.py`
- `Respond/token_selector.py`
- `Respond/generation_loop.py` 的解码思想。
- `Respond/output_writer.py` 可参考。

需要改造：

- `input_encoder.py` 的 processor 逻辑换成 `LlavaAdapter.encode_multimodal_input`。
- `visual_embeddings.py` 改为直接用 `LlavaAdapter.extract_visual_embeddings`。
- `dual_branch_forwarder.py` 的模型内部访问移入 adapter。
- regular generation 改为 VCD LLaVA 原生 generate 路径。

重点验证：

- prompt 中 image token 占位数量必须等于 visual embedding token 数量。
- padded visual embeddings shape 与原始 visual embeddings 完全一致。
- 原始分支和 padded 分支的 logits shape 一致。
- `alpha=0` 时 EnAR logits 应退化为 regular/original logits。
- greedy 与 sampling 两种模式都能跑通。

### Phase 6：完整 pipeline 与 CLI

`merge/pipeline.py`：

```python
class MergeEnARPipeline:
    def run(self) -> MergeEnARResult:
        runtime = load_llava_runtime(...)
        adapter = LlavaAdapter(runtime, ...)
        envision = EnvisionStage(...).run()
        attend = AttendStage(..., adapter).run(envision)
        respond = RespondStage(..., adapter).run(attend)
        return result
```

`merge/cli.py`：

```bash
python EnAR/merge/cli.py --config EnAR/merge/merge_config.yaml
```

输出目录建议：

```text
outputs/demo/
  resolved_config.yaml
  manifest.json
  envision/
  attend/
  respond/
    answer_regular.txt
    answer_enar.txt
    respond_result.json
```

`manifest.json` 至少包含：

- 输入图像、问题、模型路径。
- 每阶段输出路径。
- selected patch/token 数量。
- regular answer、enar answer。
- 运行时间、device、dtype。
- 关键版本信息和 git 状态摘要。

## 测试计划

### 不加载大模型的单元测试

使用 fake adapter / fake model：

- config YAML 解析与路径 resolve。
- stage artifact dataclass 序列化。
- selected patch 到 selected token 的映射。
- visual embedding padding 策略。
- contrastive logits 公式。
- `alpha=0` 退化测试。
- output manifest 字段完整性。

### 小规模集成测试

如果本地已有 LLaVA 权重：

1. 只测试 LLaVA adapter：
   - load
   - preprocess image
   - encode prompt
   - encode image
   - one-step forward

2. 跳过 Envision，使用已有 original/impression/uncertainty fixture 测 Attend + Respond。

3. 完整 pipeline smoke test：
   - `max_new_tokens=4`
   - `sample_count_K=1`
   - `langevin_steps_M=0`
   - 只验证流程跑通和产物存在。

### 回归检查

- `python3 -m py_compile EnAR/merge/*.py`
- `pytest EnAR/merge/tests`
- 对已有 `Envision/Attend/Respond` 测试不做破坏性改动；如果后续共享了模块，应同步跑相关测试。

## 风险与解决方案

### 风险 1：VCD LLaVA 与 HF LLaVA 的视觉 token 数量不一致

解决方案：

- 在 adapter 中统一记录 `visual_token_count`、`patch_grid`、`has_cls_token`。
- 每次 Attend -> Respond 交接时做 shape assert。
- 将 token layout 写入 result json。

### 风险 2：vision attention 不容易从 VCD wrapper 取出

解决方案：

- 首选通过底层 CLIP vision model 获取 `output_attentions=True`。
- 如果 wrapper 已经丢弃 attention，需要在 adapter 内新增 hook，而不是修改全局 LLaVA 架构。
- 保留一个 debug 方法输出 vision tower 类型、forward signature、返回对象字段。

### 风险 3：prompt 模板与 image token 数量错位

解决方案：

- 使用 LLaVA 原始 `conversation.py` 和 `tokenizer_image_token`。
- 每次 encode 后检查 `(input_ids == IMAGE_TOKEN_INDEX).sum()`。
- 如果模型配置启用 `<im_start>/<im_end>`，adapter 负责模板和 token 兼容。

### 风险 4：双分支 forward 与 KV cache 兼容复杂

解决方案：

- 第一版不使用 KV cache，优先保证正确性。
- 后续再优化为 prefill 一次、增量 decode。
- decode trace 中记录每步耗时，为后续优化提供依据。

### 风险 5：显存占用

解决方案：

- LLaVA 只加载一次，传入 Attend/Respond 共用。
- 支持 `load_8bit`、`load_4bit`。
- Envision 阶段结束后显式释放 Stable Diffusion 组件，并 `torch.cuda.empty_cache()`。
- 配置支持跳过 Envision 或复用已有产物，便于低显存调试。

## 建议实施顺序

1. 新增 `merge/config.py`、`schemas.py`、`merge_config.yaml`。
2. 新增 `llava_runtime.py`，跑通 VCD LLaVA 权重加载。
3. 新增 `llava_adapter.py`，跑通图像预处理、prompt encode、visual embedding、one-step forward。
4. 新增 `envision_stage.py`，薄封装现有 Envision。
5. 新增 `attend_stage.py`，替换为 adapter 提供的 vision attention。
6. 新增 `respond_stage.py`，替换为 adapter 提供的 multimodal encode 和双分支 forward。
7. 新增 `pipeline.py`、`cli.py`、`output_writer.py`。
8. 补齐单元测试和 smoke test。
9. 写 `merge/README.md`，记录运行命令、配置字段、常见错误。

## 第一版完成标准

- `EnAR/merge/` 下存在独立入口，可以从一个 YAML 启动完整 EnAR。
- LLaVA 模型由 `EnAR/VCD/experiments/llava/model/builder.py` 加载。
- Envision、Attend、Respond 三阶段产物都保存在统一输出目录。
- Respond 同时输出 regular answer 和 EnAR answer。
- 关键中间产物可追踪：uncertainty map、selected patch/token、padded visual meta、decode trace。
- 不破坏原有 `Envision/`、`Attend/`、`Respond/` 单独运行方式。
