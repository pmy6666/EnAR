# 4.3 Respond 阶段模块化实现设计

## 1. 阶段目标

Respond 阶段负责基于原始视觉输入 `v` 与 padded 视觉输入 `v'` 执行对比解码，降低模型依赖语言先验或常识偏置产生反事实幻觉的概率。该阶段是 EnAR 的最终回答生成模块。

阶段输入：

- 原始图像。
- 用户问题。
- Attend 阶段输出的反事实 token indices。
- LLaVA-1.5-7B 本地模型路径。
- 对比解码超参数 `alpha`。

阶段输出：

- Regular decoding 答案。
- EnAR contrastive decoding 答案。
- 每步解码日志，可选。
- `respond_result.json`。

## 2. 总体流程

```text
原图 + 问题 + Attend token indices
  -> 加载 LLaVA 模型与 processor
  -> 构造文本 prompt 与图像输入
  -> 提取原始 projected visual embeddings
  -> 按 token indices 构造 padded visual embeddings
  -> 每步分别前向原始分支和 padded 分支
  -> 合并 logits: (1 + alpha) * logits_orig - alpha * logits_pad
  -> 选择 next token
  -> 直到停止条件
  -> 输出最终回答与日志
```

## 3. 模块设计

### 3.1 配置模块 `RespondConfig`

职责：

- 管理 Respond 阶段所有输入路径、生成参数和对比解码参数。

建议字段：

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `llava_model_dir` | path | `EnAR/pre_model/LLM/llava-1.5-7b-hf` | 本地 LLaVA 模型 |
| `image_path` | path | 必填 | 原始图像 |
| `question` | str | 必填 | 用户问题 |
| `attend_result_json` | path | 必填 | Attend 输出 |
| `output_dir` | path | 必填 | 输出目录 |
| `alpha` | float | 1.0 | 对比解码强度 |
| `max_new_tokens` | int | 64 | 最大生成长度 |
| `do_sample` | bool | false | 初版建议 greedy |
| `temperature` | float | 1.0 | sampling 时使用 |
| `top_p` | float | 1.0 | sampling 时使用 |
| `use_apc` | bool | false | 是否启用 adaptive plausibility constraint |
| `apc_beta` | float | 0.1 | APC 阈值参数，可选 |

输入：

- 命令行参数。
- `attend_result.json`。

输出：

- 规范化配置对象。

### 3.2 LLaVA 对话输入模块 `LlavaPromptBuilder`

职责：

- 将用户问题封装为 LLaVA-1.5 的对话格式。

输入：

- `question`。

输出：

- prompt 字符串。

建议格式：

```text
USER: <image>
{question}
ASSISTANT:
```

实现要点：

- 与现有 `test_load_llava_1_5_7b.py` 中的 `build_prompt` 保持一致。
- 生成阶段需要记录 prompt token 长度，便于只解码新生成部分。

### 3.3 模型加载模块 `LlavaGenerationLoader`

职责：

- 加载 LLaVA processor 和完整生成模型。
- patch processor 配置。
- 设置 dtype、device、eval 模式。

输入：

- `llava_model_dir`。

输出：

- `processor`。
- `model`。
- `tokenizer`。
- `generation_meta`。

实现要点：

- 沿用当前测试脚本中的 processor patch 逻辑，避免 image token 数不匹配。
- 初版使用 `torch.inference_mode()`，不计算梯度。
- 如果显存不足，Respond 阶段应单独运行，不与 SD 同时驻留显存。

### 3.4 输入编码模块 `MultimodalInputEncoder`

职责：

- 将图像和 prompt 编码为模型可用输入。
- 获取 input ids、attention mask、pixel values。

输入：

- `image_path`。
- `prompt`。
- `processor`。

输出：

- `input_ids`。
- `attention_mask`。
- `pixel_values`。
- `prompt_len`。

实现要点：

- 使用 `processor(text=prompt, images=image, return_tensors="pt")`。
- 所有 tensor 移动到同一 device。
- 保存 image token 在文本序列中的位置，后续如果需要替换 embeddings 会用到。

### 3.5 视觉 embedding 提取模块 `VisualEmbeddingExtractor`

职责：

- 提取 LLaVA 中进入语言模型前的 projected visual embeddings。
- 建立 patch index 与 visual embedding index 的对应关系。

输入：

- `pixel_values`。
- `model`。
- `attend_result_json`。

输出：

- `visual_embeddings_orig`。
- `visual_token_layout`。

实现要点：

- LLaVA 通常流程是：CLIP vision tower 输出 image features，经过 multimodal projector 映射到 LLM hidden size。
- 需要确认当前 transformers 版本中函数入口，可能包括：
  - `model.get_image_features(...)`
  - `model.vision_tower(...)`
  - `model.multi_modal_projector(...)`
- Attend 输出的 `selected_patch_indices` 是纯 patch index；如果 visual features 包含 cls token，需要转换为实际 embedding index。

### 3.6 Padded visual input 构造模块 `PaddedVisualInputBuilder`

职责：

- 根据 Attend 阶段的反事实 token indices 构造 `v'`。

输入：

- `visual_embeddings_orig`。
- `selected_patch_indices` 或 `selected_vision_token_indices`。
- `model` 的 pad token embedding 或替代策略。

输出：

- `visual_embeddings_padded`。
- `padding_mask`。
- `padding_meta`。

候选 padding 策略：

1. `pad_token_embedding`：使用 LLM embedding table 中 `pad_token_id` 对应 embedding。
2. `zero_embedding`：将对应 visual embedding 置零。
3. `mean_visual_embedding`：用未选中 visual token 的均值替换。

推荐初版：

- 优先尝试 `pad_token_embedding`，最贴近论文“替换为模型内置 pad token”的描述。
- 如果 hidden size 或接入位置不兼容，退化为 `zero_embedding` 并在结果中明确记录。

实现要点：

- 不应修改原始 `visual_embeddings_orig`，需要 clone 后生成 padded 分支。
- 替换数量应与 Attend 的 padding ratio 一致。
- 输出中保存替换前后的 embedding norm，用于调试。

### 3.7 双分支前向模块 `DualBranchForwarder`

职责：

- 对原始视觉输入 `v` 和 padded 视觉输入 `v'` 分别执行语言模型前向。
- 返回两路 next-token logits。

输入：

- 当前 `input_ids` 或 `inputs_embeds`。
- `attention_mask`。
- `visual_embeddings_orig`。
- `visual_embeddings_padded`。
- `past_key_values`，可选。

输出：

- `logits_orig`。
- `logits_pad`。
- `past_key_values_orig`。
- `past_key_values_pad`。

实现难点：

- Hugging Face 的 `generate()` 默认不支持每步替换内部 image embeddings，因此需要自定义最小 generation loop。
- 首 token 前向需要把图像 token embedding 融合到文本 embedding 中。
- 后续 step 可使用各自分支的 `past_key_values` 加速。

实现策略：

- 第一版可以不做 KV cache，先保证逻辑正确。
- 第二版再为两个分支分别维护 KV cache，降低生成耗时。

### 3.8 对比 logits 合并模块 `ContrastiveLogitsProcessor`

职责：

- 实现论文 Eq.7 的 logits 合并。

输入：

- `logits_orig`。
- `logits_pad`。
- `alpha`。

输出：

- `logits_contrastive`。

公式：

```text
logits_contrastive = (1 + alpha) * logits_orig - alpha * logits_pad
```

实现要点：

- 论文写作中使用 `p(y|x,v)` 表示 next token 分布，但工程中通常对 logits 操作更稳定。
- 两路 logits shape 必须一致。
- 需要避免 fp16 下数值过大，可在必要时转 fp32 合并。

### 3.9 可选 APC 模块 `AdaptivePlausibilityConstraint`

职责：

- 约束对比解码不要选择原始分支中本来就极不可信的 token。
- 这是 VCD 中常见的防护机制，属于选做增强。

输入：

- `logits_orig`。
- `logits_contrastive`。
- `apc_beta`。

输出：

- 过滤后的 logits。

实现策略：

- 根据原始分支概率设置候选 token 集。
- 只在候选集合内使用 contrastive logits 排序。
- 初版可关闭，保证 EnAR 主流程清晰。

### 3.10 Token 选择模块 `NextTokenSelector`

职责：

- 从合并后的 logits 中选择 next token。

输入：

- `logits_contrastive`。
- `do_sample`。
- `temperature`。
- `top_p`。

输出：

- `next_token_id`。
- `next_token_logprob`，可选。

推荐初版：

- 使用 greedy decoding，即 `argmax`。
- 后续评估采样策略时再打开 `temperature` 和 `top_p`。

### 3.11 生成循环模块 `ContrastiveGenerationLoop`

职责：

- 管理完整自回归生成过程。
- 每步调用双分支前向、logits 合并、token 选择和停止条件判断。

输入：

- 编码后的 multimodal input。
- `visual_embeddings_orig`。
- `visual_embeddings_padded`。
- generation config。

输出：

- `generated_ids`。
- `decoded_text`。
- `decode_trace`。

停止条件：

- 生成到 `eos_token_id`。
- 达到 `max_new_tokens`。
- 可选遇到特殊停止字符串。

调试日志建议：

- 每步 top-5 token in original branch。
- 每步 top-5 token in padded branch。
- 每步 top-5 token after contrastive merge。
- 被选择 token。

### 3.12 Baseline 生成模块 `RegularGenerationRunner`

职责：

- 使用 LLaVA 原生 `generate()` 生成 baseline 答案。
- 为 EnAR 输出提供对照。

输入：

- `input_ids`。
- `attention_mask`。
- `pixel_values`。
- `max_new_tokens`。

输出：

- `regular_answer`。

实现要点：

- 沿用现有测试脚本中的生成方式。
- baseline 和 EnAR 使用同一 prompt、同一图像、同一 max token。

### 3.13 输出管理模块 `RespondOutputWriter`

职责：

- 保存 regular 与 EnAR 答案、参数、token indices 和生成日志。

输出目录建议：

```text
outputs/respond/{run_id}/
  respond_result.json
  decode_trace.json
  answer_regular.txt
  answer_enar.txt
```

`respond_result.json` 建议包含：

```json
{
  "image_path": "...",
  "question": "...",
  "attend_result_json": "...",
  "selected_patch_indices": [],
  "selected_vision_token_indices": [],
  "alpha": 1.0,
  "padding_strategy": "pad_token_embedding",
  "regular_answer": "...",
  "enar_answer": "...",
  "max_new_tokens": 64,
  "use_apc": false
}
```

## 4. 主控流程模块 `RespondPipeline`

职责：

- 串联输入读取、模型加载、regular baseline、padded visual input 构造和 contrastive generation。

输入：

- `RespondConfig`。

输出：

- `RespondResult`：
  - `regular_answer`
  - `enar_answer`
  - `respond_result_json`

执行顺序：

1. 读取 `attend_result.json`。
2. 构造 LLaVA prompt。
3. 加载模型和 processor。
4. 编码图像与文本。
5. 运行 regular baseline。
6. 提取原始 visual embeddings。
7. 构造 padded visual embeddings。
8. 执行 contrastive generation loop。
9. 解码输出文本。
10. 保存结果和日志。

## 5. 验收与调试

最小验收：

- regular baseline 能正常输出答案。
- padded visual embeddings shape 与原始 visual embeddings 一致。
- 两路 logits shape 完全一致。
- contrastive logits 能完成至少一个 token 的选择。
- 最终输出非空答案。

关键调试项：

- selected token 数量是否符合 Attend 输出。
- padding 前后 visual embedding norm 是否变化合理。
- `alpha = 0` 时 EnAR 输出应接近 regular 自定义 loop 输出。
- 若不用 KV cache，自定义 loop 与 `generate()` 的 baseline 可能存在轻微差异，但不应完全异常。

失败现象与处理：

| 现象 | 可能原因 | 处理 |
| --- | --- | --- |
| 无法替换 visual embeddings | transformers 内部封装较深 | 先绕过 `generate()`，实现 `inputs_embeds` 级别的最小 forward |
| 输出乱码或重复 | prompt 拼接或 attention mask 错误 | 对比现有测试脚本的 input ids 与 prompt_len |
| EnAR 与 regular 完全一样 | selected token 未真正进入 padded 分支 | 检查 visual embedding 替换位置 |
| EnAR 输出退化严重 | `alpha` 过大或 padding 过多 | 降低 alpha，检查 Attend mask |
| 显存不足 | 双分支同时前向 | 分步前向并释放中间变量，或关闭 KV cache 先跑短输出 |

## 6. 与前两阶段的接口

来自 Attend 的输入：

```text
attend_result.json
  -> selected_patch_indices
  -> selected_vision_token_indices
  -> has_cls_token
  -> patch_grid
```

来自用户或评测集的输入：

```text
image_path
question
```

Respond 输出给评测模块：

```text
regular_answer
enar_answer
decode_trace
respond_result.json
```

## 7. 推荐实现优先级

1. 先实现 regular baseline runner，确认 LLaVA 本地推理正常。
2. 实现 visual embedding extraction，并打印 shape。
3. 实现 padded visual input builder，确认替换位置正确。
4. 实现单步 dual-branch logits merge。
5. 实现无 KV cache 的 greedy contrastive generation。
6. 最后再加入 KV cache、APC、sampling 等优化项。
