# Respond 阶段 VCD 解码复现任务文档

## 1. 目标

本文档用于指导后续修改 `EnAR/Respond/` 中的 VCD 风格对比解码，使其同时对齐：

- `EnAR/VCD/` 代码中的 Visual Contrastive Decoding 实现。
- VCD 论文 `Leng_Mitigating_Object_Hallucinations...CVPR_2024_paper.pdf` 的解码公式。
- EnAR 论文 `Liang_Envision_Attend_Then_Respond...CVPR_2026_paper.pdf` 中 Respond 阶段的 padded visual input 设定。

本次只形成任务文档，不修改 Respond 实现代码。

## 2. 论文与代码结论

### 2.1 VCD 的原始解码逻辑

VCD 使用原始视觉输入 `v` 与失真视觉输入 `v'` 各前向一次，得到两路 next-token logits：

```text
logits_orig = logit_theta(y | v, x, y_<t)
logits_cd   = logit_theta(y | v', x, y_<t)
```

然后构造对比 logits：

```text
logits_vcd = (1 + alpha) * logits_orig - alpha * logits_cd
```

其中 `alpha = 0` 时退化为 regular decoding。VCD 代码位置：

- `EnAR/VCD/vcd_utils/vcd_sample.py`
- `EnAR/VCD/vcd_utils/vcd_add_noise.py`

`vcd_sample.py` 的关键实现是：

```python
diffs = (1 + cd_alpha) * next_token_logits - cd_alpha * next_token_logits_cd
```

并在采样前应用 Adaptive Plausibility Constraint, APC：

```python
cutoff = log(cd_beta) + max(next_token_logits)
cd_logits = diffs.masked_fill(next_token_logits < cutoff, -inf)
```

这等价于保留满足以下条件的 token：

```text
softmax(logits_orig)[token] >= beta * max softmax(logits_orig)
```

VCD 的 `v'` 来自对图像张量加入扩散式高斯噪声，见 `vcd_add_noise.py`。

### 2.2 EnAR Respond 与 VCD 的差异

EnAR 论文的 Respond 阶段沿用 VCD 的对比解码公式，但对比输入不再是高斯噪声图像，而是 Attend 阶段定位到反事实 token 后构造的 padded visual input：

```text
original branch: v
contrastive branch: v_pad
```

EnAR 论文 3.2 和 3.3 节说明：

- Attend 阶段得到 counterfactual token index set `H`。
- 将 `H` 对应的 visual tokens 替换为模型内置 padding token，得到 padded input。
- Respond 对 `v` 与 padded input 做对比解码，抑制反事实区域诱导的幻觉。

因此 Respond 中的 `logits_pad` 应被理解为 VCD 公式里的 `logits_cd`，但其输入来源是 token-level padded visual embeddings，而不是 noisy image。

### 2.3 当前 Respond 初步实现状态

当前 `EnAR/Respond/` 已经具备复现骨架：

- `generation_loop.py`：自定义逐 token 解码循环。
- `dual_branch_forwarder.py`：原图 visual embeddings 与 padded visual embeddings 双分支 forward。
- `logits_processor.py`：对比 logits 与 APC。
- `padded_visual_builder.py`：按 Attend indices 构造 padded visual embeddings。
- `token_selector.py`：greedy / sampling 选择 next token。

但仍需进一步对齐 VCD 代码和论文：

- APC 的实现目前基于 `softmax(logits_orig)` 显式 mask；需要确认与 VCD 的 logit cutoff 完全等价并统一命名。
- 当前 generation loop 没有维护双分支 `past_key_values`，和 VCD hook 到 HF `sample()` 的 cache 路径不同。
- 当前默认 `do_sample=false`，而 VCD 实验表述为从 modified post-softmax distribution 采样。
- 当前 `ContrastiveLogitsProcessor` 有调试 `print`，不适合正式复现。
- 当前 `padding_strategy` 有多种 fallback，需要明确论文主设置使用 `pad_token_embedding`。

## 3. 修改任务清单

### Task 1: 统一 Respond 中的 VCD 术语

目标文件：

- `EnAR/Respond/logits_processor.py`
- `EnAR/Respond/generation_loop.py`
- `EnAR/Respond/README.md`
- `EnAR/Respond/respond_config.yaml`

要求：

- 将文档和 trace 字段中的 `pad` 分支说明为 `contrastive/padded` 分支。
- 保留代码变量 `logits_pad` 也可以，但输出 JSON 中建议增加：
  - `logits_original`
  - `logits_contrastive_input`
  - `logits_vcd`
- 明确 `logits_pad == logit_theta(y | x, v_pad, y_<t)`。
- 在 README 中写清楚 EnAR 的 `v_pad` 替代 VCD 的 noisy `v'`。

验收标准：

- 用户看到输出日志时能明确区分 regular branch、padded branch、VCD logits。
- 不再把 EnAR 的 padded input 误写为 Gaussian noisy image。

### Task 2: 对齐 VCD 对比 logits 公式

目标文件：

- `EnAR/Respond/logits_processor.py`
- `EnAR/Respond/tests/test_logits_selector.py`

要求：

- `ContrastiveLogitsProcessor(alpha)` 只做公式：

```text
(1 + alpha) * logits_orig - alpha * logits_contrastive
```

- 删除正式路径里的大段 `print` 调试输出。
- 增加或保留单测：
  - `alpha = 0` 时输出等于 `logits_orig`。
  - `alpha = 1` 时输出等于 `2 * logits_orig - logits_contrastive`。
  - shape 不一致时抛错。

验收标准：

- 与 `EnAR/VCD/vcd_utils/vcd_sample.py` 中 `diffs` 公式一致。
- 单测覆盖不同 alpha。

### Task 3: 对齐 APC 实现

目标文件：

- `EnAR/Respond/logits_processor.py`
- `EnAR/Respond/generation_loop.py`
- `EnAR/Respond/tests/test_logits_selector.py`

要求：

- APC 必须以原始分支 logits 为置信度基准，而不是 padded 分支或对比后 logits。
- 保留 token 条件：

```text
p_orig(token) >= beta * max_token p_orig(token)
```

- 实现可以继续使用 softmax mask，也可以改成 VCD 代码中的 logit cutoff：

```python
cutoff = torch.log(torch.tensor(beta, device=logits_orig.device, dtype=logits_orig.dtype)) + logits_orig.max(dim=-1, keepdim=True).values
mask = logits_orig >= cutoff
```

- 如果使用 logit cutoff，需要注意 `beta == 0` 时 `log(0)` 的处理。
- trace 中记录：
  - `apc.enabled`
  - `apc.beta`
  - `apc.kept_count`
  - `apc.filtered_count`
  - `apc.cutoff_mode`

验收标准：

- 与 VCD 论文 Eq.4 / Eq.5 语义一致。
- 与 VCD 代码 `masked_fill(next_token_logits < cutoff, -inf)` 等价。
- 对所有 batch 至少保留一个 token。

### Task 4: 明确 EnAR padded visual input 主路径

目标文件：

- `EnAR/Respond/padded_visual_builder.py`
- `EnAR/Respond/config.py`
- `EnAR/Respond/respond_config.yaml`
- `EnAR/Respond/tests/test_visual_padding.py`

要求：

- 论文主路径应优先使用模型内置 `pad_token_embedding`。
- fallback 策略可以保留，但必须在 `padding_meta` 中显式记录：
  - `requested_strategy`
  - `actual_strategy`
  - fallback 原因
- 若 `pad_token_id` 不存在，建议显式使用 tokenizer 的 `pad_token_id`；若仍不存在，再 fallback。
- 确认替换发生在 projected visual embeddings 空间中，hidden size 必须与 LLM token embedding 一致。

验收标准：

- 默认配置体现论文主设置。
- `respond_result.json` 可以追踪是否真的用了 pad token embedding。
- selected indices 越界时不会 silently 污染结果，需要记录 ignored indices。

### Task 5: 决定默认解码策略是否跟随 VCD sampling

目标文件：

- `EnAR/Respond/config.py`
- `EnAR/Respond/token_selector.py`
- `EnAR/Respond/README.md`
- `EnAR/Respond/respond_config.yaml`

背景：

VCD 论文实验说明 regular decoding 和 VCD 都是从 post-softmax distribution 直接采样；当前 Respond 默认是 greedy。

要求：

- 为“复现 VCD 设置”提供一个明确配置档，建议：

```yaml
generation:
  do_sample: true
  temperature: 1.0
  top_p: 1.0
contrastive:
  alpha: 1.0
  use_apc: true
  apc_beta: 0.1
```

- 如果继续保留 greedy 默认，需要在 README 和 result meta 中标注 `decode_mode: greedy_debug` 或类似字段。
- 对 sampling 增加 `seed` 配置，确保复现实验可复现。

验收标准：

- 用户可以一眼区分“调试 greedy 跑通”和“论文式 VCD sampling”。
- sampling 路径支持固定随机种子。

### Task 6: 增加双分支 KV cache 或明确无 cache 版本限制

目标文件：

- `EnAR/Respond/dual_branch_forwarder.py`
- `EnAR/Respond/generation_loop.py`

背景：

VCD 通过 patch HuggingFace `GenerationMixin.sample()` 使用模型原生生成流程，每步更新两路 `model_kwargs` / cache。当前 Respond 每步重算完整序列，逻辑更简单但效率较低。

要求：

- 第一阶段可以保留无 cache，但必须在 README 写清楚：
  - 首版为了可读性和正确性不使用 KV cache。
  - 生成耗时约为 regular decoding 的两倍以上。
- 第二阶段实现双分支 cache：
  - `past_key_values_orig`
  - `past_key_values_padded`
  - 每步只输入上一个 token 的 embedding。
- cache 路径必须验证输出与无 cache 路径在 greedy 模式下 token 序列一致。

验收标准：

- 无 cache 与 cache greedy 结果一致。
- cache 实现不会把原始分支和 padded 分支的 `past_key_values` 混用。

### Task 7: 增强 decode trace 以便论文复现诊断

目标文件：

- `EnAR/Respond/generation_loop.py`
- `EnAR/Respond/output_writer.py`

每步 trace 建议记录：

- `step`
- `selected_token_id`
- `selected_token`
- `decode_mode`
- `alpha`
- `apc`
- `top_original`
- `top_padded`
- `top_vcd_before_apc`
- `top_final_after_apc`
- `selected_token_logits`：
  - `orig`
  - `padded`
  - `vcd`
  - `final`
- `selected_token_probs`：
  - `orig`
  - `padded`
  - `vcd`
  - `final`

验收标准：

- 可以定位某个 hallucinated token 是由原始分支、padded 分支还是 APC 导致。
- 可以直接对比 VCD 公式前后的 top tokens 变化。

### Task 8: 增加与 VCD 代码的最小一致性测试

目标文件：

- `EnAR/Respond/tests/`

新增测试建议：

- `test_vcd_formula_matches_reference`
  - 构造固定 logits，断言 Respond 公式与 VCD `diffs` 公式一致。
- `test_apc_matches_vcd_cutoff`
  - 构造固定 logits 和 beta，比较 softmax mask 与 logit cutoff mask。
- `test_alpha_zero_matches_regular_branch`
  - alpha 为 0 时，final logits 不受 padded branch 影响。
- `test_padding_branch_changes_only_selected_tokens`
  - 只替换 selected visual token indices，其余 embeddings 不变。
- `test_sampling_seed_reproducible`
  - 如果加入 seed 配置，采样结果应可复现。

验收标准：

- 不加载 LLaVA 大模型也能运行单元测试。
- 所有数学逻辑可以用小 tensor 验证。

## 4. 推荐实施顺序

1. 先清理 `ContrastiveLogitsProcessor` 与 APC，使数学公式完全对齐 VCD。
2. 再固定 padded visual input 的默认策略和 metadata。
3. 然后补充 decode trace，确保每步可诊断。
4. 接着增加 VCD 一致性单测。
5. 最后再考虑双分支 KV cache 优化。

## 5. 推荐默认配置

用于论文式复现：

```yaml
generation:
  max_new_tokens: 64
  do_sample: true
  temperature: 1.0
  top_p: 1.0
contrastive:
  alpha: 1.0
  use_apc: true
  apc_beta: 0.1
  padding_strategy: pad_token_embedding
  save_decode_trace: true
model:
  vision_feature_select_strategy: default
```

用于快速调试：

```yaml
generation:
  max_new_tokens: 32
  do_sample: false
contrastive:
  alpha: 1.0
  use_apc: true
  apc_beta: 0.1
  padding_strategy: pad_token_embedding
  save_decode_trace: true
```

## 6. 关键注意事项

- EnAR 的 `v_pad` 是 token-level padded visual input，不是 VCD 的 Gaussian noisy image。
- 对比 logits 的第二项应来自 padded 分支 logits，而不是 padded embedding 本身。
- APC 的候选集合必须由原始分支置信度决定。
- `alpha = 0` 必须严格等价于原始分支解码。
- 如果使用 `pad_token_embedding`，需要确认 LLaVA tokenizer/model 中 pad token 是否存在且 hidden size 匹配。
- 若 selected visual token indices 与 image placeholder 数量不一致，应优先检查 `vision_feature_select_strategy` 和 LLaVA processor 的 image token 配置。

## 7. 完成定义

当以下条件满足时，可认为 Respond 的 VCD 解码复现任务完成：

- Respond 的对比 logits 与 VCD 代码公式一致。
- APC 与 VCD 论文和代码语义一致。
- EnAR 的 padded visual input 被明确作为 VCD 的 contrastive visual input。
- 默认配置能区分论文式 sampling 和调试 greedy。
- 单元测试覆盖公式、APC、padding、采样可复现性。
- 输出 trace 足以解释每一步 token 选择。
