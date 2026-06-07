# EnAR × VLMBias 评估方案设计

本文档设计 `enar_eval` 子项目：在本地 VLMBias 数据集上批量运行 EnAR 三阶段 pipeline，并参考 `Liang_Envision_Attend_Then_Respond_Counterfactual_Hallucination_Mitigation_in_Large_Vision-Language_CVPR_2026_paper.pdf` 的实验设置与指标，对模型反事实视觉理解能力进行评估。

当前阶段只做方案设计，不实现代码。

## 目标

`enar_eval` 的目标是提供一个统一、可复现、可调参的评估入口：

1. 从 `EnAR/toy_dataset/VLMBias/` 读取样本。
2. 对每个样本执行 Regular baseline 与 EnAR 三阶段推理。
3. 将 Envision、Attend、Respond 的阶段参数统一放入一个 YAML。
4. 产出逐样本结果、类别汇总、总体指标、错误分析与可视化索引。
5. 支持小规模 smoke run、指定 split/category、断点续跑、阶段缓存复用。

## 论文依据

EnAR 论文中的核心评估点：

- VLMBias 是反事实幻觉评测数据集，用于衡量模型是否能在图像内容与世界先验冲突时仍按视觉证据回答。
- 论文 Table 1 对 VLMBias 报告七个类别和 Overall 的 `Accuracy (%)`。
- Table 1 同时比较 `Regular`、`VCD`、`M3ID`、`RITUAL`、`DeGF`、`AGLA`、`EnAR`，括号中为相对 Regular 的变化。
- POPE 使用 `Accuracy / Precision / Recall / F1`，但 VLMBias 主指标是 accuracy。

因此，本项目 VLMBias 的主指标定义为：

```text
accuracy = correct_count / evaluated_count
```

并按以下粒度汇总：

- overall accuracy
- topic/category accuracy
- sub_topic accuracy
- type_of_question accuracy
- with_title 分组 accuracy
- pixel 分组 accuracy
- 相对 Regular 的 delta accuracy
- expected_bias 命中率，用于衡量模型是否落入数据集预设偏差答案

## 数据集输入

本地路径：

```text
EnAR/toy_dataset/VLMBias/
  README.md
  download_manifest.json
  data/
    main-*.parquet
    identification-*.parquet
    withtitle-*.parquet
    original-*.parquet
    remove_background_q1q2-*.parquet
    remove_background_q3-*.parquet
```

VLMBias 字段：

```text
image
ID
image_path
topic
sub_topic
prompt
ground_truth
expected_bias
with_title
type_of_question
pixel
metadata
```

推荐默认评估 `main` split，因为论文 Table 1 的类别报告更接近主评测场景，且 `lmms-eval` 官方任务当前也主要支持 main subset。其他 split 作为诊断扩展。

## 子项目目录规划

建议目录结构：

```text
EnAR/enar_eval/
  vlmbias_enar_eval_design.md        # 本文档
  vlmbias_eval_config.yaml           # 未来默认统一 YAML
  README.md                          # 未来用户入口说明
  __init__.py
  cli.py                             # 未来命令行入口
  config.py                          # 未来 YAML schema/dataclass
  dataset.py                         # VLMBias parquet reader 与样本过滤
  runner.py                          # 批量调度 Envision -> Attend -> Respond
  evaluator.py                       # answer normalization 与 metric 计算
  cache.py                           # 阶段缓存、resume、fingerprint
  reports.py                         # JSON/CSV/Markdown 汇总
  tests/
```

输出目录建议：

```text
EnAR/outputs/enar_eval/vlmbias/{run_name}/
  resolved_config.yaml
  dataset_manifest.json
  sample_index.jsonl
  samples/
    {sample_id}/
      input.png
      sample.json
      envision/
      attend/
      respond/
      result.json
  predictions.jsonl
  metrics.json
  metrics_by_topic.csv
  metrics_by_sub_topic.csv
  metrics_by_question_type.csv
  error_cases.jsonl
  report.md
```

## 统一 YAML 设计

统一 YAML 应同时控制数据、模型、运行时、三阶段参数、评估指标和输出。

草案如下：

```yaml
experiment:
  name: vlmbias_llava15_7b_enar
  run_name: run_001
  seed: 42
  output_dir: outputs/enar_eval/vlmbias
  resume: true
  overwrite: false
  save_intermediate: true

dataset:
  name: VLMBias
  root_dir: toy_dataset/VLMBias
  data_dir: toy_dataset/VLMBias/data
  split: main
  data_files:
    main: data/main-*.parquet
    identification: data/identification-*.parquet
    withtitle: data/withtitle-*.parquet
    original: data/original-*.parquet
    remove_background_q1q2: data/remove_background_q1q2-*.parquet
    remove_background_q3: data/remove_background_q3-*.parquet
  filters:
    max_samples: null
    sample_ids: []
    topics: []
    sub_topics: []
    type_of_question: []
    with_title: null
    pixel: null
  image_export:
    format: png
    write_input_image: true

models:
  llava_model_dir: pre_model/LLM/llava-1.5-7b-hf
  sd_model_dir: pre_model/DDIM/stable-diffusion-v1-5

runtime:
  device: auto
  dtype: float16
  num_workers: 1
  fail_fast: false
  log_interval: 1
  torch_compile: false

pipeline:
  methods:
    regular: true
    enar: true
  prompt:
    question_field: prompt
    envision_prompt: ""
    negative_prompt: ""
  stages:
    envision:
      enabled: true
      image:
        image_size: 512
      ddim:
        num_ddim_steps: 50
        inversion_step_T: 30
        guidance_scale: 1.0
      langevin:
        langevin_steps_M: 10
        sample_count_K: 10
        eta_start: 1.0e-2
        eta_end: 1.0e-4
        temperature_tau: 0.1
      runtime:
        debug: false

    attend:
      enabled: true
      model:
        vision_feature_select_strategy: default
        num_additional_image_tokens: 1
      attention:
        vision_layer_number: 6
        attention_top_ratio: 0.10
        uncertainty_top_ratio: 0.05
        padding_ratio_limit: 0.10
        uncertainty_weight: 1.0
      visualization:
        save_raw_arrays: false
        save_heatmaps: true
        save_patch_overlay: true
        save_mask_origin: true
        save_source_masks: true
        mask_origin_mode: binary
        mask_origin_alpha: 0.45

    respond:
      enabled: true
      generation:
        max_new_tokens: 64
        do_sample: true
        temperature: 1.0
        top_p: 1.0
        seed: 42
      contrastive:
        alpha: 1.0
        use_apc: true
        apc_beta: 0.1
        padding_strategy: zero_embedding
        save_decode_trace: false
      model:
        vision_feature_select_strategy: default
        num_additional_image_tokens: 1

evaluation:
  answer_normalization:
    lowercase: true
    strip_punctuation: true
    strip_articles: true
    number_word_to_digit: true
    extract_first_number_for_counting: true
  correctness:
    mode: exact_or_numeric
    count_questions_use_numeric_match: true
    allow_ground_truth_aliases: true
  metrics:
    overall_accuracy: true
    topic_accuracy: true
    sub_topic_accuracy: true
    question_type_accuracy: true
    with_title_accuracy: true
    expected_bias_rate: true
    delta_vs_regular: true
  reports:
    save_predictions_jsonl: true
    save_metrics_json: true
    save_csv_tables: true
    save_markdown_report: true
    save_error_cases: true
```

## 数据流设计

单样本输入：

```text
VLMBias row
  image -> input.png
  prompt -> question
  ground_truth -> label
  expected_bias -> bias target
```

单样本执行流：

```text
1. dataset.py 读取 row
2. 导出 row["image"] 为 samples/{sample_id}/input.png
3. runner.py 构造单样本 pipeline config
4. Envision 生成 visual impression 与 uncertainty map
5. Attend 生成 counterfactual token/patch mask
6. Respond 生成 regular_answer 与 enar_answer
7. evaluator.py 对两个答案分别打分
8. reports.py 写 result.json 和 predictions.jsonl
```

单样本 `result.json` 建议结构：

```json
{
  "sample_id": "xxx",
  "split": "main",
  "topic": "Animals",
  "sub_topic": "llama",
  "type_of_question": "counting",
  "prompt": "How many legs does this animal have?",
  "ground_truth": "5",
  "expected_bias": "4",
  "regular": {
    "answer": "4",
    "normalized_answer": "4",
    "correct": false,
    "hits_expected_bias": true
  },
  "enar": {
    "answer": "5",
    "normalized_answer": "5",
    "correct": true,
    "hits_expected_bias": false
  },
  "paths": {
    "input_image": "...",
    "envision": "...",
    "attend": "...",
    "respond": "..."
  },
  "status": "ok",
  "error": null
}
```

## 指标定义

### 主指标

VLMBias 主指标：

```text
Regular Accuracy = mean(correct_regular)
EnAR Accuracy = mean(correct_enar)
Delta Accuracy = EnAR Accuracy - Regular Accuracy
```

类别指标：

```text
Accuracy(topic=t) = correct_count(topic=t) / evaluated_count(topic=t)
```

Table 1 风格输出：

```text
Model: LLaVA-v1.5-7B
Method: Regular / EnAR
Columns: Animals, Chess Pieces, Flags, Game Boards, Logos, Optical Illusion, Patterned Grid, Overall
Cell: accuracy_percent
Delta cell: EnAR - Regular
```

### 偏差诊断指标

`expected_bias_rate`：

```text
expected_bias_rate = answers_equal_expected_bias / evaluated_count
```

该指标不是论文 Table 1 的主指标，但非常适合 VLMBias，因为数据集显式提供 `expected_bias`。它可以显示 EnAR 是否减少“按常识/记忆回答”的倾向。

建议报告：

```text
regular_expected_bias_rate
enar_expected_bias_rate
delta_expected_bias_rate = enar - regular
```

理想情况下，EnAR accuracy 上升，同时 expected_bias_rate 下降。

### 计数题匹配

VLMBias 大量题目是 counting。建议对 counting 类问题启用数字归一化：

```text
"five" -> "5"
"There are five legs." -> "5"
"5." -> "5"
```

若 `ground_truth` 是数字，优先抽取模型回答中的首个数字进行匹配。若回答中无数字，再退化为 normalized exact match。

### 开放答案匹配

对于 identification 或非数字答案：

1. 小写。
2. 去标点。
3. 去冠词。
4. 去多余空白。
5. exact match。
6. 可选 aliases，从 `metadata` 或手写映射表扩展。

首版建议保守使用 exact/numeric，不引入 LLM judge，保证指标可复现。

## 缓存与断点续跑

EnAR 三阶段开销较大，尤其 Envision。必须设计缓存。

缓存 key：

```text
sample_id
image_hash
question_hash
envision_config_hash
attend_config_hash
respond_config_hash
model_paths_hash
```

阶段级缓存：

```text
samples/{sample_id}/envision/metadata.json
samples/{sample_id}/attend/attend_result.json
samples/{sample_id}/respond/respond_result.json
samples/{sample_id}/result.json
```

`resume: true` 时：

- 如果 `result.json` 存在且 config hash 一致，跳过整样本。
- 如果 Envision 已完成但 Attend/Respond 未完成，复用 Envision。
- 如果 Attend 已完成但 Respond 未完成，复用 Attend。
- 如果 config hash 不一致，按 `overwrite` 决定重跑或报错。

## 与现有三阶段代码的关系

现有 `EnAR/pipeline/` 已有单图 YAML-driven runner：

```text
Envision -> Attend -> Respond
```

`enar_eval` 不应复制三阶段实现，而应作为批量评估调度层：

```text
enar_eval.runner
  -> pipeline.runner 或三个 stage pipeline
  -> evaluator
  -> reports
```

推荐首版直接复用 `pipeline.runner`，每个 VLMBias 样本动态生成一个单图配置。后续如需优化速度，再拆成 stage-level object reuse，避免每个样本重复加载 LLaVA/SD。

## 执行模式

### Smoke run

用于验证路径、依赖、输出格式：

```yaml
dataset:
  split: main
  filters:
    max_samples: 3
pipeline:
  stages:
    envision:
      ddim:
        num_ddim_steps: 10
        inversion_step_T: 5
      langevin:
        langevin_steps_M: 2
        sample_count_K: 2
```

### Full run

用于最终表格：

```yaml
dataset:
  split: main
  filters:
    max_samples: null
pipeline:
  stages:
    envision:
      ddim:
        num_ddim_steps: 50
        inversion_step_T: 30
      langevin:
        langevin_steps_M: 10
        sample_count_K: 10
```

### Ablation run

用于验证组件贡献：

- `regular_only`
- `enar_without_uncertainty`
- `enar_without_attention`
- `alpha = 0`
- `use_apc = false`
- different `vision_layer_number`
- different `padding_ratio_limit`

YAML 可通过 `experiment.name` 区分。

## 报告格式

`metrics.json`：

```json
{
  "dataset": "VLMBias",
  "split": "main",
  "num_total": 2784,
  "num_evaluated": 2784,
  "regular": {
    "overall_accuracy": 0.1692,
    "expected_bias_rate": 0.0
  },
  "enar": {
    "overall_accuracy": 0.2220,
    "expected_bias_rate": 0.0
  },
  "delta": {
    "overall_accuracy": 0.0528,
    "expected_bias_rate": 0.0
  },
  "by_topic": {}
}
```

`report.md` 建议包含：

1. 运行信息：run name、model、split、样本数、配置 hash。
2. Table 1 风格 VLMBias accuracy 表。
3. expected_bias_rate 表。
4. 按 topic/sub_topic 的 top improvements 与 regressions。
5. 错误案例索引。
6. 可视化样例链接：input、impression、uncertainty、mask overlay、regular/enar answer。

## 风险与注意事项

1. **VLMBias parquet 依赖**：需要 `datasets` 或 `pyarrow` 读取 parquet，当前环境可能需要补装。
2. **评估匹配策略会影响 accuracy**：counting 推荐 numeric match；开放答案首版应保持保守，避免过度宽松。
3. **模型输出随机性**：论文式配置使用 sampling。为可复现，需要固定 seed，并在报告中记录 `do_sample/temperature/top_p`。
4. **显存与耗时**：全量 main split 有 2784 条，Envision 成本最高；必须支持 max_samples、resume、阶段缓存。
5. **图像预处理一致性**：VLMBias 的 `image` 字段导出后，应作为三阶段统一输入，避免不同阶段读取不同尺寸或格式。
6. **category 名称对齐**：Table 1 使用七个类别，实际 `topic` 字段需要在首轮数据扫描后确认大小写和命名，并做标准化映射。

## 分阶段实施计划

### Phase 1: 数据与指标最小闭环

- 新增 `dataset.py` 读取 VLMBias parquet。
- 新增 `evaluator.py` 做 answer normalization、numeric match、accuracy。
- 新增 `reports.py` 生成 `predictions.jsonl` 与 `metrics.json`。
- 使用假预测或已有 Respond 输出验证指标。

### Phase 2: 批量 Regular baseline

- 接入 LLaVA regular generation。
- 跑 `max_samples: 3` smoke run。
- 输出 Regular accuracy 与错误案例。

### Phase 3: 接入完整 EnAR 三阶段

- 基于统一 YAML 为每个样本生成单图 pipeline config。
- 复用现有 `pipeline.runner`。
- 保存每个样本的 Envision/Attend/Respond 中间结果。
- 支持 resume 和 config hash。

### Phase 4: 表格与诊断

- 生成 Table 1 风格 topic accuracy。
- 增加 expected_bias_rate。
- 增加 category/sub_topic/type_of_question 分析。
- 生成 Markdown 报告。

### Phase 5: 性能优化与消融

- 避免每样本重复加载模型。
- 增加 ablation 配置。
- 增加并行/队列化策略，但默认保持 `num_workers: 1`，优先保证显存稳定。

## 建议的首个开发验收标准

首版可认为完成，当以下命令能跑通：

```bash
cd /home/qianustb/EnAR
PYTHONPATH=/home/qianustb/EnAR \
./env/bin/python -m enar_eval.cli \
  --config enar_eval/vlmbias_eval_config.yaml
```

并在输出目录得到：

```text
resolved_config.yaml
predictions.jsonl
metrics.json
report.md
samples/{sample_id}/result.json
```

其中 `metrics.json` 至少包含：

- regular overall accuracy
- enar overall accuracy
- delta accuracy
- topic accuracy
- expected_bias_rate
