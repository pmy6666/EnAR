# EnAR

EnAR 是一个面向大视觉语言模型反事实幻觉缓解的复现与扩展项目。项目围绕
`Envision -> Attend -> Respond` 三阶段组织代码：先用扩散模型生成视觉印象和不确定性图，再在 LLaVA
视觉编码器中定位高风险视觉 token，最后通过 token-level padded visual input 做对比解码，得到 Regular
baseline 与 EnAR answer。

当前仓库已经包含单图三阶段 pipeline、阶段级调试入口、VLMBias 批量评估入口，以及若干 logits debug
脚本。大模型权重、运行输出、PDF、数据集 parquet 等本地资产不会提交到 Git。

## 项目结构

```text
EnAR/
  Envision/              # Stage 1: visual impression 与 uncertainty map
  Attend/                # Stage 2: attention/uncertainty 融合并选择视觉 patch/token
  Respond/               # Stage 3: regular decoding 与 EnAR contrastive decoding
  pipeline/              # 单图 Envision -> Attend -> Respond YAML 总入口
  enar_eval/             # VLMBias 批量评估、指标统计和报告生成
  work_scripts/          # 模型下载、调试和 logits 检查脚本
  pre_model/             # 本地模型目录；权重文件被 .gitignore 排除
  outputs/               # 运行输出目录；被 .gitignore 排除
  docs/                  # 复现计划、诊断报告和阶段设计文档
  VCD/my_work/           # VCD 复现实验中的本项目自写脚本
```

更细的模块说明请看：

- [Envision/README.md](Envision/README.md)
- [Attend/README.md](Attend/README.md)
- [Respond/README.md](Respond/README.md)
- [pipeline/README.md](pipeline/README.md)
- [enar_eval/README.md](enar_eval/README.md)

## 环境准备

推荐使用项目内已有环境：

```bash
cd /home/qianustb/EnAR
./env/bin/python --version
```

如果需要重建环境，核心依赖包括：

```bash
./env/bin/python -m pip install -U \
  torch torchvision diffusers transformers accelerate safetensors \
  sentencepiece protobuf pillow numpy PyYAML pytest pyarrow pandas
```

运行完整 pipeline 需要一张可用 GPU。CPU 模式主要用于单元测试或小模块调试；如果在 CPU 上运行模型，请把配置里的
`dtype` 改为 `float32`。

## 模型准备

默认模型路径如下：

```text
pre_model/LLM/llava-1.5-7b-hf/
pre_model/DDIM/stable-diffusion-v1-5/
```

权重文件不会进入 Git。可以使用仓库内下载脚本准备本地模型：

```bash
cd /home/qianustb/EnAR
bash work_scripts/download_llava_1_5_7b_modelscope.sh
bash work_scripts/download_sd_v1_5_ddim.sh
```

下载参数分别位于：

```text
pre_model/LLM/llava_1_5_7b_modelscope.params.env
pre_model/DDIM/stable_diffusion_v1_5.params.env
```

## 快速运行单图完整流程

先编辑 [pipeline/pipeline_config.yaml](pipeline/pipeline_config.yaml)，确认输入图、问题、模型路径和输出目录：

```yaml
paths:
  input_image: EnAR/Envision/image/data/wolf_5.png
  output_dir: outputs/pipeline
  sd_model_dir: pre_model/DDIM/stable-diffusion-v1-5
  llava_model_dir: pre_model/LLM/llava-1.5-7b-hf

prompt:
  question: How many legs does this animal have?
```

从项目根目录运行：

```bash
cd /home/qianustb/EnAR
PYTHONPATH=/home/qianustb/EnAR \
./env/bin/python -m pipeline.cli --config pipeline/pipeline_config.yaml
```

输出会写到：

```text
outputs/pipeline/{run_name}/
  envision/
  attend/
  respond/
  pipeline_result.json
```

其中 `respond/answer_regular.txt` 是 LLaVA regular baseline，`respond/answer_enar.txt` 是 EnAR 对比解码结果。

## 分阶段运行

如果要单独调试某一阶段，可以分别运行：

```bash
# Envision: 生成 visual impression 与 uncertainty map
PYTHONPATH=/home/qianustb/EnAR \
./env/bin/python -m Envision.cli --config Envision/envision_config.yaml

# Attend: 根据 Envision 输出选择高风险 patch/token
PYTHONPATH=/home/qianustb/EnAR \
./env/bin/python -m Attend.cli --config Attend/attend_config.yaml

# Respond: 使用 Attend 结果进行 regular 与 EnAR decoding
PYTHONPATH=/home/qianustb/EnAR \
./env/bin/python -m Respond.cli --config Respond/respond_config.yaml
```

阶段间主要产物流如下：

```text
Envision/metadata.json
  -> impression_image, uncertainty_map
  -> Attend/attend_result.json
  -> selected_vision_token_indices
  -> Respond/answer_regular.txt + answer_enar.txt
```

## VLMBias 评估

`enar_eval` 会读取本地 VLMBias parquet，逐样本调用单图 pipeline，并输出 predictions、metrics 和 Markdown
报告。

```bash
cd /home/qianustb/EnAR
PYTHONPATH=/home/qianustb/EnAR \
./env/bin/python -m enar_eval.cli --config enar_eval/vlmbias_eval_config.yaml
```

用于只验证 IO、指标和报告生成的 dry run：

```bash
PYTHONPATH=/home/qianustb/EnAR \
./env/bin/python -m enar_eval.cli \
  --config enar_eval/vlmbias_eval_config.yaml \
  --dry-run \
  --run-name dry_run_001 \
  --max-samples 3 \
  --overwrite
```

评估输出默认位于：

```text
outputs/enar_eval/vlmbias/{run_name}/
  predictions.jsonl
  metrics.json
  metrics_by_topic.csv
  error_cases.jsonl
  report.md
```

VLMBias 本地数据默认放在 `toy_dataset/VLMBias/`，其中 parquet 文件体积较大，已被 `.gitignore` 排除。

## 调试脚本

`work_scripts/` 中提供若干常用脚本：

- `download_llava_1_5_7b_modelscope.sh`：下载 LLaVA-1.5-7B HF 格式权重。
- `download_sd_v1_5_ddim.sh`：下载 Stable Diffusion v1.5 权重。
- `llava_only_interactive_top_p_debug.py`：检查原生 LLaVA next-token logits。
- `interactive_respond_top_p_debug.py`：检查 Respond 原始 visual branch。
- `interactive_padded_visual_top_p_debug.py`：检查 Attend-padded visual branch。
- `validate_enar_run.py`：检查一次 EnAR run 的关键产物是否齐全。

logits 调试说明见 [work_scripts/README_logits_debug.md](work_scripts/README_logits_debug.md)。

## 测试

单元测试使用 fake 组件，不会加载 LLaVA 或 Stable Diffusion 大模型：

```bash
cd /home/qianustb/EnAR
PYTHONPATH=/home/qianustb/EnAR \
./env/bin/python -m pytest Envision/tests Attend/tests Respond/tests -q
```

语法检查示例：

```bash
./env/bin/python -m py_compile Respond/*.py enar_eval/*.py pipeline/*.py
bash -n VCD/my_work/run_llava_vcd_single.sh
```

## Git 管理约定

以下内容保留在本地，不提交到 GitHub：

- `outputs/` 和各类 `**/outputs/`
- `env/`、`__pycache__/`、`.pytest_cache/`
- 模型权重：`*.safetensors`、`*.bin`、`*.ckpt` 等
- VLMBias parquet 数据：`toy_dataset/`
- 论文 PDF：`*.pdf`
- 第三方 `VCD/` 主体；仓库只保留 `VCD/my_work/` 下的自写复现实验脚本

## 推荐阅读顺序

1. 先看 [pipeline/pipeline_config.yaml](pipeline/pipeline_config.yaml)，理解完整流程的输入和阶段参数。
2. 再看 [Envision/README.md](Envision/README.md)、[Attend/README.md](Attend/README.md)、[Respond/README.md](Respond/README.md)，分别理解三阶段产物。
3. 如果要跑 VLMBias，继续看 [enar_eval/README.md](enar_eval/README.md) 和 [enar_eval/vlmbias_eval_config.yaml](enar_eval/vlmbias_eval_config.yaml)。
4. 如果要排查生成差异，使用 `work_scripts/` 下的 logits debug 脚本逐分支比较。
