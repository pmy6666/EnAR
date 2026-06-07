# EnAR VLMBias Evaluation

`enar_eval` reads the local VLMBias parquet files, exports each sample image, runs the existing single-image EnAR pipeline, and writes per-sample predictions plus aggregate metrics.

## Quick Start

Run from the `EnAR` project root:

```bash
cd /home/qianustb/EnAR
PYTHONPATH=/home/qianustb/EnAR ./env/bin/python -m enar_eval.cli \
  --config enar_eval/vlmbias_eval_config.yaml
```

The default config is a small smoke configuration with `max_samples: 3` and reduced Envision steps. For a full run, set `dataset.filters.max_samples: null` and restore the paper-style Envision parameters in `enar_eval/vlmbias_eval_config.yaml`.

## Dry Run

Use dry run to validate parquet reading, image export, resume, metrics, and report generation without loading SD or LLaVA:

```bash
cd /home/qianustb/EnAR
PYTHONPATH=/home/qianustb/EnAR ./env/bin/python -m enar_eval.cli \
  --config enar_eval/vlmbias_eval_config.yaml \
  --dry-run \
  --run-name dry_run_001 \
  --max-samples 3 \
  --overwrite
```

Dry run writes deterministic placeholder predictions: Regular uses `expected_bias`, and EnAR uses `ground_truth`. It is only an IO and metric smoke test.

## Useful Overrides

```bash
# Evaluate a different split
PYTHONPATH=/home/qianustb/EnAR ./env/bin/python -m enar_eval.cli \
  --config enar_eval/vlmbias_eval_config.yaml \
  --split identification \
  --max-samples 5

# Force recomputation of cached samples
PYTHONPATH=/home/qianustb/EnAR ./env/bin/python -m enar_eval.cli \
  --config enar_eval/vlmbias_eval_config.yaml \
  --overwrite

# Disable resume lookup
PYTHONPATH=/home/qianustb/EnAR ./env/bin/python -m enar_eval.cli \
  --config enar_eval/vlmbias_eval_config.yaml \
  --no-resume
```

## Outputs

Outputs are written to:

```text
outputs/enar_eval/vlmbias/{run_name}/
```

Default key files when `experiment.save_intermediate: false`:

- `resolved_config.yaml`: resolved absolute paths and final settings.
- `dataset_manifest.json`: parquet files, split, filters, and sample counts.
- `sample_index.jsonl`: metadata for every selected VLMBias sample.
- `samples/{sample_id}/result.json`: evaluated per-sample result.
- `predictions.jsonl`: all per-sample result rows.
- `metrics.json`: overall and grouped accuracy / expected-bias metrics.
- `metrics_by_*.csv`: topic, sub-topic, question-type, with-title, and pixel group tables.
- `error_cases.jsonl`: failed samples and samples where either method is wrong.
- `report.md`: human-readable summary.

When `experiment.save_intermediate: true`, each sample also keeps the exported `input.png`, generated `pipeline_config.yaml`, and full `pipeline/` directory with Envision, Attend, and Respond artifacts for visualization and debugging.

## Metrics

VLMBias uses accuracy as the main metric:

```text
accuracy = correct_count / evaluated_count
```

This evaluator reports:

- Regular accuracy
- EnAR accuracy
- EnAR minus Regular delta accuracy
- Topic, sub-topic, question-type, with-title, and pixel grouped accuracy
- `expected_bias_rate`, the rate at which answers match the dataset-provided `expected_bias`

Counting questions use numeric matching, so answers such as `five`, `5`, and `There are 5` can match the same numeric ground truth.

## Notes

- Normal mode reuses `pipeline.runner.EnARPipeline`; `enar_eval` does not duplicate Envision, Attend, or Respond logic.
- Resume is sample-level. If `samples/{sample_id}/result.json` has the same config hash and status `ok`, the sample is skipped.
- With `save_intermediate: false`, resume still works from `result.json`, but stage-level artifacts are removed after each sample.
- The current runner invokes the full single-image pipeline per sample. This favors correctness and isolation; later optimization can keep models loaded across samples.
