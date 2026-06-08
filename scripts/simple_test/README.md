# Small Dataset Simple Test

Use the project virtual environment:

```bash
EnAR/env/bin/python EnAR/scripts/simple_test/run_small_dataset_simple_test.py
```

Run in the background:

```bash
EnAR/scripts/simple_test/run_small_dataset_simple_test_bg.sh
```

The background script prints the PID and log path. You can pass the same filters through it:

```bash
EnAR/scripts/simple_test/run_small_dataset_simple_test_bg.sh --category Animals --max-samples 1 --run-name smoke_animals
```

Run a small smoke test before launching the full 70-sample set:

```bash
EnAR/env/bin/python EnAR/scripts/simple_test/run_small_dataset_simple_test.py --category Animals --max-samples 1 --run-name smoke_animals
```

Useful filters:

```bash
EnAR/env/bin/python EnAR/scripts/simple_test/run_small_dataset_simple_test.py --category Logos --max-samples 3
EnAR/env/bin/python EnAR/scripts/simple_test/run_small_dataset_simple_test.py --category "Game Boards"
EnAR/env/bin/python EnAR/scripts/simple_test/run_small_dataset_simple_test.py --sample-id car_066_notitle_Q2_px768
```

Recompute existing sample outputs:

```bash
EnAR/env/bin/python EnAR/scripts/simple_test/run_small_dataset_simple_test.py --overwrite --no-resume
```

Outputs are written under `EnAR/outputs/simple_test/<run_name>/`. Every sample keeps:

- `input.png`
- `sample.json`
- `pipeline_config.yaml`
- `pipeline/envision/`
- `pipeline/attend/`
- `pipeline/respond/`
- `pipeline/pipeline_result.json`

Summary files include `predictions.jsonl`, `metrics.json`, `answer_distribution.csv`, `metrics_by_*.csv`, and `report.md`.
