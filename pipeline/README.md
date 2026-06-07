# EnAR Full Pipeline

This directory contains a YAML-driven runner for the full Envision -> Attend -> Respond flow.

Run it from the `EnAR` project root:

```bash
python -m pipeline.cli --config pipeline/pipeline_config.yaml
```

The YAML defines the input image, user question, output directory, model paths, and per-stage parameters. Stage outputs are wired automatically:

- `envision/` writes the original image, visual impression, uncertainty map, and metadata.
- `attend/` consumes Envision outputs and writes the selected counterfactual region plus `attend_result.json`.
- `respond/` consumes the original image and Attend result, then writes regular and EnAR answers.

The final summary is saved as `pipeline_result.json` under the configured run directory.
