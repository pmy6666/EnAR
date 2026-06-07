# Logits Debug Scripts

These scripts inspect next-token logits/probabilities under different visual inputs. Edit the YAML next to each script for routine experiments; command-line flags can still override YAML values for one-off runs.

## Scripts

- `llava_only_interactive_top_p_debug.py`
  - Uses the native LLaVA path: `input_ids + pixel_values`.
  - Use this as the regular baseline.

- `interactive_respond_top_p_debug.py`
  - Uses Respond's manually merged original visual embeddings: `v`.
  - Use this to check whether Respond's original branch matches the native baseline.

- `interactive_padded_visual_top_p_debug.py`
  - Uses Respond's Attend-padded visual embeddings: `v_pad`.
  - Use this to compare padding strategies such as `pad_token_embedding`, `mean_visual_embedding`, `zero_embedding`, and `matched_mean_visual_embedding`.

## Run

From `/home/qianustb`:

```bash
PYTHONPATH=/home/qianustb/EnAR \
/home/qianustb/EnAR/env/bin/python EnAR/work_scripts/llava_only_interactive_top_p_debug.py
```

```bash
PYTHONPATH=/home/qianustb/EnAR \
/home/qianustb/EnAR/env/bin/python EnAR/work_scripts/interactive_respond_top_p_debug.py
```

```bash
PYTHONPATH=/home/qianustb/EnAR \
/home/qianustb/EnAR/env/bin/python EnAR/work_scripts/interactive_padded_visual_top_p_debug.py
```

To use a different YAML:

```bash
PYTHONPATH=/home/qianustb/EnAR \
/home/qianustb/EnAR/env/bin/python EnAR/work_scripts/interactive_padded_visual_top_p_debug.py \
  --script_config EnAR/work_scripts/interactive_padded_visual_top_p_debug.yaml
```

To override one field without editing YAML:

```bash
PYTHONPATH=/home/qianustb/EnAR \
/home/qianustb/EnAR/env/bin/python EnAR/work_scripts/interactive_padded_visual_top_p_debug.py \
  --padding_strategy mean_visual_embedding
```

## Recommended Comparison

Run in this order:

1. Native LLaVA baseline: `llava_only_interactive_top_p_debug.py`
2. Respond original branch `v`: `interactive_respond_top_p_debug.py`
3. Respond padded branch `v_pad`: `interactive_padded_visual_top_p_debug.py`

At each `next>` prompt, press Enter to accept the selected token and continue, or type `q` to stop after the current step.

Compare these columns:

- `logit`
- `p_full`
- `p_top_p`
- `token`

If native LLaVA and Respond original `v` differ strongly, inspect Respond's embedding merge path. If only `v_pad` differs strongly, inspect Attend indices and the selected padding strategy.

## Useful YAML Fields

- `image` / `image_path`: image being evaluated.
- `question` / `prompt`: question text.
- `attend_result_json`: Attend output used for selected visual token indices.
- `padding_strategy`: padded visual replacement strategy.
- `top_p`: nucleus filtering threshold used only for printed top-p rows and sampling.
- `temperature`: temperature used for printed probabilities and sampling.
- `force_greedy`: select argmax while still printing the top-p distribution.
- `limit`: number of rows to print; `0` prints the whole top-p set.
- `print_prompt_token_map`: prints mapping from prompt positions to visual token indices.
