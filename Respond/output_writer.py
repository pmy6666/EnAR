from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RespondOutputWriter:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_text(self, filename: str, text: str) -> str:
        path = self.output_dir / filename
        path.write_text(text, encoding="utf-8")
        return str(path)

    def save_json(self, filename: str, data: dict[str, Any]) -> str:
        path = self.output_dir / filename
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return str(path)

    def save_result(
        self,
        result_data: dict[str, Any],
        decode_trace: list[dict[str, Any]],
        save_trace: bool = True,
        token_logits_trace: list[dict[str, Any]] | None = None,
    ) -> str:
        if save_trace:
            result_data["decode_trace_path"] = self.save_json("decode_trace.json", {"steps": decode_trace})
        if token_logits_trace is not None:
            result_data["token_logits_trace_path"] = self.save_json(
                "token_logits_trace.json",
                {
                    "description": "Top-20 next-token logits for each EnAR decoding step, saved for both original and padded visual branches.",
                    "branches": {
                        "origin": "logit_theta(y | x, v, y_<t)",
                        "pad": "logit_theta(y | x, v_pad, y_<t)",
                    },
                    "top_k": 20,
                    "steps": token_logits_trace,
                },
            )
        return self.save_json("respond_result.json", result_data)
