from pathlib import Path

from Respond.output_writer import RespondOutputWriter


def test_output_writer_saves_result_and_trace(tmp_path: Path):
    writer = RespondOutputWriter(tmp_path)
    result_path = writer.save_result(
        {"regular_answer": "a", "enar_answer": "b"},
        [{"step": 0}],
        token_logits_trace=[
            {
                "step": 0,
                "origin": [{"rank": 1, "token_id": 1, "token": "a", "logit": 2.0}],
                "pad": [{"rank": 1, "token_id": 2, "token": "b", "logit": 1.0}],
            }
        ],
    )
    assert Path(result_path).is_file()
    assert (tmp_path / "decode_trace.json").is_file()
    assert (tmp_path / "token_logits_trace.json").is_file()
    assert writer.save_text("answer.txt", "hello").endswith("answer.txt")
