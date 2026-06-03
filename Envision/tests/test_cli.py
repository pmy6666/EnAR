from Envision.cli import build_parser


def test_cli_parser_accepts_core_args():
    parser = build_parser()
    args = parser.parse_args(["--input_image", "a.png", "--output_dir", "out", "--sample_count_K", "2"])
    assert str(args.input_image) == "a.png"
    assert args.sample_count_K == 2
