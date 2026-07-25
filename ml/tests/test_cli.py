from ml.cli import _build_parser


def test_profile_subcommand_requires_input():
    parser = _build_parser()
    args = parser.parse_args(["profile", "--input", "some.csv"])

    assert args.command == "profile"
    assert str(args.input) == "some.csv"
    assert args.label_column == "Label"


def test_profile_subcommand_runs_end_to_end(tmp_path, cicids_like_df, capsys):
    csv_path = tmp_path / "sample.csv"
    cicids_like_df.to_csv(csv_path, index=False)
    reports_dir = tmp_path / "reports"

    parser = _build_parser()
    args = parser.parse_args(
        [
            "profile",
            "--input",
            str(csv_path),
            "--reports-dir",
            str(reports_dir),
            "--label-column",
            "Label",
        ]
    )
    args.func(args)

    assert (reports_dir / "data_profile_report.md").exists()
    assert "Generated:" in capsys.readouterr().out


def test_preprocess_subcommand_requires_input():
    parser = _build_parser()
    args = parser.parse_args(["preprocess", "--input", "some.csv"])

    assert args.command == "preprocess"
    assert str(args.input) == "some.csv"
    assert args.train_size == 0.7
    assert args.val_size == 0.15
    assert args.test_size == 0.15


def test_preprocess_subcommand_runs_end_to_end(tmp_path, cicids_like_df, capsys):
    csv_path = tmp_path / "sample.csv"
    cicids_like_df.to_csv(csv_path, index=False)
    output_dir = tmp_path / "processed"
    artifacts_dir = tmp_path / "artifacts"

    parser = _build_parser()
    args = parser.parse_args(
        [
            "preprocess",
            "--input",
            str(csv_path),
            "--output-dir",
            str(output_dir),
            "--artifacts-dir",
            str(artifacts_dir),
            "--label-column",
            "Label",
        ]
    )
    args.func(args)

    assert (output_dir / "train.csv").exists()
    assert (artifacts_dir / "metadata.json").exists()
    assert "Generated:" in capsys.readouterr().out
