import pytest

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


def test_train_subcommand_requires_no_arguments():
    parser = _build_parser()
    args = parser.parse_args(["train"])

    assert args.command == "train"
    assert args.selection_metric == "f1_macro"


def test_train_subcommand_runs_end_to_end(tmp_path, cicids_like_df, capsys):
    csv_path = tmp_path / "sample.csv"
    cicids_like_df.to_csv(csv_path, index=False)
    processed_dir = tmp_path / "processed"
    preprocessing_artifacts_dir = tmp_path / "artifacts" / "preprocessing"
    models_dir = tmp_path / "artifacts"

    parser = _build_parser()
    preprocess_args = parser.parse_args(
        [
            "preprocess",
            "--input",
            str(csv_path),
            "--output-dir",
            str(processed_dir),
            "--artifacts-dir",
            str(preprocessing_artifacts_dir),
        ]
    )
    preprocess_args.func(preprocess_args)
    capsys.readouterr()  # discard preprocess output

    train_args = parser.parse_args(
        [
            "train",
            "--processed-dir",
            str(processed_dir),
            "--preprocessing-artifacts-dir",
            str(preprocessing_artifacts_dir),
            "--models-dir",
            str(models_dir),
            "--n-estimators",
            "10",
        ]
    )
    train_args.func(train_args)

    assert (models_dir / "v1" / "model.joblib").exists()
    assert (models_dir / "v1" / "metadata.json").exists()
    assert "Generated:" in capsys.readouterr().out


def test_serve_subcommand_defaults():
    parser = _build_parser()
    args = parser.parse_args(["serve"])

    assert args.command == "serve"
    assert args.host == "127.0.0.1"
    assert args.port == 8000
    assert args.reload is False


def test_serve_subcommand_accepts_overrides():
    parser = _build_parser()
    args = parser.parse_args(["serve", "--host", "0.0.0.0", "--port", "9000", "--reload"])

    assert args.host == "0.0.0.0"
    assert args.port == 9000
    assert args.reload is True


def test_serve_subcommand_invokes_uvicorn_with_expected_command(monkeypatch):
    captured = {}

    class _FakeProcess:
        def send_signal(self, signum):
            captured.setdefault("signals", []).append(signum)

        def wait(self):
            return 0

    def fake_popen(cmd, cwd):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        return _FakeProcess()

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    # Don't let _run_serve install real handlers on the pytest process itself.
    monkeypatch.setattr("signal.signal", lambda signum, handler: None)

    parser = _build_parser()
    args = parser.parse_args(["serve", "--host", "0.0.0.0", "--port", "9000"])
    args.func(args)

    assert captured["cmd"][:2] == ["uvicorn", "app.main:app"]
    assert "--host" in captured["cmd"] and "0.0.0.0" in captured["cmd"]
    assert "--port" in captured["cmd"] and "9000" in captured["cmd"]
    assert "--no-access-log" in captured["cmd"]
    assert "--reload" not in captured["cmd"]
    assert captured["cwd"].name == "backend"


def test_serve_subcommand_forwards_sigterm_to_uvicorn_and_exits_cleanly(monkeypatch):
    import signal

    class _FakeProcess:
        def __init__(self):
            self.signals_received: list[int] = []

        def send_signal(self, signum):
            self.signals_received.append(signum)

        def wait(self):
            return 0

    fake_process = _FakeProcess()
    monkeypatch.setattr("subprocess.Popen", lambda cmd, cwd: fake_process)

    registered_handlers = {}

    def fake_signal(signum, handler):
        registered_handlers[signum] = handler

    monkeypatch.setattr("signal.signal", fake_signal)

    parser = _build_parser()
    args = parser.parse_args(["serve"])
    args.func(args)

    assert signal.SIGTERM in registered_handlers
    assert signal.SIGINT in registered_handlers

    registered_handlers[signal.SIGTERM](signal.SIGTERM, None)
    assert fake_process.signals_received == [signal.SIGTERM]


def test_serve_subcommand_exits_nonzero_when_uvicorn_exits_nonzero(monkeypatch):
    class _FakeProcess:
        def send_signal(self, signum):
            pass

        def wait(self):
            return 3

    monkeypatch.setattr("subprocess.Popen", lambda cmd, cwd: _FakeProcess())
    monkeypatch.setattr("signal.signal", lambda signum, handler: None)

    parser = _build_parser()
    args = parser.parse_args(["serve"])

    with pytest.raises(SystemExit) as exc_info:
        args.func(args)
    assert exc_info.value.code == 3
