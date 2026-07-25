"""Milestone 2 preprocessing pipeline: load -> dedup -> split -> fit -> persist.

Orchestrates the individual `ml.preprocessing` components into the single
entry point the CLI calls:

    uv run sentinel preprocess --input ml/data/raw/Wednesday-workingHours.pcap_ISCX.csv
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from ml.data.feature_stats import DEFAULT_LOW_VARIANCE_THRESHOLD
from ml.data.loader import load_cicids_csv
from ml.preprocessing.cleaning import drop_duplicate_rows
from ml.preprocessing.pipeline import PreprocessingPipeline
from ml.preprocessing.split import (
    DEFAULT_TEST_SIZE,
    DEFAULT_TRAIN_SIZE,
    DEFAULT_VAL_SIZE,
    stratified_split,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_DIR = REPO_ROOT / "ml" / "data" / "processed"
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "ml" / "models" / "artifacts" / "preprocessing"


@dataclass
class PreprocessingRunSummary:
    source_path: str
    label_column: str
    n_rows_input: int
    n_rows_after_dedup: int
    n_duplicate_rows_removed: int
    n_train_rows: int
    n_val_rows: int
    n_test_rows: int
    train_size: float
    val_size: float
    test_size: float
    random_state: int


def _write_split_csv(X: pd.DataFrame, y, label_column: str, path: Path) -> Path:
    out = X.copy()
    out[label_column] = y
    out.to_csv(path, index=False)
    return path


def run_preprocessing(
    input_path: str | Path,
    *,
    output_dir: str | Path = DEFAULT_PROCESSED_DIR,
    artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
    label_column: str = "Label",
    train_size: float = DEFAULT_TRAIN_SIZE,
    val_size: float = DEFAULT_VAL_SIZE,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = 42,
    low_variance_threshold: float = DEFAULT_LOW_VARIANCE_THRESHOLD,
) -> dict[str, Path]:
    """Run the full M2 pipeline against the CSV at `input_path`.

    Preprocessing state (imputer, scaler, label encoder, dropped columns)
    is fit on the training split only, then replayed on validation/test.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    artifacts_dir = Path(artifacts_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_cicids_csv(input_path)
    n_rows_input = len(df)

    df = drop_duplicate_rows(df)
    n_rows_after_dedup = len(df)

    split = stratified_split(
        df,
        label_column=label_column,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
    )

    pipeline = PreprocessingPipeline(
        label_column=label_column, low_variance_threshold=low_variance_threshold
    )
    X_train, y_train = pipeline.fit_transform(split.train)
    X_val, y_val = pipeline.transform(split.val)
    X_test, y_test = pipeline.transform(split.test)

    outputs: dict[str, Path] = {
        "train_csv": _write_split_csv(X_train, y_train, label_column, output_dir / "train.csv"),
        "val_csv": _write_split_csv(X_val, y_val, label_column, output_dir / "val.csv"),
        "test_csv": _write_split_csv(X_test, y_test, label_column, output_dir / "test.csv"),
    }
    outputs.update(
        {f"artifact_{name}": path for name, path in pipeline.save(artifacts_dir).items()}
    )

    summary = PreprocessingRunSummary(
        source_path=str(input_path),
        label_column=label_column,
        n_rows_input=n_rows_input,
        n_rows_after_dedup=n_rows_after_dedup,
        n_duplicate_rows_removed=n_rows_input - n_rows_after_dedup,
        n_train_rows=len(split.train),
        n_val_rows=len(split.val),
        n_test_rows=len(split.test),
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
    )
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(asdict(summary), indent=2))
    outputs["run_summary"] = summary_path

    return outputs
