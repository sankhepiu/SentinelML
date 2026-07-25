import json

import pandas as pd

from ml.preprocessing.run import run_preprocessing


def test_run_preprocessing_end_to_end(tmp_path, cicids_like_df):
    input_path = tmp_path / "input.csv"
    cicids_like_df.to_csv(input_path, index=False)
    output_dir = tmp_path / "processed"
    artifacts_dir = tmp_path / "artifacts"

    outputs = run_preprocessing(
        input_path,
        output_dir=output_dir,
        artifacts_dir=artifacts_dir,
        label_column="Label",
    )

    for path in outputs.values():
        assert path.exists()

    train_df = pd.read_csv(outputs["train_csv"])
    val_df = pd.read_csv(outputs["val_csv"])
    test_df = pd.read_csv(outputs["test_csv"])

    assert "Label" in train_df.columns
    assert pd.api.types.is_integer_dtype(train_df["Label"])
    assert not train_df.drop(columns=["Label"]).isna().any().any()

    total_rows = len(train_df) + len(val_df) + len(test_df)
    assert total_rows < len(cicids_like_df)  # duplicates were removed before splitting

    summary = json.loads(outputs["run_summary"].read_text())
    assert summary["n_duplicate_rows_removed"] > 0
    assert summary["n_train_rows"] == len(train_df)
    assert summary["n_val_rows"] == len(val_df)
    assert summary["n_test_rows"] == len(test_df)

    metadata = json.loads(outputs["artifact_metadata"].read_text())
    assert "Fwd URG Flags" in metadata["dropped_constant_columns"]
    assert set(metadata["label_mapping"].values()) == set(cicids_like_df[" Label"].unique())


def test_run_preprocessing_artifacts_reproduce_processed_features(tmp_path, cicids_like_df):
    input_path = tmp_path / "input.csv"
    cicids_like_df.to_csv(input_path, index=False)
    output_dir = tmp_path / "processed"
    artifacts_dir = tmp_path / "artifacts"

    run_preprocessing(
        input_path, output_dir=output_dir, artifacts_dir=artifacts_dir, label_column="Label"
    )

    from ml.preprocessing.pipeline import PreprocessingPipeline

    pipeline = PreprocessingPipeline.load(artifacts_dir)
    test_df = pd.read_csv(input_path)
    test_df.columns = test_df.columns.str.strip()
    X, _ = pipeline.transform(test_df.iloc[:5])

    assert list(X.columns) == pipeline.metadata.feature_columns
