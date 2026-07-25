import json

import pandas as pd
import pytest

from ml.preprocessing.pipeline import PreprocessingPipeline
from ml.preprocessing.split import stratified_split
from ml.training.run import run_training


@pytest.fixture
def processed_dataset(tmp_path, cicids_like_df):
    """Build real processed CSVs + preprocessing artifacts, mirroring `sentinel preprocess`."""
    split = stratified_split(cicids_like_df, label_column=" Label", random_state=42)
    pipeline = PreprocessingPipeline(label_column=" Label").fit(split.train)

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    for name, split_df in [("train", split.train), ("val", split.val), ("test", split.test)]:
        X, y = pipeline.transform(split_df)
        out = X.copy()
        out["Label"] = y
        out.to_csv(processed_dir / f"{name}.csv", index=False)

    artifacts_dir = tmp_path / "artifacts"
    pipeline.save(artifacts_dir)

    return processed_dir, artifacts_dir


def test_run_training_end_to_end(tmp_path, processed_dataset):
    processed_dir, artifacts_dir = processed_dataset
    models_dir = tmp_path / "models"

    outputs = run_training(
        processed_dir=processed_dir,
        preprocessing_artifacts_dir=artifacts_dir,
        models_dir=models_dir,
        n_estimators=10,
    )

    for path in outputs.values():
        assert path.exists()
    assert outputs["model"] == models_dir / "v1" / "model.joblib"

    metadata = json.loads(outputs["metadata"].read_text())
    assert metadata["model_type"] in {"random_forest", "xgboost", "lightgbm"}
    assert "accuracy" in metadata["metrics"]
    assert "f1_macro" in metadata["metrics"]

    summary = json.loads(outputs["training_summary"].read_text())
    assert set(summary["trained_models"]) >= {"random_forest", "xgboost"}
    assert summary["best_model_type"] == metadata["model_type"]
    for model_type in summary["trained_models"]:
        assert model_type in summary["val_metrics"]
        assert "confusion_matrix" in summary["val_metrics"][model_type]
        assert "classification_report" in summary["val_metrics"][model_type]
    assert "confusion_matrix" in summary["test_metrics"]

    for model_type in summary["trained_models"]:
        assert model_type in summary["feature_importances"]
        assert set(summary["feature_importances"][model_type]) == set(summary["feature_columns"])

    assert set(summary["class_distribution"]) == {"train", "val", "test"}
    for split, n_rows in [
        ("train", summary["n_train_rows"]),
        ("val", summary["n_val_rows"]),
        ("test", summary["n_test_rows"]),
    ]:
        assert sum(summary["class_distribution"][split].values()) == n_rows


def test_run_training_creates_figures_for_every_trained_model(tmp_path, processed_dataset):
    processed_dir, artifacts_dir = processed_dataset

    outputs = run_training(
        processed_dir=processed_dir,
        preprocessing_artifacts_dir=artifacts_dir,
        models_dir=tmp_path / "models",
        n_estimators=10,
    )

    summary = json.loads(outputs["training_summary"].read_text())
    for model_type in summary["trained_models"]:
        assert outputs[f"figure_confusion_matrix_{model_type}"].exists()
        assert outputs[f"figure_feature_importance_{model_type}"].exists()


def test_run_training_increments_version_across_runs(tmp_path, processed_dataset):
    processed_dir, artifacts_dir = processed_dataset
    models_dir = tmp_path / "models"

    first = run_training(
        processed_dir=processed_dir,
        preprocessing_artifacts_dir=artifacts_dir,
        models_dir=models_dir,
        n_estimators=10,
    )
    second = run_training(
        processed_dir=processed_dir,
        preprocessing_artifacts_dir=artifacts_dir,
        models_dir=models_dir,
        n_estimators=10,
    )

    assert first["model"].parent.name == "v1"
    assert second["model"].parent.name == "v2"


def test_run_training_never_refits_preprocessing(tmp_path, processed_dataset, monkeypatch):
    processed_dir, artifacts_dir = processed_dataset

    def _fail_if_called(self, train_df):
        raise AssertionError("PreprocessingPipeline.fit() must not be called during training")

    monkeypatch.setattr(PreprocessingPipeline, "fit", _fail_if_called)

    run_training(
        processed_dir=processed_dir,
        preprocessing_artifacts_dir=artifacts_dir,
        models_dir=tmp_path / "models",
        n_estimators=10,
    )  # must not raise


def test_run_training_raises_on_missing_expected_columns(tmp_path, processed_dataset):
    processed_dir, artifacts_dir = processed_dataset
    train_df = pd.read_csv(processed_dir / "train.csv")
    train_df.drop(columns=[train_df.columns[0]]).to_csv(processed_dir / "train.csv", index=False)

    with pytest.raises(ValueError, match="missing columns"):
        run_training(
            processed_dir=processed_dir,
            preprocessing_artifacts_dir=artifacts_dir,
            models_dir=tmp_path / "models",
            n_estimators=10,
        )


def test_run_training_respects_selection_metric(tmp_path, processed_dataset):
    processed_dir, artifacts_dir = processed_dataset

    outputs = run_training(
        processed_dir=processed_dir,
        preprocessing_artifacts_dir=artifacts_dir,
        models_dir=tmp_path / "models",
        n_estimators=10,
        selection_metric="accuracy",
    )

    summary = json.loads(outputs["training_summary"].read_text())
    val_metrics = summary["val_metrics"]
    best_accuracy = val_metrics[summary["best_model_type"]]["accuracy"]
    assert best_accuracy == max(m["accuracy"] for m in val_metrics.values())
