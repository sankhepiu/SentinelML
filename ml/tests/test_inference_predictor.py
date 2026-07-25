import pandas as pd
import pytest

from ml.inference.predictor import ModelMetadata, Predictor
from ml.preprocessing.pipeline import PreprocessingPipeline
from ml.preprocessing.split import stratified_split
from ml.training.run import run_training


@pytest.fixture
def trained_artifacts(tmp_path, cicids_like_df):
    """Run the real M2+M3 pipelines against the shared fixture.

    Returns (models_root, raw_test_df) where raw_test_df still has original
    (unscaled) feature values plus " Label" -- i.e. what a real inference
    caller's raw data looks like before any preprocessing.
    """
    split = stratified_split(cicids_like_df, label_column=" Label", random_state=42)
    pipeline = PreprocessingPipeline(label_column=" Label").fit(split.train)

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    for name, split_df in [("train", split.train), ("val", split.val), ("test", split.test)]:
        X, y = pipeline.transform(split_df)
        out = X.copy()
        out["Label"] = y
        out.to_csv(processed_dir / f"{name}.csv", index=False)

    models_root = tmp_path / "artifacts"
    preprocessing_dir = models_root / "preprocessing"
    pipeline.save(preprocessing_dir)

    run_training(
        processed_dir=processed_dir,
        preprocessing_artifacts_dir=preprocessing_dir,
        models_dir=models_root,
        n_estimators=10,
    )

    return models_root, split.test


def test_from_registry_loads_latest_version(trained_artifacts):
    models_root, _ = trained_artifacts

    predictor = Predictor.from_registry(models_root)

    assert predictor.is_loaded
    assert isinstance(predictor.metadata, ModelMetadata)
    assert predictor.metadata.version == "v1"


def test_from_registry_resolves_explicit_version(trained_artifacts):
    models_root, _ = trained_artifacts

    predictor = Predictor.from_registry(models_root, version="v1")

    assert predictor.metadata.version == "v1"


def test_preprocessing_dir_defaults_to_sibling_directory(trained_artifacts):
    models_root, _ = trained_artifacts

    predictor = Predictor(models_root / "v1").load()

    assert predictor.preprocessing_dir == models_root / "preprocessing"
    assert predictor.is_loaded


def test_feature_names_matches_metadata(trained_artifacts):
    models_root, _ = trained_artifacts
    predictor = Predictor.from_registry(models_root)

    assert predictor.feature_names == predictor.metadata.feature_names


def test_label_classes_are_the_original_class_names(trained_artifacts):
    models_root, test_df = trained_artifacts
    predictor = Predictor.from_registry(models_root)

    assert set(predictor.label_classes) == set(test_df[" Label"].unique())


def test_predict_returns_one_result_per_row_with_valid_probabilities(trained_artifacts):
    models_root, test_df = trained_artifacts
    predictor = Predictor.from_registry(models_root)

    raw_features = test_df[predictor.feature_names]
    results = predictor.predict(raw_features)

    assert len(results) == len(raw_features)
    for result in results:
        assert result.predicted_class in predictor.label_classes
        assert 0.0 <= result.confidence <= 1.0
        assert result.class_probabilities is not None
        assert set(result.class_probabilities) == set(predictor.label_classes)
        assert abs(sum(result.class_probabilities.values()) - 1.0) < 1e-6


def test_predict_proba_shape(trained_artifacts):
    models_root, test_df = trained_artifacts
    predictor = Predictor.from_registry(models_root)

    raw_features = test_df[predictor.feature_names]
    proba = predictor.predict_proba(raw_features)

    assert proba.shape == (len(raw_features), len(predictor.label_classes))


def test_predict_does_not_require_a_label_column(trained_artifacts):
    models_root, test_df = trained_artifacts
    predictor = Predictor.from_registry(models_root)

    raw_features = test_df[predictor.feature_names]  # no label column at all
    results = predictor.predict(raw_features)  # must not raise

    assert len(results) == len(raw_features)


def test_predict_never_refits_preprocessing(trained_artifacts, monkeypatch):
    models_root, test_df = trained_artifacts

    def _fail_if_called(self, train_df):
        raise AssertionError("PreprocessingPipeline.fit() must not be called during inference")

    monkeypatch.setattr(PreprocessingPipeline, "fit", _fail_if_called)

    predictor = Predictor.from_registry(models_root)
    predictor.predict(test_df[predictor.feature_names])  # must not raise


def test_predict_raises_before_load():
    predictor = Predictor(model_dir="unused")

    with pytest.raises(RuntimeError, match="must be load"):
        predictor.predict(pd.DataFrame())


def test_feature_names_raises_before_load():
    predictor = Predictor(model_dir="unused")

    with pytest.raises(RuntimeError, match="must be load"):
        _ = predictor.feature_names


def test_load_raises_when_artifacts_missing(tmp_path):
    (tmp_path / "preprocessing").mkdir()
    (tmp_path / "v1").mkdir()

    with pytest.raises(FileNotFoundError):
        Predictor(tmp_path / "v1", preprocessing_dir=tmp_path / "preprocessing").load()


def test_training_summary_contains_expected_sections(trained_artifacts):
    models_root, _ = trained_artifacts
    predictor = Predictor.from_registry(models_root)

    summary = predictor.training_summary

    assert summary is not None
    assert summary["best_model_type"] == predictor.metadata.model_type
    assert set(summary["trained_models"]) >= {"random_forest", "xgboost"}
    assert summary["best_model_type"] in summary["feature_importances"]
    assert set(summary["feature_importances"][summary["best_model_type"]]) == set(
        predictor.feature_names
    )
    assert set(summary["class_distribution"]) == {"train", "val", "test"}
    assert set(summary["class_distribution"]["test"]) <= set(predictor.label_classes)


def test_training_summary_is_none_when_file_missing(trained_artifacts):
    models_root, _ = trained_artifacts
    predictor = Predictor.from_registry(models_root)
    (predictor.model_dir / "training_summary.json").unlink()

    assert predictor.training_summary is None


def test_training_summary_raises_before_load():
    predictor = Predictor(model_dir="unused")

    with pytest.raises(RuntimeError, match="must be load"):
        _ = predictor.training_summary
