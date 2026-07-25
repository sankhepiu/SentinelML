import numpy as np
import pandas as pd
import pytest

from ml.preprocessing.pipeline import PreprocessingPipeline
from ml.preprocessing.split import stratified_split


@pytest.fixture
def fitted_pipeline_and_split(cicids_like_df):
    split = stratified_split(cicids_like_df, label_column=" Label", random_state=42)
    pipeline = PreprocessingPipeline(label_column=" Label").fit(split.train)
    return pipeline, split


def test_fit_drops_constant_and_low_variance_columns(fitted_pipeline_and_split):
    pipeline, _ = fitted_pipeline_and_split

    metadata = pipeline.metadata
    assert " Fwd URG Flags" in metadata.dropped_constant_columns
    assert " Bwd PSH Flags" in metadata.dropped_low_variance_columns
    assert " Fwd URG Flags" not in metadata.feature_columns
    assert " Bwd PSH Flags" not in metadata.feature_columns
    assert " Label" not in metadata.feature_columns


def test_transform_returns_expected_shapes(fitted_pipeline_and_split):
    pipeline, split = fitted_pipeline_and_split

    X_test, y_test = pipeline.transform(split.test)

    assert len(X_test) == len(split.test)
    assert len(y_test) == len(split.test)
    assert list(X_test.columns) == pipeline.metadata.feature_columns


def test_transform_produces_no_missing_or_infinite_values(fitted_pipeline_and_split):
    pipeline, split = fitted_pipeline_and_split

    X_test, _ = pipeline.transform(split.test)

    assert not X_test.isna().any().any()
    assert not np.isinf(X_test.to_numpy()).any()


def test_label_encoding_round_trips_through_metadata(fitted_pipeline_and_split):
    pipeline, split = fitted_pipeline_and_split

    _, y_test = pipeline.transform(split.test)
    metadata = pipeline.metadata

    decoded = [metadata.label_mapping[str(code)] for code in y_test]
    assert set(decoded) == set(split.test[" Label"].unique())


def test_does_not_leak_val_test_statistics_into_scaling():
    train_df = pd.DataFrame(
        {
            "feature": [10.0, 10.0, 12.0, 8.0] * 5,
            "Label": (["BENIGN"] * 2 + ["ATTACK"] * 2) * 5,
        }
    )
    out_of_distribution_df = pd.DataFrame(
        {"feature": [10_000.0] * 4, "Label": ["BENIGN", "BENIGN", "ATTACK", "ATTACK"]}
    )

    pipeline = PreprocessingPipeline(label_column="Label").fit(train_df)
    X_train_before, _ = pipeline.transform(train_df)

    X_ood, _ = pipeline.transform(out_of_distribution_df)
    assert X_ood["feature"].abs().min() > 100  # wildly out-of-distribution vs. train scaling

    X_train_after, _ = pipeline.transform(train_df)
    pd.testing.assert_frame_equal(X_train_before, X_train_after)


def test_missing_and_infinite_values_are_imputed_from_train_median():
    train_df = pd.DataFrame(
        {
            "feature": [10.0, 20.0, 30.0, np.inf, np.nan] * 4,
            "Label": (["BENIGN"] * 3 + ["ATTACK"] * 2) * 4,
        }
    )

    pipeline = PreprocessingPipeline(label_column="Label").fit(train_df)

    assert pipeline._imputer.statistics_[0] == pytest.approx(20.0)


def test_save_and_load_round_trip_produces_identical_transform(tmp_path, fitted_pipeline_and_split):
    pipeline, split = fitted_pipeline_and_split

    pipeline.save(tmp_path)
    loaded = PreprocessingPipeline.load(tmp_path)

    X_original, y_original = pipeline.transform(split.test)
    X_loaded, y_loaded = loaded.transform(split.test)

    pd.testing.assert_frame_equal(X_original, X_loaded)
    assert list(y_original) == list(y_loaded)
    assert loaded.metadata == pipeline.metadata


def test_save_writes_expected_artifact_files(tmp_path, fitted_pipeline_and_split):
    pipeline, _ = fitted_pipeline_and_split

    paths = pipeline.save(tmp_path)

    for path in paths.values():
        assert path.exists()
    assert (tmp_path / "imputer.joblib").exists()
    assert (tmp_path / "scaler.joblib").exists()
    assert (tmp_path / "label_encoder.joblib").exists()
    assert (tmp_path / "metadata.json").exists()


def test_transform_before_fit_raises():
    pipeline = PreprocessingPipeline()

    with pytest.raises(RuntimeError, match="must be fit"):
        pipeline.transform(pd.DataFrame({"a": [1]}))


def test_metadata_before_fit_raises():
    pipeline = PreprocessingPipeline()

    with pytest.raises(RuntimeError, match="must be fit"):
        _ = pipeline.metadata


def test_fit_raises_when_every_feature_column_is_dropped():
    df = pd.DataFrame(
        {
            "constant": [1.0] * 10,
            "Label": ["BENIGN"] * 5 + ["ATTACK"] * 5,
        }
    )

    with pytest.raises(ValueError, match="No feature columns remain"):
        PreprocessingPipeline(label_column="Label").fit(df)
