"""Settings.model_version lets an operator pin a specific trained-model
version instead of always resolving to the latest one under
model_registry_path -- verifies main.py's lifespan actually threads that
setting through to Predictor.from_registry().
"""

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from ml.preprocessing.pipeline import PreprocessingPipeline
from ml.training.run import run_training


def _train_second_version(trained_models_root, synthetic_flow_df):
    """Adds a v2 alongside the v1 the `trained_models_root` fixture already built."""
    from ml.preprocessing.split import stratified_split

    split = stratified_split(synthetic_flow_df, label_column="Label", random_state=7)
    preprocessing_dir = trained_models_root / "preprocessing"
    pipeline = PreprocessingPipeline.load(preprocessing_dir)

    processed_dir = trained_models_root.parent / "processed_v2"
    processed_dir.mkdir()
    for name, split_df in [("train", split.train), ("val", split.val), ("test", split.test)]:
        X, y = pipeline.transform(split_df)
        out = X.copy()
        out["Label"] = y
        out.to_csv(processed_dir / f"{name}.csv", index=False)

    run_training(
        processed_dir=processed_dir,
        preprocessing_artifacts_dir=preprocessing_dir,
        models_dir=trained_models_root,
        n_estimators=10,
    )


def test_pinned_model_version_overrides_latest(trained_models_root, synthetic_flow_df):
    _train_second_version(trained_models_root, synthetic_flow_df)
    assert (trained_models_root / "v2").exists()

    pinned_settings = Settings(model_registry_path=str(trained_models_root), model_version="v1")
    with TestClient(create_app(pinned_settings)) as client:
        assert client.get("/api/v1/model").json()["model_version"] == "v1"

    latest_settings = Settings(model_registry_path=str(trained_models_root))
    with TestClient(create_app(latest_settings)) as client:
        assert client.get("/api/v1/model").json()["model_version"] == "v2"
