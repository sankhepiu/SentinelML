"""Shared fixtures for backend API tests.

Builds a tiny, real (not mocked) trained-model registry -- a fitted
preprocessing pipeline plus a fast RandomForest -- so tests exercise the
actual `ml.inference.Predictor` code path without depending on the real
CICIDS2017 dataset or a long training run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from ml.preprocessing.pipeline import PreprocessingPipeline
from ml.preprocessing.split import stratified_split
from ml.training.run import run_training

N_ROWS = 300


@pytest.fixture
def synthetic_flow_df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "Destination Port": rng.integers(1, 65535, size=N_ROWS),
            "Flow Duration": rng.integers(1, 1_000_000, size=N_ROWS).astype(float),
            "Total Fwd Packets": rng.integers(1, 50, size=N_ROWS).astype(float),
            "Flow Bytes/s": rng.uniform(0, 1e6, size=N_ROWS),
            "Label": rng.choice(["BENIGN"] * 70 + ["DDoS"] * 20 + ["PortScan"] * 10, size=N_ROWS),
        }
    )


@pytest.fixture
def trained_models_root(tmp_path, synthetic_flow_df) -> Path:
    """A real, fitted preprocessing pipeline plus one trained model under `tmp_path`."""
    split = stratified_split(synthetic_flow_df, label_column="Label", random_state=42)
    pipeline = PreprocessingPipeline(label_column="Label").fit(split.train)

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

    return models_root


@pytest.fixture
def ready_client(trained_models_root) -> TestClient:
    """A TestClient whose app successfully loads a real model at startup."""
    settings = Settings(model_registry_path=str(trained_models_root))
    app = create_app(settings)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def not_ready_client(tmp_path) -> TestClient:
    """A TestClient whose app has no model to load (empty registry)."""
    settings = Settings(model_registry_path=str(tmp_path / "empty"))
    app = create_app(settings)
    with TestClient(app) as client:
        yield client
