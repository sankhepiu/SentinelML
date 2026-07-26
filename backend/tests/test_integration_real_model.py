"""End-to-end integration test against the REAL trained model artifacts.

`ml/models/artifacts/v1` is committed to the repo, so this runs in CI too;
the skip guard only matters for a checkout where it's been deliberately
removed or replaced. Every other backend test uses synthetic artifacts
(see conftest.py) so the suite doesn't depend on this.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

_REAL_MODEL_DIR = Path(__file__).resolve().parents[2] / "ml" / "models" / "artifacts" / "v1"


@pytest.mark.skipif(
    not _REAL_MODEL_DIR.exists(), reason="Real trained model artifacts not present locally"
)
def test_full_app_serves_the_real_trained_model():
    with TestClient(app) as client:
        health = client.get("/api/v1/health")
        assert health.status_code == 200

        ready = client.get("/api/v1/ready")
        assert ready.status_code == 200
        assert ready.json()["ready"] is True

        model_info = client.get("/api/v1/model")
        assert model_info.status_code == 200
        body = model_info.json()
        assert body["model_type"] in {"random_forest", "xgboost", "lightgbm"}
        assert "BENIGN" in body["label_classes"]

        features = dict.fromkeys(body["feature_names"], 0.0)
        prediction = client.post("/api/v1/predict", json={"features": features})
        assert prediction.status_code == 200
        assert prediction.json()["predicted_class"] in body["label_classes"]
