def test_model_info_returns_metadata_when_ready(ready_client):
    response = ready_client.get("/api/v1/model")

    assert response.status_code == 200
    body = response.json()
    assert body["model_type"] in {"random_forest", "xgboost", "lightgbm"}
    assert body["model_version"] == "v1"
    assert isinstance(body["feature_names"], list) and body["feature_names"]
    assert isinstance(body["label_classes"], list) and body["label_classes"]
    assert "accuracy" in body["metrics"]


def test_model_info_returns_503_when_not_ready(not_ready_client):
    response = not_ready_client.get("/api/v1/model")

    assert response.status_code == 503


def test_training_summary_returns_full_comparison_when_ready(ready_client):
    response = ready_client.get("/api/v1/model/training-summary")

    assert response.status_code == 200
    body = response.json()
    assert set(body["trained_models"]) >= {"random_forest", "xgboost"}
    assert body["best_model_type"] in body["val_metrics"]
    assert body["best_model_type"] in body["feature_importances"]
    assert set(body["class_distribution"]) == {"train", "val", "test"}
    for model_type in body["trained_models"]:
        val = body["val_metrics"][model_type]
        assert "confusion_matrix" in val
        assert "classification_report" in val


def test_training_summary_returns_503_when_not_ready(not_ready_client):
    response = not_ready_client.get("/api/v1/model/training-summary")

    assert response.status_code == 503


def test_training_summary_returns_404_when_file_missing(ready_client, trained_models_root):
    (trained_models_root / "v1" / "training_summary.json").unlink()

    response = ready_client.get("/api/v1/model/training-summary")

    assert response.status_code == 404
