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
