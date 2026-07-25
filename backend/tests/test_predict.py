import time


def _sample_features(client) -> dict[str, float]:
    model_info = client.get("/api/v1/model").json()
    # Zeros are fine here -- these tests exercise request/response plumbing, not accuracy.
    return dict.fromkeys(model_info["feature_names"], 0.0)


def test_predict_returns_valid_prediction(ready_client):
    features = _sample_features(ready_client)

    response = ready_client.post("/api/v1/predict", json={"features": features})

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"]
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["class_probabilities"] is not None
    assert abs(sum(body["class_probabilities"].values()) - 1.0) < 1e-6
    assert body["model_version"] == "v1"


def test_predict_rejects_missing_features(ready_client):
    response = ready_client.post("/api/v1/predict", json={"features": {}})

    assert response.status_code == 422
    assert "missing_features" in response.json()["detail"]


def test_predict_rejects_unexpected_features(ready_client):
    features = _sample_features(ready_client)
    features["totally_made_up_feature"] = 1.0

    response = ready_client.post("/api/v1/predict", json={"features": features})

    assert response.status_code == 422
    assert "unexpected_features" in response.json()["detail"]


def test_predict_rejects_non_numeric_feature_value(ready_client):
    features = _sample_features(ready_client)
    features[next(iter(features))] = "not-a-number"

    response = ready_client.post("/api/v1/predict", json={"features": features})

    assert response.status_code == 422


def test_predict_returns_503_when_model_not_loaded(not_ready_client):
    response = not_ready_client.post("/api/v1/predict", json={"features": {}})

    assert response.status_code == 503


def test_predict_batch_returns_predictions_for_every_instance(ready_client):
    features = _sample_features(ready_client)

    response = ready_client.post(
        "/api/v1/predict/batch", json={"instances": [features, features, features]}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 3
    assert len(body["predictions"]) == 3


def test_predict_batch_rejects_empty_instances(ready_client):
    response = ready_client.post("/api/v1/predict/batch", json={"instances": []})

    assert response.status_code == 422


def test_predict_batch_returns_503_when_model_not_loaded(not_ready_client):
    response = not_ready_client.post("/api/v1/predict/batch", json={"instances": [{}]})

    assert response.status_code == 503


def test_predict_latency_is_reasonable(ready_client):
    features = _sample_features(ready_client)

    start = time.perf_counter()
    response = ready_client.post("/api/v1/predict", json={"features": features})
    duration = time.perf_counter() - start

    assert response.status_code == 200
    assert duration < 2.0  # generous bound -- real latency is single-digit ms
