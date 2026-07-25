def test_ready_returns_200_when_model_loaded(ready_client):
    response = ready_client.get("/api/v1/ready")

    assert response.status_code == 200
    assert response.json() == {"ready": True, "detail": None}


def test_ready_returns_503_when_model_not_loaded(not_ready_client):
    response = not_ready_client.get("/api/v1/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["ready"] is False
    assert body["detail"]
