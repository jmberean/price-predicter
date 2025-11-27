from fastapi.testclient import TestClient

from aetheris_oracle.service import create_app


def test_service_forecast_endpoint():
    app = create_app()
    client = TestClient(app)

    payload = {"asset_id": "BTC-USD", "horizon": 1, "num_paths": 5}
    resp = client.post("/forecast", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "quantile_paths" in data
    assert "threshold_probabilities" in data


def test_service_health():
    app = create_app()
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
