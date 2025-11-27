from fastapi.testclient import TestClient

from aetheris_oracle.service import create_app


def test_forecast_requires_api_key_when_set():
    app = create_app(api_key="secret")
    client = TestClient(app)
    resp = client.post("/forecast", json={"asset_id": "BTC-USD", "horizon": 1, "num_paths": 5})
    assert resp.status_code == 401

    resp_ok = client.post(
        "/forecast",
        headers={"x-api-key": "secret"},
        json={"asset_id": "BTC-USD", "horizon": 1, "num_paths": 5},
    )
    assert resp_ok.status_code == 200
    assert "scenario_label" in resp_ok.json()
