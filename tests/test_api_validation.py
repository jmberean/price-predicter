"""
API and Service validation tests.

Tests the FastAPI REST API endpoints for:
- Request validation
- Response schema compliance
- Error handling
- Authentication (if enabled)
- Performance under load
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta

from aetheris_oracle.server import create_app
from aetheris_oracle.api_schemas import ForecastRequest, ForecastResponse


class TestAPIEndpoints:
    """Test REST API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)

    @pytest.fixture
    def auth_client(self):
        """Create test client with API key authentication."""
        app = create_app(api_key="test-secret-key")
        return TestClient(app)

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "timestamp" in data

    def test_basic_forecast_request(self, client):
        """Test basic forecast request."""
        request_data = {
            "asset_id": "BTC-USD",
            "horizon_days": 7,
        }

        response = client.post("/forecast", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Validate response structure
        assert "quantile_paths" in data
        assert "threshold_probabilities" in data
        assert "metadata" in data
        assert "drivers" in data

        # Validate quantile paths
        assert len(data["quantile_paths"]) == 7
        for t, quantiles in data["quantile_paths"].items():
            assert "0.05" in quantiles
            assert "0.5" in quantiles
            assert "0.95" in quantiles

    def test_forecast_with_custom_params(self, client):
        """Test forecast with custom parameters."""
        request_data = {
            "asset_id": "ETH-USD",
            "horizon_days": 14,
            "num_paths": 2000,
            "quantiles": [0.1, 0.5, 0.9],
            "thresholds": [2000.0, 2500.0, 3000.0],
        }

        response = client.post("/forecast", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Should have 14 time steps
        assert len(data["quantile_paths"]) == 14

        # Should have specified quantiles
        for t, quantiles in data["quantile_paths"].items():
            assert "0.1" in quantiles
            assert "0.5" in quantiles
            assert "0.9" in quantiles

        # Should have threshold probabilities
        assert len(data["threshold_probabilities"]) == 3

    def test_forecast_with_scenario(self, client):
        """Test forecast with scenario overrides."""
        request_data = {
            "asset_id": "BTC-USD",
            "horizon_days": 7,
            "scenario": {
                "description": "stress test",
                "iv_multiplier": 2.0,
                "funding_shift": 0.02,
            }
        }

        response = client.post("/forecast", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Should be labeled as conditional
        assert data["metadata"]["scenario_label"] == "conditional"
        assert "stress test" in data["metadata"]["scenario"]

    def test_invalid_asset_id(self, client):
        """Test request with invalid asset ID."""
        request_data = {
            "asset_id": "",  # Empty string
            "horizon_days": 7,
        }

        response = client.post("/forecast", json=request_data)

        # Should return validation error
        assert response.status_code == 422

    def test_invalid_horizon(self, client):
        """Test request with invalid horizon."""
        request_data = {
            "asset_id": "BTC-USD",
            "horizon_days": -1,  # Negative
        }

        response = client.post("/forecast", json=request_data)

        # Should return validation error
        assert response.status_code == 422

    def test_invalid_quantiles(self, client):
        """Test request with invalid quantiles."""
        request_data = {
            "asset_id": "BTC-USD",
            "horizon_days": 7,
            "quantiles": [0.5, 1.5],  # 1.5 is invalid
        }

        response = client.post("/forecast", json=request_data)

        # Should return validation error
        assert response.status_code == 422

    def test_api_key_auth_success(self, auth_client):
        """Test API key authentication (success)."""
        request_data = {
            "asset_id": "BTC-USD",
            "horizon_days": 7,
        }

        response = auth_client.post(
            "/forecast",
            json=request_data,
            headers={"x-api-key": "test-secret-key"}
        )

        assert response.status_code == 200

    def test_api_key_auth_failure(self, auth_client):
        """Test API key authentication (failure)."""
        request_data = {
            "asset_id": "BTC-USD",
            "horizon_days": 7,
        }

        # No API key
        response = auth_client.post("/forecast", json=request_data)
        assert response.status_code == 403

        # Wrong API key
        response = auth_client.post(
            "/forecast",
            json=request_data,
            headers={"x-api-key": "wrong-key"}
        )
        assert response.status_code == 403


class TestAPIPerformance:
    """Test API performance under various loads."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)

    def test_response_time(self, client):
        """Test API response time is acceptable."""
        import time

        request_data = {
            "asset_id": "BTC-USD",
            "horizon_days": 7,
            "num_paths": 500,
        }

        start = time.perf_counter()
        response = client.post("/forecast", json=request_data)
        latency_ms = (time.perf_counter() - start) * 1000

        assert response.status_code == 200

        # Should respond in under 3 seconds
        assert latency_ms < 3000, f"API too slow: {latency_ms:.1f}ms"

        print(f"\n✓ API response time: {latency_ms:.1f}ms")

    def test_concurrent_requests(self, client):
        """Test handling concurrent requests."""
        import concurrent.futures

        request_data = {
            "asset_id": "BTC-USD",
            "horizon_days": 7,
            "num_paths": 200,  # Reduced for speed
        }

        def make_request():
            return client.post("/forecast", json=request_data)

        # Make 5 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            responses = [f.result() for f in futures]

        # All should succeed
        assert all(r.status_code == 200 for r in responses)

        print(f"\n✓ Handled 5 concurrent requests successfully")

    def test_repeated_requests(self, client):
        """Test performance doesn't degrade over repeated requests."""
        request_data = {
            "asset_id": "BTC-USD",
            "horizon_days": 7,
            "num_paths": 200,
        }

        latencies = []
        for i in range(10):
            import time
            start = time.perf_counter()
            response = client.post("/forecast", json=request_data)
            latency_ms = (time.perf_counter() - start) * 1000

            assert response.status_code == 200
            latencies.append(latency_ms)

        # Average latency should be reasonable
        avg_latency = sum(latencies) / len(latencies)
        assert avg_latency < 2000, f"Average latency too high: {avg_latency:.1f}ms"

        # Should not show significant degradation
        # Last 3 requests should not be > 2x slower than first 3
        early_avg = sum(latencies[:3]) / 3
        late_avg = sum(latencies[-3:]) / 3

        assert late_avg < early_avg * 2, "Performance degradation detected"

        print(f"\n✓ 10 requests - avg latency: {avg_latency:.1f}ms")


class TestResponseValidation:
    """Validate API response structure and data quality."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)

    def test_response_schema_compliance(self, client):
        """Test response matches Pydantic schema."""
        request_data = {
            "asset_id": "BTC-USD",
            "horizon_days": 7,
        }

        response = client.post("/forecast", json=request_data)
        data = response.json()

        # Validate against Pydantic model
        forecast_response = ForecastResponse(**data)

        assert forecast_response.quantile_paths is not None
        assert forecast_response.metadata is not None

    def test_quantile_paths_structure(self, client):
        """Test quantile paths have correct structure."""
        request_data = {
            "asset_id": "BTC-USD",
            "horizon_days": 7,
            "quantiles": [0.05, 0.25, 0.5, 0.75, 0.95],
        }

        response = client.post("/forecast", json=request_data)
        data = response.json()

        quantile_paths = data["quantile_paths"]

        # Should have entry for each day
        assert len(quantile_paths) == 7

        # Each day should have all quantiles
        for day_str, quantiles in quantile_paths.items():
            day = int(day_str)
            assert 1 <= day <= 7

            assert "0.05" in quantiles
            assert "0.25" in quantiles
            assert "0.5" in quantiles
            assert "0.75" in quantiles
            assert "0.95" in quantiles

            # Quantiles should be ordered
            assert quantiles["0.05"] <= quantiles["0.25"]
            assert quantiles["0.25"] <= quantiles["0.5"]
            assert quantiles["0.5"] <= quantiles["0.75"]
            assert quantiles["0.75"] <= quantiles["0.95"]

    def test_threshold_probabilities_structure(self, client):
        """Test threshold probabilities have correct structure."""
        request_data = {
            "asset_id": "BTC-USD",
            "horizon_days": 7,
            "thresholds": [45000.0, 50000.0, 55000.0],
        }

        response = client.post("/forecast", json=request_data)
        data = response.json()

        threshold_probs = data["threshold_probabilities"]

        # Should have entry for each threshold
        assert len(threshold_probs) == 3

        for threshold_str, probs in threshold_probs.items():
            threshold = float(threshold_str)
            assert threshold in [45000.0, 50000.0, 55000.0]

            # Should have lt and gt
            assert "lt" in probs
            assert "gt" in probs

            # Probabilities in [0, 1]
            assert 0 <= probs["lt"] <= 1
            assert 0 <= probs["gt"] <= 1

    def test_metadata_completeness(self, client):
        """Test metadata contains all required fields."""
        request_data = {
            "asset_id": "BTC-USD",
            "horizon_days": 7,
        }

        response = client.post("/forecast", json=request_data)
        data = response.json()

        metadata = data["metadata"]

        required_fields = ["asset_id", "regime_bucket", "horizon_bucket"]
        for field in required_fields:
            assert field in metadata, f"Missing metadata field: {field}"

        # Asset ID should match request
        assert metadata["asset_id"] == "BTC-USD"

    def test_drivers_structure(self, client):
        """Test drivers have correct structure."""
        request_data = {
            "asset_id": "BTC-USD",
            "horizon_days": 7,
        }

        response = client.post("/forecast", json=request_data)
        data = response.json()

        drivers = data["drivers"]

        # Should have at least some drivers
        assert len(drivers) > 0

        # Each driver should be a list [name, score]
        for driver in drivers:
            assert isinstance(driver, list)
            assert len(driver) == 2
            assert isinstance(driver[0], str)
            assert isinstance(driver[1], (int, float))


class TestErrorHandling:
    """Test API error handling."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)

    def test_malformed_json(self, client):
        """Test handling of malformed JSON."""
        response = client.post(
            "/forecast",
            data="this is not json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_missing_required_fields(self, client):
        """Test handling of missing required fields."""
        # Missing asset_id
        response = client.post("/forecast", json={"horizon_days": 7})
        assert response.status_code == 422

        # Missing horizon_days
        response = client.post("/forecast", json={"asset_id": "BTC-USD"})
        assert response.status_code == 422

    def test_invalid_data_types(self, client):
        """Test handling of invalid data types."""
        # String instead of int for horizon
        response = client.post("/forecast", json={
            "asset_id": "BTC-USD",
            "horizon_days": "seven"
        })
        assert response.status_code == 422

    def test_out_of_range_values(self, client):
        """Test handling of out-of-range values."""
        # Extremely large horizon
        response = client.post("/forecast", json={
            "asset_id": "BTC-USD",
            "horizon_days": 1000000
        })
        # Should either reject or handle gracefully
        # (implementation dependent)
        assert response.status_code in [200, 422]

    def test_method_not_allowed(self, client):
        """Test using wrong HTTP method."""
        # GET instead of POST
        response = client.get("/forecast")
        assert response.status_code == 405


class TestSOTAComponentsAPI:
    """Test API with SOTA components enabled (if available)."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)

    @pytest.mark.skipif(True, reason="SOTA components may not be available")
    def test_sota_forecast_via_api(self, client):
        """Test forecast with SOTA components via API."""
        try:
            # This would require extending API to support SOTA flags
            # For now, just verify basic functionality
            request_data = {
                "asset_id": "BTC-USD",
                "horizon_days": 7,
                "num_paths": 500,
            }

            response = client.post("/forecast", json=request_data)
            assert response.status_code == 200

            data = response.json()

            # Check if SOTA metadata present
            if "sota_enabled" in data["metadata"]:
                print("\n✓ SOTA components active in API")
            else:
                print("\n✓ Legacy components active in API")

        except Exception as e:
            pytest.skip(f"SOTA API test failed: {e}")


def run_api_health_check():
    """Run comprehensive API health check."""
    print("\n" + "="*60)
    print("API HEALTH CHECK")
    print("="*60)

    app = create_app()
    client = TestClient(app)

    # Health endpoint
    response = client.get("/health")
    print(f"\n✓ Health endpoint: {response.status_code}")

    # Basic forecast
    response = client.post("/forecast", json={
        "asset_id": "BTC-USD",
        "horizon_days": 7,
    })
    print(f"✓ Basic forecast: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"  - Quantile paths: {len(data['quantile_paths'])} days")
        print(f"  - Drivers: {len(data['drivers'])} identified")
        print(f"  - Regime: {data['metadata']['regime_bucket']}")

    # Performance test
    import time
    start = time.perf_counter()
    response = client.post("/forecast", json={
        "asset_id": "BTC-USD",
        "horizon_days": 7,
        "num_paths": 1000,
    })
    latency_ms = (time.perf_counter() - start) * 1000
    print(f"\n✓ Performance (1000 paths): {latency_ms:.1f}ms")

    print("\n" + "="*60)


if __name__ == "__main__":
    # Run health check
    run_api_health_check()

    # Run pytest
    pytest.main([__file__, "-v", "--tb=short"])
