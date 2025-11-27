from aetheris_oracle.api import forecast_endpoint
from aetheris_oracle.pipeline.forecast import ForecastEngine
from aetheris_oracle.monitoring import LoggingMetricsSink


def test_api_forecast_basic():
    payload = {"asset_id": "BTC-USD", "horizon": 2, "num_paths": 10, "thresholds": [10000, 60000]}
    result = forecast_endpoint(payload, engine=ForecastEngine(seed=1))

    assert "quantile_paths" in result
    assert "threshold_probabilities" in result
    assert result["quantile_paths"]
    assert "scenario_label" in result
    assert "coverage" in result
    assert "hits" in result["coverage"]


def test_metrics_hook_is_called():
    calls = {}

    class RecordingMetrics:
        def emit_forecast_metrics(self, **kwargs):
            calls["latency_ms"] = kwargs["latency_ms"]
            calls["asset_id"] = kwargs["asset_id"]

        def emit_error(self, name, detail):
            calls["error"] = name

    engine = ForecastEngine(seed=1, metrics=RecordingMetrics())
    payload = {"asset_id": "BTC-USD", "horizon": 1, "num_paths": 5}
    forecast_endpoint(payload, engine=engine)

    assert "latency_ms" in calls
    assert calls["asset_id"] == "BTC-USD"


def test_logging_metrics_sink_no_error():
    sink = LoggingMetricsSink()
    sink.emit_forecast_metrics(
        latency_ms=10.0, asset_id="BTC", horizon=1, num_paths=10, regime_bucket="normal"
    )
    sink.emit_error("test_error", {"detail": "msg"})
