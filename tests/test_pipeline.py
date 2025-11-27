from aetheris_oracle.config import ForecastConfig
from aetheris_oracle.pipeline.forecast import ForecastEngine


def test_forecast_shapes():
    engine = ForecastEngine(seed=42)
    cfg = ForecastConfig(horizon_days=3, num_paths=50, thresholds=(10000.0, 60000.0), seed=42)
    result = engine.forecast(cfg)

    assert set(result.quantile_paths.keys()) == {1, 2, 3}
    for step, qs in result.quantile_paths.items():
        assert 0.5 in qs
        assert min(qs.keys()) >= 0.0
        assert max(qs.keys()) <= 1.0
    assert set(result.threshold_probabilities.keys()) == set(cfg.thresholds)
    for probs in result.threshold_probabilities.values():
        assert abs(probs["lt"] + probs["gt"] - 1.0) <= 1.0
