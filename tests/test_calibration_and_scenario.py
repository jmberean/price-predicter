from datetime import datetime, timedelta

from aetheris_oracle.config import ForecastConfig, ScenarioOverrides
from aetheris_oracle.data.schemas import MarketFeatureFrame, RegimeVector
from aetheris_oracle.pipeline.calibration import BucketStats, CalibrationEngine, CalibrationState
from aetheris_oracle.pipeline.forecast import ForecastEngine
from aetheris_oracle.pipeline.scenario import apply_scenario
from aetheris_oracle.modules.residual import ResidualConfig, ResidualGenerator


def test_calibration_applies_bucket_scale():
    state = CalibrationState(
        global_width_scale=1.0,
        bucket_scales={("volatile", "short"): BucketStats(width_scale=1.2, count=5)},
    )
    engine = CalibrationEngine(state=state)
    quantiles = {0.1: 1.0, 0.5: 2.0, 0.9: 3.0}

    calibrated = engine.calibrate_quantiles(
        quantiles, regime=RegimeVector(values=[0.2]), horizon=1
    )

    assert calibrated[0.1] < quantiles[0.1]
    assert calibrated[0.9] > quantiles[0.9]
    assert calibrated[0.5] == quantiles[0.5]


def test_apply_scenario_clamps_and_whitelists():
    now = datetime.utcnow()
    frame = MarketFeatureFrame(
        timestamps=[now - timedelta(days=1), now],
        closes=[100.0, 105.0],
        volumes=[1000.0, 1100.0],
        iv_points={"iv_7d_atm": 0.5, "iv_30d_atm": 0.6},
        funding_rate=0.01,
        basis=0.02,
        order_imbalance=0.1,
        narrative_scores={"RegulationRisk": 0.4},
        skew=0.0,
    )
    scenario = ScenarioOverrides(
        iv_multiplier=10.0,
        funding_shift=0.2,
        basis_shift=-0.2,
        narrative_overrides={"RegulationRisk": 2.0, "Unknown": 0.5},
    )

    adjusted = apply_scenario(frame, scenario)

    assert adjusted.iv_points["iv_7d_atm"] == 0.5 * 3.0
    assert adjusted.funding_rate == 0.01 + 0.05
    assert adjusted.basis == 0.02 - 0.05
    assert "Unknown" not in adjusted.narrative_scores
    assert adjusted.narrative_scores["RegulationRisk"] == 1.0


def test_calibration_state_roundtrip(tmp_path):
    state = CalibrationState(
        global_width_scale=1.1,
        bucket_scales={("normal", "mid"): BucketStats(width_scale=1.05, count=3)},
        coverage={( "normal", "mid"): {"hits": 2, "total": 3}},
    )
    engine = CalibrationEngine(state=state)
    target = tmp_path / "calibration.json"
    engine.save(target)

    reloaded = CalibrationEngine.load(target)
    bucket = reloaded.state.bucket_scales[("normal", "mid")]
    assert reloaded.state.global_width_scale == 1.1
    assert bucket.width_scale == 1.05
    assert bucket.count == 3
    assert reloaded.state.coverage[("normal", "mid")]["hits"] == 2


def test_forecast_batch_uses_custom_connector():
    class StubConnector:
        def fetch_window(self, asset_id, as_of, window):
            now = as_of
            timestamps = [now - timedelta(days=1), now]
            closes = [100.0, 101.0]
            volumes = [1000.0, 1005.0]
            iv_points = {"iv_7d_atm": 0.4, "iv_30d_atm": 0.45}
            return MarketFeatureFrame(
                timestamps=timestamps,
                closes=closes,
                volumes=volumes,
                iv_points=iv_points,
                funding_rate=0.0,
                basis=0.0,
                order_imbalance=0.0,
                narrative_scores={"RegulationRisk": 0.1},
                skew=0.0,
            )

    engine = ForecastEngine(seed=1, connector=StubConnector())
    configs = [
        ForecastConfig(asset_id="ASSET1", horizon_days=2, num_paths=10),
        ForecastConfig(asset_id="ASSET2", horizon_days=2, num_paths=10),
    ]
    results = engine.forecast_batch(configs)

    assert len(results) == 2
    for cfg in configs:
        key = f"{cfg.asset_id}:{cfg.as_of.isoformat()}:{cfg.horizon_days}"
        assert key in results
        res = results[key]
        assert set(res.quantile_paths.keys()) == {1, 2}
        assert res.regime is not None


def test_update_calibration_with_realized_increases_width():
    engine = ForecastEngine(seed=2)
    cfg = ForecastConfig(horizon_days=2, num_paths=20, seed=2)
    res = engine.forecast(cfg)
    before = engine.calibration.state.global_width_scale

    # Realized far outside predicted bounds should widen calibration
    engine.update_calibration_with_realized(res, realized_price=10_000_000.0, horizon=2)

    assert engine.calibration.state.global_width_scale == before
    bucket = engine.calibration.state.bucket_scales[
        (engine.calibration._regime_bucket(res.regime), engine.calibration._horizon_bucket(2))
    ]
    assert bucket.width_scale > 1.0


def test_residual_generator_zero_centers_paths():
    cfg = ResidualConfig(ar_coefficient=0.8, base_vol=0.1, tail_weight=0.0)
    gen = ResidualGenerator(config=cfg, seed=123)
    paths = gen.sample_paths(
        horizon=6,
        num_paths=1,
        vol_path=[0.05] * 6,
        regime_strength=1.0,
    ).paths
    path = paths[0]
    assert len(path) == 6
    # Paths are zero-centered to avoid double counting trend
    assert abs(sum(path)) < 0.1
