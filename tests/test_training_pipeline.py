from aetheris_oracle.modules.trend import TrendConfig, TrendEnsemble
from aetheris_oracle.modules.vol_path import VolPathEngine
from aetheris_oracle.modules.jump import JumpModel
from aetheris_oracle.pipeline.offline_evaluation import run_walk_forward
from aetheris_oracle.pipeline.train import TrainConfig, run_training
from aetheris_oracle.config import ForecastConfig


def test_run_training_produces_artifacts(tmp_path):
    artifact_root = tmp_path / "artifacts"
    cfg = TrainConfig(horizon_days=3, num_samples=4, trailing_window_days=20, artifact_root=str(artifact_root))
    artifacts = run_training(cfg)

    assert (artifact_root / "trend_state.json").exists()
    assert artifacts.trend.meta_weights

    trend = TrendEnsemble(config=TrendConfig(artifact_path=str(artifact_root / "trend_state.json")))
    path = trend.predict_trend([0.1, 0.2, 0.3, 0.4], horizon=2, regime_strength=1.0).path
    assert len(path) == 2


def test_vol_jump_and_eval_paths_work():
    vol_engine = VolPathEngine()
    vol = vol_engine.forecast(
        {"iv_7d_atm": 0.4, "iv_14d_atm": 0.42, "iv_30d_atm": 0.45},
        horizon=3,
        regime_strength=1.1,
        mm_indices=(0.2, -0.1, 0.05),
    )
    assert len(vol.path) == 3

    jump_model = JumpModel()
    jump_path = jump_model.sample_path(
        horizon=3,
        vol_path=vol.path,
        narrative_score=0.2,
        gamma_squeeze=0.1,
        regime_strength=1.0,
        basis_pressure=0.0,
    )
    assert len(jump_path.path) == 3

    eval_result = run_walk_forward([ForecastConfig(horizon_days=2, num_paths=5)])
    assert 0.0 <= eval_result.coverage.get("rate", 0.0) <= 1.0
