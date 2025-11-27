"""
Tests for SOTA neural components.

Tests all new state-of-the-art components:
- Neural Conformal Control
- FM-GP Residual Generator
- Neural Jump SDE
- Differentiable Greeks MM Engine
- Neural Rough Volatility
- MambaTS Trend
- Integrated Gradients Explainability
- Importance Sampling
"""

import numpy as np
import pytest
import torch
from pathlib import Path
import tempfile

from aetheris_oracle.config import ForecastConfig
from aetheris_oracle.data.connectors import SyntheticDataConnector


def test_importance_sampling():
    """Test importance sampling for quantile computation."""
    from aetheris_oracle.utils.importance_sampling import (
        compute_quantiles_with_importance_sampling,
        adaptive_path_count,
    )

    # Generate sample paths
    n_paths = 1000
    horizon = 14
    paths = [[np.random.randn() * 0.02 for _ in range(horizon)] for _ in range(n_paths)]

    # Compute quantiles
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    results = compute_quantiles_with_importance_sampling(paths, quantiles)

    # Should return one result per quantile
    assert len(results) == len(quantiles)

    # Each quantile should have horizon values
    for q_vals in results:
        assert len(q_vals) == horizon

    # Test adaptive path count
    preliminary_paths = paths[:100]
    n_recommended = adaptive_path_count(preliminary_paths, target_tail_se=0.01)

    assert n_recommended >= 1000
    assert n_recommended <= 50000


def test_integrated_gradients():
    """Test Integrated Gradients explainability."""
    from aetheris_oracle.pipeline.integrated_gradients import IntegratedGradientsExplainer

    explainer = IntegratedGradientsExplainer()

    # Sample features
    features = {
        "regime_volatility": 0.5,
        "iv_level": 0.4,
        "gamma_squeeze": 0.2,
        "inventory_unwind": -0.1,
        "basis_pressure": 0.05,
    }

    # Simple model function
    def model_fn(x):
        return x[0] * 0.5 + x[1] * 0.3 + x[2] * 0.2

    result = explainer.explain_forecast(features, model_fn=model_fn)

    # Check result structure
    assert hasattr(result, 'feature_attributions')
    assert hasattr(result, 'top_drivers')
    assert hasattr(result, 'concept_explanations')

    # Should have attribution for each feature
    assert len(result.feature_attributions) == len(features)

    # Top drivers should be sorted by absolute value
    assert len(result.top_drivers) <= len(features)

    # Should have concept explanations
    assert len(result.concept_explanations) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_neural_conformal_control_cpu():
    """Test NCC calibration engine on CPU."""
    _test_ncc("cpu")


def _test_ncc(device):
    """Test NCC on specified device."""
    from aetheris_oracle.pipeline.neural_conformal_control import (
        NCCCalibrationEngine,
        NCCConfig,
    )

    config = NCCConfig(device=device)
    engine = NCCCalibrationEngine(config=config, device=device)

    # Generate synthetic training data
    n_samples = 50
    base_quantiles = [np.array([40000, 42000, 45000, 48000, 51000, 54000, 56000]) for _ in range(n_samples)]
    actuals = [np.random.uniform(40000, 56000) for _ in range(n_samples)]
    features = [np.random.randn(10).tolist() for _ in range(n_samples)]
    horizon_indices = [0] * n_samples

    # Train
    metrics = engine.train_online(
        base_quantiles_batch=base_quantiles,
        actuals_batch=actuals,
        features_batch=features,
        horizon_indices_batch=horizon_indices,
        epochs=3,
        batch_size=16,
    )

    # Check metrics
    assert "loss" in metrics
    assert len(metrics["loss"]) == 3  # 3 epochs

    # Test save/load
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "ncc.pt"
        engine.save(path)
        loaded = NCCCalibrationEngine.load(path, device=device)

        assert loaded.config.device == device


def test_fmgp_residuals():
    """Test FM-GP Residual Generator."""
    from aetheris_oracle.modules.fm_gp_residual import FMGPResidualEngine, FMGPConfig

    config = FMGPConfig(device="cpu")
    engine = FMGPResidualEngine(config=config, device="cpu")

    # Sample paths
    conditioning = [0.5, 0.4, 0.2, -0.1, 0.05]
    vol_path = [0.3, 0.32, 0.35, 0.33, 0.31, 0.30, 0.29]

    paths = engine.sample_paths(
        horizon=7,
        num_paths=100,
        vol_path=vol_path,
        conditioning=conditioning,
    )

    # Check shape
    assert len(paths) == 100
    assert len(paths[0]) == 7

    # Check zero mean constraint (approximately)
    mean_residuals = np.mean([np.mean(path) for path in paths])
    assert abs(mean_residuals) < 0.1  # Should be close to zero

    # Test save/load
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "fmgp.pt"
        engine.save(path)
        loaded = FMGPResidualEngine.load(path, device="cpu")

        # Verify loaded model works
        paths_loaded = loaded.sample_paths(7, 10, vol_path, conditioning)
        assert len(paths_loaded) == 10


def test_neural_jump_sde():
    """Test Neural Jump SDE."""
    from aetheris_oracle.modules.neural_jump_sde import NeuralJumpSDEEngine, NeuralJumpSDEConfig

    config = NeuralJumpSDEConfig(device="cpu")
    engine = NeuralJumpSDEEngine(config=config, device="cpu")

    # Sample paths
    x0 = torch.zeros(50, device="cpu")
    conditioning = [0.5, 0.4, 0.2, -0.1, 0.05, 0.3]
    vol_path = [0.3] * 14

    paths = engine.sample_sde_paths(
        x0=x0,
        conditioning=conditioning,
        horizon=14,
        vol_path=vol_path,
    )

    # Check shape
    assert paths.shape == (50, 14)

    # Test save/load
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "neural_jump.pt"
        engine.save(path)
        loaded = NeuralJumpSDEEngine.load(path, device="cpu")

        # Verify loaded model works
        paths_loaded = loaded.sample_sde_paths(x0, conditioning, 14, vol_path)
        assert paths_loaded.shape == (50, 14)


def test_differentiable_greeks():
    """Test Differentiable Greeks MM Engine."""
    from aetheris_oracle.modules.differentiable_greeks import (
        DifferentiableMMEngineWrapper,
        DifferentiableGreeksConfig,
    )

    config = DifferentiableGreeksConfig()
    wrapper = DifferentiableMMEngineWrapper(config=config, device="cpu")

    # Compute indices
    mm_state, attn_weights = wrapper.compute_indices(
        spot=45000.0,
        iv_term_structure={"iv_7d_atm": 0.5, "iv_30d_atm": 0.55},
        funding_rate=0.01,
        basis=50.0,
        order_imbalance=0.1,
        skew=-0.05,
    )

    # Check outputs
    assert mm_state.shape[0] == config.mm_state_dim
    assert attn_weights is not None

    # Test save/load
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "diff_greeks.pt"
        wrapper.save(path)
        loaded = DifferentiableMMEngineWrapper.load(path, device="cpu")

        # Verify loaded model works
        mm_state_loaded, _ = loaded.compute_indices(
            45000.0,
            {"iv_7d_atm": 0.5},
            0.01,
            50.0,
            0.1,
            -0.05,
        )
        assert mm_state_loaded.shape == mm_state.shape


def test_neural_rough_vol():
    """Test Neural Rough Volatility."""
    from aetheris_oracle.modules.neural_rough_vol import NeuralRoughVolWrapper

    wrapper = NeuralRoughVolWrapper(device="cpu")

    # Forecast volatility
    iv_points = {"iv_7d_atm": 0.5, "iv_14d_atm": 0.52, "iv_30d_atm": 0.55}
    vol_forecast = wrapper.forecast(
        iv_points=iv_points,
        horizon=14,
        regime_strength=1.2,
        mm_indices=(0.2, -0.1, 0.05),
    )

    # Check output
    assert len(vol_forecast) == 14
    assert all(v > 0 for v in vol_forecast)  # Volatility must be positive

    # Test save/load
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "neural_vol.pt"
        wrapper.save(path)
        loaded = NeuralRoughVolWrapper.load(path, device="cpu")

        # Verify loaded model works
        vol_loaded = loaded.forecast(iv_points, 14, 1.2, (0.2, -0.1, 0.05))
        assert len(vol_loaded) == 14


def test_mamba_trend():
    """Test MambaTS Trend."""
    from aetheris_oracle.modules.mamba_trend import MambaTrendWrapper

    wrapper = MambaTrendWrapper(device="cpu")

    # Predict trend
    normalized_closes = [1.0 + 0.01 * i for i in range(60)]
    trend = wrapper.predict_trend(
        normalized_closes=normalized_closes,
        horizon=14,
        regime_strength=1.0,
    )

    # Check output
    assert len(trend) == 14

    # Test save/load
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "mamba.pt"
        wrapper.save(path)
        loaded = MambaTrendWrapper.load(path, device="cpu")

        # Verify loaded model works
        trend_loaded = loaded.predict_trend(normalized_closes, 14, 1.0)
        assert len(trend_loaded) == 14


def test_forecast_engine_with_sota():
    """Test ForecastEngine with SOTA components enabled."""
    from aetheris_oracle.pipeline.forecast import ForecastEngine

    # Test with all SOTA components enabled
    engine = ForecastEngine(
        seed=42,
        use_importance_sampling=True,
        use_integrated_gradients=True,
        use_ncc_calibration=False,  # Skip for speed
        use_fm_gp_residuals=False,
        use_neural_jumps=False,
        use_diff_greeks=False,
        use_neural_rough_vol=False,
        use_mamba_trend=False,
        device="cpu",
    )

    config = ForecastConfig(
        asset_id="BTC-USD",
        horizon_days=7,
        num_paths=100,  # Reduced for speed
    )

    result = engine.forecast(config)

    # Check result structure
    assert result.quantile_paths
    assert result.threshold_probabilities
    assert result.metadata

    # Check SOTA metadata
    sota_enabled = result.metadata.get("sota_enabled", {})
    assert sota_enabled["importance_sampling"] == True
    assert sota_enabled["integrated_gradients"] == True


def test_forecast_engine_with_all_sota():
    """Test ForecastEngine with all SOTA components (if available)."""
    from aetheris_oracle.pipeline.forecast import ForecastEngine

    # This may fail if dependencies not installed, so we wrap in try/except
    try:
        engine = ForecastEngine(
            seed=42,
            use_importance_sampling=True,
            use_integrated_gradients=True,
            use_ncc_calibration=True,
            use_fm_gp_residuals=True,
            use_neural_jumps=True,
            use_diff_greeks=True,
            use_neural_rough_vol=True,
            use_mamba_trend=True,
            device="cpu",
        )

        config = ForecastConfig(
            asset_id="BTC-USD",
            horizon_days=7,
            num_paths=100,
        )

        result = engine.forecast(config)

        # Check all SOTA flags are enabled
        sota_enabled = result.metadata.get("sota_enabled", {})
        assert all(sota_enabled.values())

    except ImportError as e:
        pytest.skip(f"SOTA dependencies not available: {e}")


def test_training_script_imports():
    """Test that training script can be imported."""
    try:
        from aetheris_oracle.pipeline.train_sota import (
            train_ncc_calibration,
            train_fmgp_residuals,
            train_neural_jump_sde,
            train_differentiable_greeks,
            train_neural_rough_vol,
            train_mamba_trend,
            train_all_components,
        )

        # Just verify functions exist
        assert callable(train_ncc_calibration)
        assert callable(train_fmgp_residuals)
        assert callable(train_neural_jump_sde)
        assert callable(train_differentiable_greeks)
        assert callable(train_neural_rough_vol)
        assert callable(train_mamba_trend)
        assert callable(train_all_components)

    except ImportError as e:
        pytest.skip(f"Training script dependencies not available: {e}")


def test_advanced_metrics():
    """Test advanced metrics functions."""
    try:
        from aetheris_oracle.monitoring.advanced_metrics import (
            compute_crps,
            compute_qice,
            compute_energy_score,
        )

        # Generate fake paths
        n_paths = 100
        horizon = 14
        generated_paths = np.random.randn(n_paths, horizon)
        actual_path = np.random.randn(horizon)

        # Test CRPS
        crps = compute_crps(generated_paths, actual_path)
        assert crps >= 0

        # Test QICE
        quantile_levels = [0.05, 0.25, 0.5, 0.75, 0.95]
        qice = compute_qice(generated_paths, actual_path, quantile_levels)
        assert isinstance(qice, float)

        # Test Energy Score
        energy = compute_energy_score(generated_paths, actual_path)
        assert energy >= 0

    except ImportError as e:
        pytest.skip(f"Advanced metrics dependencies not available: {e}")


def test_bellman_conformal():
    """Test Bellman Conformal Inference."""
    try:
        from aetheris_oracle.pipeline.bellman_conformal import BellmanConformalOptimizer

        optimizer = BellmanConformalOptimizer(horizons=[1, 3, 7, 14])

        # Generate fake data
        n_samples = 50
        base_quantiles = {
            h: {0.05: 40000, 0.5: 48000, 0.95: 56000} for h in [1, 3, 7, 14]
        }

        historical_actuals = {h: [np.random.uniform(40000, 56000) for _ in range(n_samples)] for h in [1, 3, 7, 14]}
        historical_predictions = {h: [base_quantiles[h] for _ in range(n_samples)] for h in [1, 3, 7, 14]}

        # Optimize
        optimal_thresholds = optimizer.optimize_thresholds(
            base_quantiles, historical_actuals, historical_predictions
        )

        # Check structure
        assert len(optimal_thresholds) == 4  # 4 horizons

    except ImportError as e:
        pytest.skip(f"Bellman conformal dependencies not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
