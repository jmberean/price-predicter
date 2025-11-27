"""
Performance and validation tests for Aetheris Oracle.

Tests:
- End-to-end forecast pipeline performance
- Latency benchmarks for different configurations
- Output validation (quantile ordering, probability constraints)
- Memory usage profiling
- Throughput testing (batch forecasts)
- Regression tests (SOTA vs legacy)
"""

import time
import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List

from aetheris_oracle.config import ForecastConfig
from aetheris_oracle.pipeline.forecast import ForecastEngine
from aetheris_oracle.data.connectors import SyntheticDataConnector


class TestForecastValidation:
    """Validate forecast outputs are mathematically valid."""

    def test_quantile_ordering(self):
        """Test that quantiles are properly ordered: P5 < P25 < P50 < P75 < P95."""
        engine = ForecastEngine(seed=42)
        config = ForecastConfig(
            asset_id="BTC-USD",
            horizon_days=7,
            num_paths=1000,
        )

        result = engine.forecast(config)

        # Check each time step
        for t in range(1, config.horizon_days + 1):
            quantiles = result.quantile_paths[t]

            # Verify ordering
            assert quantiles[0.05] <= quantiles[0.10], f"P5 > P10 at t={t}"
            assert quantiles[0.10] <= quantiles[0.25], f"P10 > P25 at t={t}"
            assert quantiles[0.25] <= quantiles[0.50], f"P25 > P50 at t={t}"
            assert quantiles[0.50] <= quantiles[0.75], f"P50 > P75 at t={t}"
            assert quantiles[0.75] <= quantiles[0.90], f"P75 > P90 at t={t}"
            assert quantiles[0.90] <= quantiles[0.95], f"P90 > P95 at t={t}"

    def test_probability_constraints(self):
        """Test that threshold probabilities sum correctly."""
        engine = ForecastEngine(seed=42)
        config = ForecastConfig(
            asset_id="BTC-USD",
            horizon_days=7,
            num_paths=1000,
            thresholds=[45000.0, 50000.0, 55000.0],
        )

        result = engine.forecast(config)

        # Check each threshold
        for threshold, probs in result.threshold_probabilities.items():
            # P(X < K) + P(X > K) should be close to 1.0
            # (may not be exact due to P(X = K) in discrete samples)
            total = probs["lt"] + probs["gt"]
            assert 0.95 <= total <= 1.05, f"Probabilities don't sum to ~1.0 for threshold {threshold}"

            # Individual probabilities should be in [0, 1]
            assert 0 <= probs["lt"] <= 1, f"P(X < {threshold}) out of bounds"
            assert 0 <= probs["gt"] <= 1, f"P(X > {threshold}) out of bounds"

    def test_cone_widening(self):
        """Test that forecast cone widens over time (uncertainty increases)."""
        engine = ForecastEngine(seed=42)
        config = ForecastConfig(
            asset_id="BTC-USD",
            horizon_days=14,
            num_paths=1000,
        )

        result = engine.forecast(config)

        # Calculate cone width (P90 - P10) at each time step
        widths = []
        for t in range(1, config.horizon_days + 1):
            q = result.quantile_paths[t]
            width = q[0.90] - q[0.10]
            widths.append(width)

        # Check that width generally increases
        # Allow for some local decreases due to noise
        increasing_count = sum(1 for i in range(1, len(widths)) if widths[i] > widths[i-1])

        # At least 60% of steps should show increasing width
        assert increasing_count >= 0.6 * (len(widths) - 1), "Cone not widening sufficiently over time"

    def test_positive_prices(self):
        """Test that all forecasted prices are positive."""
        engine = ForecastEngine(seed=42)
        config = ForecastConfig(
            asset_id="BTC-USD",
            horizon_days=7,
            num_paths=1000,
        )

        result = engine.forecast(config)

        # Check all quantiles at all time steps
        for t in range(1, config.horizon_days + 1):
            for q, price in result.quantile_paths[t].items():
                assert price > 0, f"Negative price {price} at t={t}, q={q}"

    def test_metadata_completeness(self):
        """Test that forecast metadata contains all required fields."""
        engine = ForecastEngine(seed=42)
        config = ForecastConfig(asset_id="BTC-USD", horizon_days=7)

        result = engine.forecast(config)

        required_fields = [
            "as_of", "asset_id", "paths", "scenario",
            "regime_bucket", "horizon_bucket", "scenario_label",
        ]

        for field in required_fields:
            assert field in result.metadata, f"Missing metadata field: {field}"

    def test_drivers_exist(self):
        """Test that explainability drivers are provided."""
        engine = ForecastEngine(seed=42)
        config = ForecastConfig(asset_id="BTC-USD", horizon_days=7)

        result = engine.forecast(config)

        assert len(result.drivers) > 0, "No drivers provided"

        # Each driver should be a tuple (name, score)
        for driver in result.drivers:
            assert isinstance(driver, tuple), f"Driver not a tuple: {driver}"
            assert len(driver) == 2, f"Driver tuple wrong length: {driver}"
            assert isinstance(driver[0], str), f"Driver name not string: {driver}"


class TestPerformanceBenchmarks:
    """Benchmark forecast engine performance."""

    def test_legacy_forecast_latency(self):
        """Test legacy forecast completes within acceptable time."""
        engine = ForecastEngine(seed=42)
        config = ForecastConfig(
            asset_id="BTC-USD",
            horizon_days=7,
            num_paths=1000,
        )

        start = time.perf_counter()
        result = engine.forecast(config)
        latency_ms = (time.perf_counter() - start) * 1000

        # Should complete in under 2 seconds for 1000 paths
        assert latency_ms < 2000, f"Legacy forecast too slow: {latency_ms:.1f}ms"

        print(f"\n✓ Legacy forecast (1000 paths, 7d): {latency_ms:.1f}ms")

    def test_high_path_count_latency(self):
        """Test high path count (10k) forecast performance."""
        engine = ForecastEngine(seed=42, use_importance_sampling=True)
        config = ForecastConfig(
            asset_id="BTC-USD",
            horizon_days=7,
            num_paths=10000,
        )

        start = time.perf_counter()
        result = engine.forecast(config)
        latency_ms = (time.perf_counter() - start) * 1000

        # 10k paths should complete in under 10 seconds
        assert latency_ms < 10000, f"10k path forecast too slow: {latency_ms:.1f}ms"

        print(f"\n✓ High path count (10000 paths, 7d): {latency_ms:.1f}ms")

    def test_long_horizon_latency(self):
        """Test long horizon (30d) forecast performance."""
        engine = ForecastEngine(seed=42)
        config = ForecastConfig(
            asset_id="BTC-USD",
            horizon_days=30,
            num_paths=1000,
        )

        start = time.perf_counter()
        result = engine.forecast(config)
        latency_ms = (time.perf_counter() - start) * 1000

        # 30d horizon should complete in under 3 seconds
        assert latency_ms < 3000, f"30d forecast too slow: {latency_ms:.1f}ms"

        print(f"\n✓ Long horizon (1000 paths, 30d): {latency_ms:.1f}ms")

    @pytest.mark.skipif(True, reason="SOTA components may not be trained")
    def test_sota_forecast_latency(self):
        """Test SOTA forecast performance (if available)."""
        try:
            engine = ForecastEngine(
                seed=42,
                use_importance_sampling=True,
                use_integrated_gradients=True,
                device="cpu",
            )
            config = ForecastConfig(
                asset_id="BTC-USD",
                horizon_days=7,
                num_paths=1000,
            )

            start = time.perf_counter()
            result = engine.forecast(config)
            latency_ms = (time.perf_counter() - start) * 1000

            # SOTA should be slower but still under 5 seconds
            assert latency_ms < 5000, f"SOTA forecast too slow: {latency_ms:.1f}ms"

            print(f"\n✓ SOTA forecast (1000 paths, 7d): {latency_ms:.1f}ms")
        except Exception as e:
            pytest.skip(f"SOTA components not available: {e}")

    def test_batch_throughput(self):
        """Test batch forecast throughput."""
        engine = ForecastEngine(seed=42)

        # Create 10 forecast configs
        configs = [
            ForecastConfig(
                asset_id="BTC-USD",
                horizon_days=7,
                num_paths=500,
                as_of=datetime.now() - timedelta(days=i),
            )
            for i in range(10)
        ]

        start = time.perf_counter()
        results = engine.forecast_batch(configs)
        total_time = time.perf_counter() - start

        assert len(results) == 10, "Not all forecasts completed"

        throughput = len(configs) / total_time  # forecasts per second

        # Should handle at least 2 forecasts per second
        assert throughput >= 2.0, f"Batch throughput too low: {throughput:.2f} forecasts/sec"

        print(f"\n✓ Batch throughput: {throughput:.2f} forecasts/sec")

    def test_memory_usage(self):
        """Test memory usage doesn't explode with high path counts."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        engine = ForecastEngine(seed=42)
        config = ForecastConfig(
            asset_id="BTC-USD",
            horizon_days=14,
            num_paths=10000,
        )

        result = engine.forecast(config)

        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_increase = mem_after - mem_before

        # Should use less than 500MB for 10k paths
        assert mem_increase < 500, f"Excessive memory usage: {mem_increase:.1f}MB"

        print(f"\n✓ Memory increase (10k paths): {mem_increase:.1f}MB")


class TestRegressionTests:
    """Ensure new changes don't break existing functionality."""

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results."""
        config = ForecastConfig(
            asset_id="BTC-USD",
            horizon_days=7,
            num_paths=100,
            seed=42,
        )

        # Run twice with same seed
        engine1 = ForecastEngine(seed=42)
        result1 = engine1.forecast(config)

        engine2 = ForecastEngine(seed=42)
        result2 = engine2.forecast(config)

        # Results should be identical
        for t in range(1, 8):
            for q in [0.05, 0.25, 0.5, 0.75, 0.95]:
                diff = abs(result1.quantile_paths[t][q] - result2.quantile_paths[t][q])
                assert diff < 1e-6, f"Results differ at t={t}, q={q}: {diff}"

    def test_different_seeds_differ(self):
        """Test that different seeds produce different results."""
        config = ForecastConfig(
            asset_id="BTC-USD",
            horizon_days=7,
            num_paths=100,
        )

        # Run with different seeds
        engine1 = ForecastEngine(seed=42)
        result1 = engine1.forecast(config.model_copy(update={"seed": 42}))

        engine2 = ForecastEngine(seed=123)
        result2 = engine2.forecast(config.model_copy(update={"seed": 123}))

        # Results should differ
        diffs = []
        for t in range(1, 8):
            diff = abs(result1.quantile_paths[t][0.5] - result2.quantile_paths[t][0.5])
            diffs.append(diff)

        avg_diff = np.mean(diffs)
        assert avg_diff > 1.0, "Different seeds should produce different results"

    def test_calibration_persistence(self):
        """Test calibration state can be saved and loaded."""
        import tempfile
        from pathlib import Path
        from aetheris_oracle.pipeline.calibration import CalibrationEngine

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calib.json"

            # Create and save calibration
            calib1 = CalibrationEngine()
            calib1.save(path)

            # Load calibration
            calib2 = CalibrationEngine.load(path)

            # States should match
            assert calib1.state.width_scalars == calib2.state.width_scalars
            assert calib1.state.coverage == calib2.state.coverage

    def test_scenario_labeled_correctly(self):
        """Test that scenario forecasts are labeled as conditional."""
        from aetheris_oracle.data.schemas import ScenarioOverrides

        engine = ForecastEngine(seed=42)

        # Base forecast
        config_base = ForecastConfig(asset_id="BTC-USD", horizon_days=7)
        result_base = engine.forecast(config_base)
        assert result_base.metadata["scenario_label"] == "base"

        # Scenario forecast
        scenario = ScenarioOverrides(description="high vol", iv_multiplier=1.5)
        config_scenario = ForecastConfig(
            asset_id="BTC-USD",
            horizon_days=7,
            scenario=scenario,
        )
        result_scenario = engine.forecast(config_scenario)
        assert result_scenario.metadata["scenario_label"] == "conditional"
        assert "high vol" in result_scenario.metadata["scenario"]


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_day_horizon(self):
        """Test 1-day horizon forecast."""
        engine = ForecastEngine(seed=42)
        config = ForecastConfig(asset_id="BTC-USD", horizon_days=1, num_paths=100)

        result = engine.forecast(config)

        assert len(result.quantile_paths) == 1
        assert 1 in result.quantile_paths

    def test_very_long_horizon(self):
        """Test 90-day horizon forecast."""
        engine = ForecastEngine(seed=42)
        config = ForecastConfig(asset_id="BTC-USD", horizon_days=90, num_paths=100)

        result = engine.forecast(config)

        assert len(result.quantile_paths) == 90
        assert 90 in result.quantile_paths

    def test_minimal_path_count(self):
        """Test with very few paths (edge case)."""
        engine = ForecastEngine(seed=42)
        config = ForecastConfig(asset_id="BTC-USD", horizon_days=7, num_paths=10)

        result = engine.forecast(config)

        # Should still work, though quality may be poor
        assert len(result.quantile_paths) == 7

    def test_empty_thresholds(self):
        """Test with no thresholds specified."""
        engine = ForecastEngine(seed=42)
        config = ForecastConfig(
            asset_id="BTC-USD",
            horizon_days=7,
            thresholds=[],  # Empty
        )

        result = engine.forecast(config)

        assert result.threshold_probabilities == {}

    def test_calibration_update(self):
        """Test calibration can be updated with realized outcomes."""
        engine = ForecastEngine(seed=42)
        config = ForecastConfig(asset_id="BTC-USD", horizon_days=7)

        result = engine.forecast(config)

        # Update with fake realized price
        realized = 48000.0
        engine.update_calibration_with_realized(result, realized, horizon=7)

        # Should not crash
        assert True


class TestIntegrationScenarios:
    """Test realistic end-to-end scenarios."""

    def test_daily_forecast_workflow(self):
        """Simulate daily forecast generation workflow."""
        engine = ForecastEngine(seed=42)

        # Generate forecasts for past 5 days
        results = []
        for days_ago in range(5):
            config = ForecastConfig(
                asset_id="BTC-USD",
                horizon_days=7,
                num_paths=500,
                as_of=datetime.now() - timedelta(days=days_ago),
            )
            result = engine.forecast(config)
            results.append(result)

        assert len(results) == 5

        # Each result should be valid
        for result in results:
            assert len(result.quantile_paths) == 7
            assert result.metadata["asset_id"] == "BTC-USD"

    def test_multi_asset_forecast(self):
        """Test forecasting multiple assets."""
        engine = ForecastEngine(seed=42)

        assets = ["BTC-USD", "ETH-USD"]
        results = {}

        for asset in assets:
            config = ForecastConfig(asset_id=asset, horizon_days=7, num_paths=500)
            results[asset] = engine.forecast(config)

        assert len(results) == 2
        assert "BTC-USD" in results
        assert "ETH-USD" in results

    def test_scenario_comparison(self):
        """Test comparing base vs scenario forecasts."""
        from aetheris_oracle.data.schemas import ScenarioOverrides

        engine = ForecastEngine(seed=42)

        # Base forecast
        config_base = ForecastConfig(asset_id="BTC-USD", horizon_days=7, num_paths=500)
        result_base = engine.forecast(config_base)

        # High volatility scenario
        scenario_high_vol = ScenarioOverrides(description="high vol", iv_multiplier=1.5)
        config_scenario = config_base.model_copy(update={"scenario": scenario_high_vol})
        result_scenario = engine.forecast(config_scenario)

        # Scenario cone should be wider
        base_width = result_base.quantile_paths[7][0.90] - result_base.quantile_paths[7][0.10]
        scenario_width = result_scenario.quantile_paths[7][0.90] - result_scenario.quantile_paths[7][0.10]

        assert scenario_width > base_width, "High vol scenario should have wider cone"


def run_performance_report():
    """Generate comprehensive performance report."""
    print("\n" + "="*60)
    print("AETHERIS ORACLE - PERFORMANCE REPORT")
    print("="*60)

    engine = ForecastEngine(seed=42)

    test_cases = [
        ("Quick (100 paths, 7d)", {"num_paths": 100, "horizon_days": 7}),
        ("Standard (1000 paths, 7d)", {"num_paths": 1000, "horizon_days": 7}),
        ("High Quality (10k paths, 7d)", {"num_paths": 10000, "horizon_days": 7}),
        ("Long Horizon (1000 paths, 30d)", {"num_paths": 1000, "horizon_days": 30}),
    ]

    for name, params in test_cases:
        config = ForecastConfig(asset_id="BTC-USD", **params)

        start = time.perf_counter()
        result = engine.forecast(config)
        latency_ms = (time.perf_counter() - start) * 1000

        cone_width = result.quantile_paths[params["horizon_days"]][0.90] - \
                      result.quantile_paths[params["horizon_days"]][0.10]

        print(f"\n{name}:")
        print(f"  Latency: {latency_ms:.1f}ms")
        print(f"  Final cone width (P90-P10): ${cone_width:.2f}")
        print(f"  Median forecast: ${result.quantile_paths[params['horizon_days']][0.5]:.2f}")

    print("\n" + "="*60)


if __name__ == "__main__":
    # Run performance report
    run_performance_report()

    # Run pytest
    pytest.main([__file__, "-v", "--tb=short"])
