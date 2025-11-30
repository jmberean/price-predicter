"""
Proper backtesting with train/test split for SOTA models.

Tests calibration: Does P10-P90 interval contain 80% of actual outcomes?
"""

import sys
sys.path.insert(0, "src")

from datetime import datetime, timedelta
from aetheris_oracle.config import ForecastConfig
from aetheris_oracle.pipeline.forecast import ForecastEngine
from aetheris_oracle.pipeline.offline_evaluation import run_walk_forward
from aetheris_oracle.data.free_connectors import FreeDataConnector

print("=" * 70)
print("SOTA Model Backtesting - Walk-Forward Validation")
print("=" * 70)

# Configure test period (use last 60 days as test set for better statistics)
end_date = datetime.utcnow()
test_start = end_date - timedelta(days=60)

# Generate walk-forward test configs (1 forecast every 3 days for more samples)
test_configs = []
current = test_start
horizon = 7  # 7-day forecasts

print(f"\nGenerating test forecasts from {test_start.date()} to {end_date.date()}")
print(f"Horizon: {horizon} days")
print(f"Frequency: Every 3 days (for more statistical samples)")

while current < end_date - timedelta(days=horizon):
    config = ForecastConfig(
        asset_id="BTC-USD",
        horizon_days=horizon,
        as_of=current,
        num_paths=1000,
        seed=42,
    )
    test_configs.append(config)
    current += timedelta(days=3)  # Every 3 days for more samples

print(f"Total forecasts to evaluate: {len(test_configs)}")

if len(test_configs) == 0:
    print("\n⚠️ Not enough historical data for backtesting")
    print("Backtesting requires at least 30 days of past data + 7 days forward")
    sys.exit(0)

# Test 1: Legacy Baseline
print("\n" + "=" * 70)
print("[1/3] Backtesting Legacy Baseline...")
print("=" * 70)

connector = FreeDataConnector(enable_cache=True)
engine_legacy = ForecastEngine(
    connector=connector,
    seed=42,
    use_ncc_calibration=False,
    use_fm_gp_residuals=False,
    use_neural_jumps=False,
    use_diff_greeks=False,
    use_neural_rough_vol=False,
    use_mamba_trend=False,
)

try:
    result_legacy = run_walk_forward(test_configs, engine_legacy)

    print(f"\n📊 Results:")
    print(f"  CRPS (lower is better):     {result_legacy.crps:.2f}")
    print(f"  P10-P90 Coverage:           {result_legacy.coverage['rate']:.1%}")
    print(f"  Forecasts evaluated:        {result_legacy.coverage['total']}")
    print(f"  Hits (actual in P10-P90):   {result_legacy.coverage['hits']}")

    if result_legacy.coverage['rate'] >= 0.75 and result_legacy.coverage['rate'] <= 0.85:
        print(f"  ✅ Calibration: GOOD (target 80%)")
    elif result_legacy.coverage['rate'] < 0.75:
        print(f"  ⚠️ Calibration: TOO TIGHT (under-covering)")
    else:
        print(f"  ⚠️ Calibration: TOO WIDE (over-covering)")

except Exception as e:
    print(f"\n❌ Legacy backtest failed: {e}")
    result_legacy = None

# Test 2: NCC + Diff Greeks
print("\n" + "=" * 70)
print("[2/3] Backtesting NCC + Diff Greeks...")
print("=" * 70)

engine_ncc = ForecastEngine(
    connector=connector,
    seed=42,
    use_ncc_calibration=True,
    use_diff_greeks=True,
    ncc_artifact_path="artifacts/ncc_calibration.pt",
    diff_greeks_artifact_path="artifacts/diff_greeks.pt",
)

try:
    result_ncc = run_walk_forward(test_configs, engine_ncc)

    print(f"\n📊 Results:")
    print(f"  CRPS (lower is better):     {result_ncc.crps:.2f}")
    print(f"  P10-P90 Coverage:           {result_ncc.coverage['rate']:.1%}")
    print(f"  Forecasts evaluated:        {result_ncc.coverage['total']}")
    print(f"  Hits (actual in P10-P90):   {result_ncc.coverage['hits']}")

    if result_ncc.coverage['rate'] >= 0.75 and result_ncc.coverage['rate'] <= 0.85:
        print(f"  ✅ Calibration: GOOD (target 80%)")
    elif result_ncc.coverage['rate'] < 0.75:
        print(f"  ⚠️ Calibration: TOO TIGHT (under-covering)")
    else:
        print(f"  ⚠️ Calibration: TOO WIDE (over-covering)")

    if result_legacy:
        crps_improvement = (result_legacy.crps - result_ncc.crps) / result_legacy.crps * 100
        print(f"  CRPS Improvement vs Legacy: {crps_improvement:+.1f}%")

except Exception as e:
    print(f"\n❌ NCC backtest failed: {e}")
    result_ncc = None

# Test 3: Full SOTA (5-component recommended config)
print("\n" + "=" * 70)
print("[3/3] Backtesting Full 5-SOTA (without Mamba)...")
print("=" * 70)

engine_full = ForecastEngine(
    connector=connector,
    seed=42,
    use_ncc_calibration=True,
    use_diff_greeks=True,
    use_fm_gp_residuals=True,
    use_neural_rough_vol=True,
    use_neural_jumps=True,
    use_mamba_trend=False,  # Disabled - has directional bias issues
    ncc_artifact_path="artifacts/ncc_calibration.pt",
    diff_greeks_artifact_path="artifacts/diff_greeks.pt",
    fmgp_artifact_path="artifacts/fmgp_residuals.pt",
    neural_vol_artifact_path="artifacts/neural_rough_vol.pt",
    neural_jump_artifact_path="artifacts/neural_jump_sde.pt",
)

try:
    result_full = run_walk_forward(test_configs, engine_full)

    print(f"\n📊 Results:")
    print(f"  CRPS (lower is better):     {result_full.crps:.2f}")
    print(f"  P10-P90 Coverage:           {result_full.coverage['rate']:.1%}")
    print(f"  Forecasts evaluated:        {result_full.coverage['total']}")
    print(f"  Hits (actual in P10-P90):   {result_full.coverage['hits']}")

    if result_full.coverage['rate'] >= 0.75 and result_full.coverage['rate'] <= 0.85:
        print(f"  ✅ Calibration: GOOD (target 80%)")
    elif result_full.coverage['rate'] < 0.75:
        print(f"  ⚠️ Calibration: TOO TIGHT (under-covering, overconfident)")
    else:
        print(f"  ⚠️ Calibration: TOO WIDE (over-covering)")

    if result_legacy:
        crps_improvement = (result_legacy.crps - result_full.crps) / result_legacy.crps * 100
        print(f"  CRPS Improvement vs Legacy: {crps_improvement:+.1f}%")

except Exception as e:
    print(f"\n❌ Full SOTA backtest failed: {e}")
    result_full = None

# Summary
print("\n" + "=" * 70)
print("Backtest Summary")
print("=" * 70)

if result_legacy and result_ncc and result_full:
    print(f"\n{'Model':<25s} {'CRPS':<10s} {'Coverage':<12s} {'Status':<20s}")
    print("-" * 70)

    legacy_status = "✅ Baseline" if 0.75 <= result_legacy.coverage['rate'] <= 0.85 else "⚠️ Miscalibrated"
    ncc_status = "✅ Good" if 0.75 <= result_ncc.coverage['rate'] <= 0.85 else "⚠️ Miscalibrated"
    full_status = "✅ Good" if 0.75 <= result_full.coverage['rate'] <= 0.85 else "⚠️ Overconfident" if result_full.coverage['rate'] < 0.75 else "⚠️ Too Wide"

    print(f"{'Legacy':<25s} {result_legacy.crps:<10.2f} {result_legacy.coverage['rate']:<12.1%} {legacy_status}")
    print(f"{'NCC + Diff Greeks':<25s} {result_ncc.crps:<10.2f} {result_ncc.coverage['rate']:<12.1%} {ncc_status}")
    print(f"{'Full SOTA':<25s} {result_full.crps:<10.2f} {result_full.coverage['rate']:<12.1%} {full_status}")

    print("\n📝 Interpretation:")
    print("  • Coverage target: 75-85% (ideally 80%)")
    print("  • CRPS: Lower is better (measures forecast sharpness + accuracy)")
    print("  • Good model: Low CRPS + 80% coverage")

print("\n" + "=" * 70)
