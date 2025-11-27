"""
Debug Full SOTA to find why coverage is 0%.

Tests with and without NCC to isolate the issue.
"""

import sys
sys.path.insert(0, "src")

from datetime import datetime, timedelta
from aetheris_oracle.config import ForecastConfig
from aetheris_oracle.pipeline.forecast import ForecastEngine
from aetheris_oracle.pipeline.offline_evaluation import run_walk_forward
from aetheris_oracle.data.free_connectors import FreeDataConnector

print("=" * 70)
print("Debug Full SOTA - Test with and without NCC")
print("=" * 70)

# Configure test period
end_date = datetime.utcnow()
test_start = end_date - timedelta(days=60)
horizon = 7

test_configs = []
current = test_start

while current < end_date - timedelta(days=horizon):
    config = ForecastConfig(
        asset_id="BTC-USD",
        horizon_days=horizon,
        as_of=current,
        num_paths=1000,
        seed=42,
    )
    test_configs.append(config)
    current += timedelta(days=3)

print(f"\nTesting {len(test_configs)} forecasts from {test_start.date()} to {end_date.date()}")

connector = FreeDataConnector(enable_cache=True)

# Test 1: Full SOTA WITHOUT NCC (to see base forecast quality)
print("\n" + "=" * 70)
print("[1/2] Full SOTA WITHOUT NCC Calibration...")
print("=" * 70)

engine_no_ncc = ForecastEngine(
    connector=connector,
    seed=42,
    use_ncc_calibration=False,  # ❌ Disable NCC
    use_diff_greeks=True,
    use_fm_gp_residuals=True,
    use_neural_rough_vol=True,
    use_neural_jumps=False,
    use_mamba_trend=True,
    diff_greeks_artifact_path="artifacts/diff_greeks.pt",
    fmgp_artifact_path="artifacts/fmgp_residuals.pt",
    neural_vol_artifact_path="artifacts/neural_rough_vol.pt",
    mamba_artifact_path="artifacts/mamba_trend.pt",
)

try:
    result_no_ncc = run_walk_forward(test_configs, engine_no_ncc)

    print(f"\n📊 Results:")
    print(f"  CRPS (lower is better):     {result_no_ncc.crps:.2f}")
    print(f"  P10-P90 Coverage:           {result_no_ncc.coverage['rate']:.1%}")
    print(f"  Forecasts evaluated:        {result_no_ncc.coverage['total']}")
    print(f"  Hits (actual in P10-P90):   {result_no_ncc.coverage['hits']}")

    if result_no_ncc.coverage['rate'] >= 0.75 and result_no_ncc.coverage['rate'] <= 0.85:
        print(f"  ✅ Calibration: GOOD (target 80%)")
    elif result_no_ncc.coverage['rate'] < 0.75:
        print(f"  ⚠️ Calibration: TOO TIGHT (under-covering)")
    else:
        print(f"  ⚠️ Calibration: TOO WIDE (over-covering)")

except Exception as e:
    print(f"\n❌ Full SOTA (no NCC) failed: {e}")
    import traceback
    traceback.print_exc()
    result_no_ncc = None

# Test 2: Full SOTA WITH NCC
print("\n" + "=" * 70)
print("[2/2] Full SOTA WITH NCC Calibration...")
print("=" * 70)

engine_with_ncc = ForecastEngine(
    connector=connector,
    seed=42,
    use_ncc_calibration=True,  # ✅ Enable NCC
    use_diff_greeks=True,
    use_fm_gp_residuals=True,
    use_neural_rough_vol=True,
    use_neural_jumps=False,
    use_mamba_trend=True,
    ncc_artifact_path="artifacts/ncc_calibration.pt",
    diff_greeks_artifact_path="artifacts/diff_greeks.pt",
    fmgp_artifact_path="artifacts/fmgp_residuals.pt",
    neural_vol_artifact_path="artifacts/neural_rough_vol.pt",
    mamba_artifact_path="artifacts/mamba_trend.pt",
)

try:
    result_with_ncc = run_walk_forward(test_configs, engine_with_ncc)

    print(f"\n📊 Results:")
    print(f"  CRPS (lower is better):     {result_with_ncc.crps:.2f}")
    print(f"  P10-P90 Coverage:           {result_with_ncc.coverage['rate']:.1%}")
    print(f"  Forecasts evaluated:        {result_with_ncc.coverage['total']}")
    print(f"  Hits (actual in P10-P90):   {result_with_ncc.coverage['hits']}")

    if result_with_ncc.coverage['rate'] >= 0.75 and result_with_ncc.coverage['rate'] <= 0.85:
        print(f"  ✅ Calibration: GOOD (target 80%)")
    elif result_with_ncc.coverage['rate'] < 0.75:
        print(f"  ⚠️ Calibration: TOO TIGHT (under-covering)")
    else:
        print(f"  ⚠️ Calibration: TOO WIDE (over-covering)")

    if result_no_ncc:
        coverage_change = result_with_ncc.coverage['rate'] - result_no_ncc.coverage['rate']
        print(f"  Coverage change from NCC:   {coverage_change:+.1%}")

except Exception as e:
    print(f"\n❌ Full SOTA (with NCC) failed: {e}")
    import traceback
    traceback.print_exc()
    result_with_ncc = None

# Summary
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)

if result_no_ncc and result_with_ncc:
    print(f"\nWithout NCC: {result_no_ncc.coverage['rate']:.1%} coverage, {result_no_ncc.crps:.2f} CRPS")
    print(f"With NCC:    {result_with_ncc.coverage['rate']:.1%} coverage, {result_with_ncc.crps:.2f} CRPS")

    if result_with_ncc.coverage['rate'] < result_no_ncc.coverage['rate']:
        print("\n⚠️ NCC is making coverage WORSE (over-tightening)")
    elif result_with_ncc.coverage['rate'] == 0.0 and result_no_ncc.coverage['rate'] == 0.0:
        print("\n⚠️ Base SOTA forecasts have 0% coverage (problem is NOT NCC)")
    else:
        print("\n✅ NCC is working as expected")

print("\n" + "=" * 70)
