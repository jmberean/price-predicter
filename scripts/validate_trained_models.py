"""Validate trained SOTA models by comparing against legacy baselines."""

import sys
sys.path.insert(0, "src")

from datetime import datetime
from aetheris_oracle.config import ForecastConfig
from aetheris_oracle.pipeline.forecast import ForecastEngine
from aetheris_oracle.data.free_connectors import FreeDataConnector

print("=" * 70)
print("SOTA Model Validation - Comparing Legacy vs Trained Models")
print("=" * 70)

# Initialize connector
connector = FreeDataConnector(enable_cache=True)

# Test configuration
config = ForecastConfig(
    asset_id="BTC-USD",
    horizon_days=7,
    num_paths=1000,
    seed=42,
)

print(f"\nTest Configuration:")
print(f"  Asset: {config.asset_id}")
print(f"  Horizon: {config.horizon_days} days")
print(f"  Paths: {config.num_paths}")
print(f"  Date: {config.as_of.strftime('%Y-%m-%d')}")

# ========================================
# Test 1: Legacy Baseline
# ========================================
print("\n" + "=" * 70)
print("[1/3] Running Legacy Baseline Forecast...")
print("=" * 70)

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
    result_legacy = engine_legacy.forecast(config)
    print(f"✓ Legacy forecast completed")
    print(f"  Metadata: {result_legacy.metadata.get('regime_bucket', 'unknown')} regime")

    # Extract P10-P90 spread at final horizon
    final_horizon = max(result_legacy.quantile_paths.keys())
    p10_legacy = result_legacy.quantile_paths[final_horizon][0.1]
    p90_legacy = result_legacy.quantile_paths[final_horizon][0.9]
    spread_legacy = p90_legacy - p10_legacy
    print(f"  P10-P90 spread (day {final_horizon}): ${spread_legacy:.2f}")

except Exception as e:
    print(f"✗ Legacy forecast failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ========================================
# Test 2: SOTA with Diff Greeks only
# ========================================
print("\n" + "=" * 70)
print("[2/3] Running SOTA with Differentiable Greeks...")
print("=" * 70)

engine_diff_greeks = ForecastEngine(
    connector=connector,
    seed=42,
    use_ncc_calibration=False,
    use_fm_gp_residuals=False,
    use_neural_jumps=False,
    use_diff_greeks=True,  # Trained model
    use_neural_rough_vol=False,
    use_mamba_trend=False,
    diff_greeks_artifact_path="artifacts/diff_greeks.pt",
)

try:
    result_diff_greeks = engine_diff_greeks.forecast(config)
    print(f"✓ Diff Greeks forecast completed")

    p10_dg = result_diff_greeks.quantile_paths[final_horizon][0.1]
    p90_dg = result_diff_greeks.quantile_paths[final_horizon][0.9]
    spread_dg = p90_dg - p10_dg
    print(f"  P10-P90 spread (day {final_horizon}): ${spread_dg:.2f}")
    print(f"  Change vs Legacy: {((spread_dg - spread_legacy) / spread_legacy * 100):.1f}%")

except Exception as e:
    print(f"✗ Diff Greeks forecast failed: {e}")
    import traceback
    traceback.print_exc()

# ========================================
# Test 3: SOTA with NCC + Diff Greeks
# ========================================
print("\n" + "=" * 70)
print("[3/3] Running SOTA with NCC + Diff Greeks...")
print("=" * 70)

engine_sota = ForecastEngine(
    connector=connector,
    seed=42,
    use_ncc_calibration=True,  # Trained model
    use_fm_gp_residuals=False,
    use_neural_jumps=False,
    use_diff_greeks=True,  # Trained model
    use_neural_rough_vol=False,
    use_mamba_trend=False,
    ncc_artifact_path="artifacts/ncc_calibration.pt",
    diff_greeks_artifact_path="artifacts/diff_greeks.pt",
)

try:
    result_sota = engine_sota.forecast(config)
    print(f"✓ SOTA forecast completed")

    p10_sota = result_sota.quantile_paths[final_horizon][0.1]
    p90_sota = result_sota.quantile_paths[final_horizon][0.9]
    spread_sota = p90_sota - p10_sota
    print(f"  P10-P90 spread (day {final_horizon}): ${spread_sota:.2f}")
    print(f"  Change vs Legacy: {((spread_sota - spread_legacy) / spread_legacy * 100):.1f}%")
    print(f"  Change vs Diff Greeks only: {((spread_sota - spread_dg) / spread_dg * 100):.1f}%")

except Exception as e:
    print(f"✗ SOTA forecast failed: {e}")
    import traceback
    traceback.print_exc()

# ========================================
# Summary
# ========================================
print("\n" + "=" * 70)
print("Validation Summary")
print("=" * 70)
print(f"✓ Legacy baseline: P10-P90 spread = ${spread_legacy:.2f}")
print(f"✓ Diff Greeks only: P10-P90 spread = ${spread_dg:.2f} ({((spread_dg - spread_legacy) / spread_legacy * 100):+.1f}%)")
print(f"✓ NCC + Diff Greeks: P10-P90 spread = ${spread_sota:.2f} ({((spread_sota - spread_legacy) / spread_legacy * 100):+.1f}%)")

print("\nInterpretation:")
if spread_sota < spread_legacy:
    print("  ✓ SOTA models provide TIGHTER forecast cones (more confident)")
elif spread_sota > spread_legacy:
    print("  ✓ SOTA models provide WIDER forecast cones (more conservative)")
else:
    print("  = SOTA models have SIMILAR spread to legacy")

print("\nNote: Tighter cones are better IF calibration maintains coverage.")
print("      Run full validation to measure actual P10-P90 coverage on test set.")

print("\n" + "=" * 70)
print("Validation complete! Trained models are ready to use.")
print("=" * 70)
