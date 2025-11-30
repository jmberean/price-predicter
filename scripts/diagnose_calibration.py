"""
Diagnostic script to investigate forecast calibration issues.

Examines:
1. Current forecast quantile spread
2. Component-by-component outputs
3. NCC calibration behavior
"""

import sys
sys.path.insert(0, "src")

from datetime import datetime, timedelta
from aetheris_oracle.config import ForecastConfig
from aetheris_oracle.pipeline.forecast import ForecastEngine
from aetheris_oracle.data.free_connectors import FreeDataConnector

print("=" * 70)
print("CALIBRATION DIAGNOSTIC")
print("=" * 70)

connector = FreeDataConnector(enable_cache=True)

# Get current price
frame = connector.fetch_window("BTC-USD", datetime.utcnow(), timedelta(days=1))
current_price = frame.closes[-1]
print(f"\nCurrent BTC price: ${current_price:,.0f}")

# Test 1: SOTA without NCC (baseline)
print("\n" + "=" * 70)
print("[1/3] Testing SOTA WITHOUT NCC calibration...")
print("=" * 70)

engine_no_ncc = ForecastEngine(
    connector=connector,
    use_ncc_calibration=False,  # No calibration
    use_diff_greeks=True,
    use_fm_gp_residuals=True,
    use_neural_rough_vol=True,
    use_neural_jumps=True,
    use_mamba_trend=False,
    diff_greeks_artifact_path="artifacts/diff_greeks.pt",
    fmgp_artifact_path="artifacts/fmgp_residuals.pt",
    neural_vol_artifact_path="artifacts/neural_rough_vol.pt",
    neural_jump_artifact_path="artifacts/neural_jump_sde.pt",
)

config = ForecastConfig(asset_id="BTC-USD", horizon_days=7, num_paths=1000, seed=42)
result_no_ncc = engine_no_ncc.forecast(config)

p05 = result_no_ncc.quantile_paths[7][0.05]
p10 = result_no_ncc.quantile_paths[7][0.1]
p50 = result_no_ncc.quantile_paths[7][0.5]
p90 = result_no_ncc.quantile_paths[7][0.9]
p95 = result_no_ncc.quantile_paths[7][0.95]

print(f"\nDay 7 Forecast (no NCC):")
print(f"  P05: ${p05:>10,.0f}")
print(f"  P10: ${p10:>10,.0f}")
print(f"  P50: ${p50:>10,.0f}")
print(f"  P90: ${p90:>10,.0f}")
print(f"  P95: ${p95:>10,.0f}")

spread_80 = (p90 - p10) / p50 * 100
spread_90 = (p95 - p05) / p50 * 100
print(f"\nP10-P90 Spread: {spread_80:.1f}%")
print(f"P05-P95 Spread: {spread_90:.1f}%")
print(f"Expected: 30-40% for P10-P90, 40-60% for P05-P95")

# Test 2: SOTA with NCC
print("\n" + "=" * 70)
print("[2/3] Testing SOTA WITH NCC calibration...")
print("=" * 70)

engine_with_ncc = ForecastEngine(
    connector=connector,
    use_ncc_calibration=True,  # With calibration
    use_diff_greeks=True,
    use_fm_gp_residuals=True,
    use_neural_rough_vol=True,
    use_neural_jumps=True,
    use_mamba_trend=False,
    ncc_artifact_path="artifacts/ncc_calibration.pt",
    diff_greeks_artifact_path="artifacts/diff_greeks.pt",
    fmgp_artifact_path="artifacts/fmgp_residuals.pt",
    neural_vol_artifact_path="artifacts/neural_rough_vol.pt",
    neural_jump_artifact_path="artifacts/neural_jump_sde.pt",
)

result_with_ncc = engine_with_ncc.forecast(config)

p05_ncc = result_with_ncc.quantile_paths[7][0.05]
p10_ncc = result_with_ncc.quantile_paths[7][0.1]
p50_ncc = result_with_ncc.quantile_paths[7][0.5]
p90_ncc = result_with_ncc.quantile_paths[7][0.9]
p95_ncc = result_with_ncc.quantile_paths[7][0.95]

print(f"\nDay 7 Forecast (with NCC):")
print(f"  P05: ${p05_ncc:>10,.0f}")
print(f"  P10: ${p10_ncc:>10,.0f}")
print(f"  P50: ${p50_ncc:>10,.0f}")
print(f"  P90: ${p90_ncc:>10,.0f}")
print(f"  P95: ${p95_ncc:>10,.0f}")

spread_80_ncc = (p90_ncc - p10_ncc) / p50_ncc * 100
spread_90_ncc = (p95_ncc - p05_ncc) / p50_ncc * 100
print(f"\nP10-P90 Spread: {spread_80_ncc:.1f}%")
print(f"P05-P95 Spread: {spread_90_ncc:.1f}%")

# Test 3: Compare NCC effect
print("\n" + "=" * 70)
print("[3/3] NCC Calibration Effect Analysis")
print("=" * 70)

print(f"\nNCC Effect on Spread:")
print(f"  Before NCC: {spread_80:.1f}%")
print(f"  After NCC:  {spread_80_ncc:.1f}%")
spread_change = spread_80_ncc - spread_80
print(f"  Change:     {spread_change:+.1f}%")

if spread_change < -5:
    print(f"  ⚠️ NCC is NARROWING the forecast (making it more overconfident)")
elif spread_change > 5:
    print(f"  ✅ NCC is WIDENING the forecast (adding uncertainty)")
else:
    print(f"  ℹ️ NCC has minimal effect")

print(f"\nDiagnosis:")
if spread_80_ncc < 20:
    print(f"  ❌ CRITICAL: Forecast spread is way too narrow ({spread_80_ncc:.1f}%)")
    print(f"  Root cause: Models are generating overly confident predictions")
elif spread_80_ncc < 25:
    print(f"  ⚠️ WARNING: Forecast spread is too narrow ({spread_80_ncc:.1f}%)")
    print(f"  Expected: 30-40% for good calibration")
elif spread_80_ncc > 50:
    print(f"  ⚠️ WARNING: Forecast spread is too wide ({spread_80_ncc:.1f}%)")
    print(f"  Models may be too uncertain")
else:
    print(f"  ✅ GOOD: Forecast spread is reasonable ({spread_80_ncc:.1f}%)")

print("\n" + "=" * 70)
