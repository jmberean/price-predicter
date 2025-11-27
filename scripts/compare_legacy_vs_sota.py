"""
Compare legacy vs SOTA forecasting performance.

Runs both configurations and reports:
- Latency differences
- Quantile spread (sharpness)
- Model components used
"""

import sys
sys.path.insert(0, "src")

import time
import os
from datetime import datetime, timedelta

print("=" * 80)
print("🔬 LEGACY vs SOTA COMPARISON")
print("=" * 80)

from aetheris_oracle.config import ForecastConfig
from aetheris_oracle.data.free_connectors import FreeDataConnector
from aetheris_oracle.pipeline.forecast import ForecastEngine

connector = FreeDataConnector()
config = ForecastConfig(
    asset_id="BTC-USD",
    horizon_days=7,
    num_paths=1000,
    seed=42,
)

# ============================================================================
# 1. LEGACY FORECAST
# ============================================================================
print("\n" + "─" * 80)
print("1️⃣  LEGACY FORECAST (Baseline)")
print("─" * 80)

engine_legacy = ForecastEngine(
    seed=42,
    connector=connector,
    use_neural_rough_vol=False,
    use_fm_gp_residuals=False,
    use_ncc_calibration=False,
    use_neural_jumps=False,
    use_diff_greeks=False,
    use_mamba_trend=False,
)

print(f"  Neural Rough Vol: {engine_legacy.use_neural_rough_vol}")
print(f"  FM-GP Residuals: {engine_legacy.use_fm_gp_residuals}")
print(f"  Neural Jumps: {engine_legacy.use_neural_jumps}")
print(f"\n  Running forecast...")

start = time.perf_counter()
result_legacy = engine_legacy.forecast(config)
latency_legacy = time.perf_counter() - start

print(f"  ✅ Completed in {latency_legacy*1000:.1f}ms")

# Get terminal quantiles
q_legacy = result_legacy.quantile_paths[7]
p10_legacy = q_legacy[0.1]
p90_legacy = q_legacy[0.9]
p50_legacy = q_legacy[0.5]
spread_legacy = (p90_legacy - p10_legacy) / p50_legacy

print(f"  Terminal (day 7) quantiles:")
print(f"    P10: ${p10_legacy:,.2f}")
print(f"    P50: ${p50_legacy:,.2f}")
print(f"    P90: ${p90_legacy:,.2f}")
print(f"    Spread (P90-P10)/P50: {spread_legacy:.2%}")

# ============================================================================
# 2. SOTA FORECAST
# ============================================================================
print("\n" + "─" * 80)
print("2️⃣  SOTA FORECAST (Neural Components)")
print("─" * 80)

# Check if models exist
from pathlib import Path
neural_vol_path = Path("artifacts/neural_rough_vol_sota.pt")
fmgp_path = Path("artifacts/fmgp_residuals_sota.pt")

if not neural_vol_path.exists() or not fmgp_path.exists():
    print("  ⚠️  SOTA models not found. Run train_all_sota.py first.")
    sys.exit(1)

engine_sota = ForecastEngine(
    seed=42,
    connector=connector,
    use_neural_rough_vol=True,
    use_fm_gp_residuals=True,
    use_ncc_calibration=False,  # Not trained yet
    use_neural_jumps=False,  # Not trained yet
    use_diff_greeks=False,  # Not trained yet
    use_mamba_trend=False,  # Requires mamba-ssm
    neural_vol_artifact_path=str(neural_vol_path),
    fmgp_artifact_path=str(fmgp_path),
)

print(f"  Neural Rough Vol: {engine_sota.use_neural_rough_vol}")
print(f"  FM-GP Residuals: {engine_sota.use_fm_gp_residuals}")
print(f"  Neural Jumps: {engine_sota.use_neural_jumps}")
print(f"\n  Running forecast...")

start = time.perf_counter()
result_sota = engine_sota.forecast(config)
latency_sota = time.perf_counter() - start

print(f"  ✅ Completed in {latency_sota*1000:.1f}ms")

# Get terminal quantiles
q_sota = result_sota.quantile_paths[7]
p10_sota = q_sota[0.1]
p90_sota = q_sota[0.9]
p50_sota = q_sota[0.5]
spread_sota = (p90_sota - p10_sota) / p50_sota

print(f"  Terminal (day 7) quantiles:")
print(f"    P10: ${p10_sota:,.2f}")
print(f"    P50: ${p50_sota:,.2f}")
print(f"    P90: ${p90_sota:,.2f}")
print(f"    Spread (P90-P10)/P50: {spread_sota:.2%}")

# ============================================================================
# 3. COMPARISON SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("📊 COMPARISON SUMMARY")
print("=" * 80)

latency_overhead = ((latency_sota / latency_legacy) - 1) * 100
spread_change = ((spread_sota / spread_legacy) - 1) * 100

print(f"\n⏱️  Latency:")
print(f"  Legacy: {latency_legacy*1000:.1f}ms")
print(f"  SOTA: {latency_sota*1000:.1f}ms")
if latency_overhead > 0:
    print(f"  SOTA is {latency_overhead:.1f}% slower (expected for neural models)")
else:
    print(f"  SOTA is {-latency_overhead:.1f}% faster (surprising!)")

print(f"\n📏 Forecast Spread (Sharpness):")
print(f"  Legacy: {spread_legacy:.2%}")
print(f"  SOTA: {spread_sota:.2%}")
if spread_change < 0:
    print(f"  SOTA is {-spread_change:.1f}% tighter (better - more confident)")
else:
    print(f"  SOTA is {spread_change:.1f}% wider (less confident)")

print(f"\n💰 Price Predictions (Terminal Day 7):")
print(f"  {'Quantile':<10} {'Legacy':<15} {'SOTA':<15} {'Difference':<15}")
print(f"  {'-'*10} {'-'*15} {'-'*15} {'-'*15}")
for q in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]:
    legacy_val = result_legacy.quantile_paths[7][q]
    sota_val = result_sota.quantile_paths[7][q]
    diff_pct = ((sota_val / legacy_val) - 1) * 100
    print(f"  P{int(q*100):<8} ${legacy_val:>12,.2f} ${sota_val:>12,.2f} {diff_pct:>+12.2f}%")

print("\n" + "=" * 80)
print("✅ Comparison complete!")
print("=" * 80)

print("\n📝 Notes:")
print("  - SOTA models are trained on real BTC data (120 days)")
print("  - Neural Rough Vol captures fractional Brownian motion (H≈0.1)")
print("  - FM-GP Residuals use Gaussian Process priors for temporal correlation")
print("  - Both use the same seed (42) for reproducibility")
print("  - Latency difference is acceptable for quality improvement")
