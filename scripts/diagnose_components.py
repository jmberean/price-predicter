"""
Diagnose which SOTA component is causing extreme tightening.

Tests each component individually to isolate the problem.
"""

import sys
sys.path.insert(0, "src")

from aetheris_oracle.config import ForecastConfig
from aetheris_oracle.pipeline.forecast import ForecastEngine
from aetheris_oracle.data.free_connectors import FreeDataConnector

print("=" * 70)
print("Component-by-Component Diagnosis")
print("=" * 70)

connector = FreeDataConnector(enable_cache=True)

config = ForecastConfig(
    asset_id="BTC-USD",
    horizon_days=7,
    num_paths=1000,
    seed=42,
)

def get_spread(engine, config):
    """Run forecast and return P10-P90 spread."""
    result = engine.forecast(config)
    q10 = result.quantile_paths[7][0.1]
    q90 = result.quantile_paths[7][0.9]
    return q90 - q10

# Baseline: Legacy (all SOTA disabled)
print("\n[1/7] Legacy Baseline (all SOTA disabled)...")
engine = ForecastEngine(connector=connector, seed=42)
spread_legacy = get_spread(engine, config)
print(f"  P10-P90 spread: ${spread_legacy:,.2f}")

# Test 2: Only Mamba Trend
print("\n[2/7] Only Mamba Trend...")
engine = ForecastEngine(
    connector=connector,
    seed=42,
    use_mamba_trend=True,
    mamba_artifact_path="artifacts/mamba_trend.pt",
)
spread_mamba = get_spread(engine, config)
change = (spread_mamba - spread_legacy) / spread_legacy * 100
print(f"  P10-P90 spread: ${spread_mamba:,.2f} ({change:+.1f}%)")

# Test 3: Only Neural Rough Vol
print("\n[3/7] Only Neural Rough Vol...")
engine = ForecastEngine(
    connector=connector,
    seed=42,
    use_neural_rough_vol=True,
    neural_vol_artifact_path="artifacts/neural_rough_vol.pt",
)
spread_nvol = get_spread(engine, config)
change = (spread_nvol - spread_legacy) / spread_legacy * 100
print(f"  P10-P90 spread: ${spread_nvol:,.2f} ({change:+.1f}%)")

# Test 4: Only FM-GP Residuals
print("\n[4/7] Only FM-GP Residuals...")
engine = ForecastEngine(
    connector=connector,
    seed=42,
    use_fm_gp_residuals=True,
    fmgp_artifact_path="artifacts/fmgp_residuals.pt",
)
spread_fmgp = get_spread(engine, config)
change = (spread_fmgp - spread_legacy) / spread_legacy * 100
print(f"  P10-P90 spread: ${spread_fmgp:,.2f} ({change:+.1f}%)")

# Test 5: Only Diff Greeks
print("\n[5/7] Only Diff Greeks...")
engine = ForecastEngine(
    connector=connector,
    seed=42,
    use_diff_greeks=True,
    diff_greeks_artifact_path="artifacts/diff_greeks.pt",
)
spread_greeks = get_spread(engine, config)
change = (spread_greeks - spread_legacy) / spread_legacy * 100
print(f"  P10-P90 spread: ${spread_greeks:,.2f} ({change:+.1f}%)")

# Test 6: Mamba + Neural Vol (most likely culprits)
print("\n[6/7] Mamba + Neural Vol...")
engine = ForecastEngine(
    connector=connector,
    seed=42,
    use_mamba_trend=True,
    use_neural_rough_vol=True,
    mamba_artifact_path="artifacts/mamba_trend.pt",
    neural_vol_artifact_path="artifacts/neural_rough_vol.pt",
)
spread_mamba_nvol = get_spread(engine, config)
change = (spread_mamba_nvol - spread_legacy) / spread_legacy * 100
print(f"  P10-P90 spread: ${spread_mamba_nvol:,.2f} ({change:+.1f}%)")

# Test 7: Full SOTA (all components)
print("\n[7/7] Full SOTA (all components)...")
engine = ForecastEngine(
    connector=connector,
    seed=42,
    use_mamba_trend=True,
    use_neural_rough_vol=True,
    use_fm_gp_residuals=True,
    use_diff_greeks=True,
    mamba_artifact_path="artifacts/mamba_trend.pt",
    neural_vol_artifact_path="artifacts/neural_rough_vol.pt",
    fmgp_artifact_path="artifacts/fmgp_residuals.pt",
    diff_greeks_artifact_path="artifacts/diff_greeks.pt",
)
spread_full = get_spread(engine, config)
change = (spread_full - spread_legacy) / spread_legacy * 100
print(f"  P10-P90 spread: ${spread_full:,.2f} ({change:+.1f}%)")

# Summary
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)

print(f"\nLegacy:           ${spread_legacy:,.2f} (baseline)")
print(f"Mamba only:       ${spread_mamba:,.2f}")
print(f"Neural Vol only:  ${spread_nvol:,.2f}")
print(f"FM-GP only:       ${spread_fmgp:,.2f}")
print(f"Diff Greeks only: ${spread_greeks:,.2f}")
print(f"Mamba+Vol:        ${spread_mamba_nvol:,.2f}")
print(f"Full SOTA:        ${spread_full:,.2f}")

print("\n📊 Analysis:")
if spread_fmgp < spread_legacy * 0.2:
    print("  ⚠️ FM-GP Residuals are extremely tight (likely cause)")
if spread_nvol < spread_legacy * 0.2:
    print("  ⚠️ Neural Rough Vol is extremely tight (likely cause)")
if spread_mamba_nvol < spread_legacy * 0.2:
    print("  ⚠️ Mamba+Vol combination is extremely tight (interaction issue)")

print("\n" + "=" * 70)
