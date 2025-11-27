"""
Isolate which SOTA component is causing 0% coverage.

Test components one at a time to find the problematic one.
"""

import sys
sys.path.insert(0, "src")

from datetime import datetime, timedelta
from aetheris_oracle.config import ForecastConfig
from aetheris_oracle.pipeline.forecast import ForecastEngine
from aetheris_oracle.pipeline.offline_evaluation import run_walk_forward
from aetheris_oracle.data.free_connectors import FreeDataConnector

print("=" * 70)
print("SOTA Component Isolation Diagnostic")
print("=" * 70)

# Configure test period
end_date = datetime.utcnow()
test_start = end_date - timedelta(days=60)

# Generate test configs
test_configs = []
current = test_start
horizon = 7

while current < end_date - timedelta(days=horizon):
    config = ForecastConfig(
        asset_id="BTC-USD",
        horizon_days=horizon,
        as_of=current,
        num_paths=1000,
        seed=42,
    )
    test_configs.append(config)
    current += timedelta(days=10)  # Every 10 days for faster testing

print(f"\nTest period: {test_start.date()} to {end_date.date()}")
print(f"Total forecasts: {len(test_configs)}")

connector = FreeDataConnector(enable_cache=True)

# Test configurations
tests = [
    {
        "name": "Baseline (NCC + Diff Greeks)",
        "config": {
            "use_ncc_calibration": True,
            "use_diff_greeks": True,
            "use_fm_gp_residuals": False,
            "use_neural_rough_vol": False,
            "use_neural_jumps": False,
            "use_mamba_trend": False,
            "ncc_artifact_path": "artifacts/ncc_calibration.pt",
            "diff_greeks_artifact_path": "artifacts/diff_greeks.pt",
        }
    },
    {
        "name": "+ FM-GP Residuals",
        "config": {
            "use_ncc_calibration": True,
            "use_diff_greeks": True,
            "use_fm_gp_residuals": True,
            "use_neural_rough_vol": False,
            "use_neural_jumps": False,
            "use_mamba_trend": False,
            "ncc_artifact_path": "artifacts/ncc_calibration.pt",
            "diff_greeks_artifact_path": "artifacts/diff_greeks.pt",
            "fmgp_artifact_path": "artifacts/fmgp_residuals.pt",
        }
    },
    {
        "name": "+ Neural Rough Vol",
        "config": {
            "use_ncc_calibration": True,
            "use_diff_greeks": True,
            "use_fm_gp_residuals": False,
            "use_neural_rough_vol": True,
            "use_neural_jumps": False,
            "use_mamba_trend": False,
            "ncc_artifact_path": "artifacts/ncc_calibration.pt",
            "diff_greeks_artifact_path": "artifacts/diff_greeks.pt",
            "neural_vol_artifact_path": "artifacts/neural_rough_vol.pt",
        }
    },
    {
        "name": "+ Neural Jump SDE",
        "config": {
            "use_ncc_calibration": True,
            "use_diff_greeks": True,
            "use_fm_gp_residuals": False,
            "use_neural_rough_vol": False,
            "use_neural_jumps": True,
            "use_mamba_trend": False,
            "ncc_artifact_path": "artifacts/ncc_calibration.pt",
            "diff_greeks_artifact_path": "artifacts/diff_greeks.pt",
            "neural_jump_artifact_path": "artifacts/neural_jump_sde.pt",
        }
    },
    {
        "name": "+ MambaTS Trend",
        "config": {
            "use_ncc_calibration": True,
            "use_diff_greeks": True,
            "use_fm_gp_residuals": False,
            "use_neural_rough_vol": False,
            "use_neural_jumps": False,
            "use_mamba_trend": True,
            "ncc_artifact_path": "artifacts/ncc_calibration.pt",
            "diff_greeks_artifact_path": "artifacts/diff_greeks.pt",
            "mamba_artifact_path": "artifacts/mamba_trend.pt",
        }
    },
    {
        "name": "Full SOTA (ALL Components)",
        "config": {
            "use_ncc_calibration": True,
            "use_diff_greeks": True,
            "use_fm_gp_residuals": True,
            "use_neural_rough_vol": True,
            "use_neural_jumps": True,
            "use_mamba_trend": True,
            "ncc_artifact_path": "artifacts/ncc_calibration.pt",
            "diff_greeks_artifact_path": "artifacts/diff_greeks.pt",
            "fmgp_artifact_path": "artifacts/fmgp_residuals.pt",
            "neural_vol_artifact_path": "artifacts/neural_rough_vol.pt",
            "neural_jump_artifact_path": "artifacts/neural_jump_sde.pt",
            "mamba_artifact_path": "artifacts/mamba_trend.pt",
        }
    },
]

results = []

for test in tests:
    print(f"\n{'=' * 70}")
    print(f"Testing: {test['name']}")
    print("=" * 70)

    engine = ForecastEngine(connector=connector, seed=42, **test['config'])

    try:
        result = run_walk_forward(test_configs, engine)

        print(f"\n📊 Results:")
        print(f"  CRPS:             {result.crps:.2f}")
        print(f"  P10-P90 Coverage: {result.coverage['rate']:.1%}")
        print(f"  Hits:             {result.coverage['hits']}/{result.coverage['total']}")

        status = "✅ PASS" if 0.60 <= result.coverage['rate'] <= 1.0 else "❌ FAIL"
        print(f"  Status:           {status}")

        results.append({
            "name": test['name'],
            "crps": result.crps,
            "coverage": result.coverage['rate'],
            "status": status,
        })
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append({
            "name": test['name'],
            "crps": 9999,
            "coverage": 0.0,
            "status": "❌ ERROR",
        })

# Summary
print("\n" + "=" * 70)
print("Summary: Component Isolation Results")
print("=" * 70)

print(f"\n{'Component':<35s} {'CRPS':<10s} {'Coverage':<12s} {'Status':<10s}")
print("-" * 70)

for r in results:
    print(f"{r['name']:<35s} {r['crps']:<10.2f} {r['coverage']:<12.1%} {r['status']}")

print("\n📝 Interpretation:")
print("  • First failure indicates which component breaks the model")
print("  • Coverage target: 60%+ (anything below is broken)")

print("\n" + "=" * 70)
