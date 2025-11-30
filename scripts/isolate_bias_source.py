"""
Test each SOTA component individually to isolate bias source.
"""

import sys
sys.path.insert(0, "src")

from datetime import datetime, timedelta
from aetheris_oracle.config import ForecastConfig
from aetheris_oracle.pipeline.forecast import ForecastEngine
from aetheris_oracle.data.free_connectors import FreeDataConnector

print("=" * 70)
print("ISOLATING BIAS SOURCE - Component-by-Component Testing")
print("=" * 70)

connector = FreeDataConnector(enable_cache=True)

# Test period: last 30 days
end_date = datetime.utcnow()
test_start = end_date - timedelta(days=30)

# Test configurations
configs = [
    {
        "name": "Legacy (Baseline)",
        "use_ncc_calibration": False,
        "use_diff_greeks": False,
        "use_fm_gp_residuals": False,
        "use_neural_rough_vol": False,
        "use_neural_jumps": False,
    },
    {
        "name": "Only Diff Greeks",
        "use_ncc_calibration": False,
        "use_diff_greeks": True,
        "use_fm_gp_residuals": False,
        "use_neural_rough_vol": False,
        "use_neural_jumps": False,
    },
    {
        "name": "Diff Greeks + FM-GP",
        "use_ncc_calibration": False,
        "use_diff_greeks": True,
        "use_fm_gp_residuals": True,
        "use_neural_rough_vol": False,
        "use_neural_jumps": False,
    },
    {
        "name": "DG + FM-GP + Neural Vol",
        "use_ncc_calibration": False,
        "use_diff_greeks": True,
        "use_fm_gp_residuals": True,
        "use_neural_rough_vol": True,
        "use_neural_jumps": False,
    },
    {
        "name": "DG + FM-GP + Vol + Jumps",
        "use_ncc_calibration": False,
        "use_diff_greeks": True,
        "use_fm_gp_residuals": True,
        "use_neural_rough_vol": True,
        "use_neural_jumps": True,
    },
    {
        "name": "Full 5-SOTA (with NCC)",
        "use_ncc_calibration": True,
        "use_diff_greeks": True,
        "use_fm_gp_residuals": True,
        "use_neural_rough_vol": True,
        "use_neural_jumps": True,
    },
]

results = {}

for config_dict in configs:
    name = config_dict.pop("name")
    print(f"\n{'=' * 70}")
    print(f"Testing: {name}")
    print(f"{'=' * 70}")

    # Build engine with this config
    engine_kwargs = {
        "connector": connector,
        "seed": 42,
        "use_mamba_trend": False,
    }
    engine_kwargs.update(config_dict)

    # Add artifact paths if using SOTA components
    if config_dict.get("use_ncc_calibration"):
        engine_kwargs["ncc_artifact_path"] = "artifacts/ncc_calibration.pt"
    if config_dict.get("use_diff_greeks"):
        engine_kwargs["diff_greeks_artifact_path"] = "artifacts/diff_greeks.pt"
    if config_dict.get("use_fm_gp_residuals"):
        engine_kwargs["fmgp_artifact_path"] = "artifacts/fmgp_residuals.pt"
    if config_dict.get("use_neural_rough_vol"):
        engine_kwargs["neural_vol_artifact_path"] = "artifacts/neural_rough_vol.pt"
    if config_dict.get("use_neural_jumps"):
        engine_kwargs["neural_jump_artifact_path"] = "artifacts/neural_jump_sde.pt"

    engine = ForecastEngine(**engine_kwargs)

    # Test on 3 recent dates
    errors = []
    current = test_start
    test_count = 0

    while current < end_date - timedelta(days=7) and test_count < 3:
        try:
            # Generate forecast
            forecast_config = ForecastConfig(
                asset_id="BTC-USD",
                horizon_days=7,
                as_of=current,
                num_paths=1000,
                seed=42,
            )
            result = engine.forecast(forecast_config)

            # Get actual price
            actual_date = current + timedelta(days=7)
            frame = connector.fetch_window("BTC-USD", actual_date, timedelta(days=1))
            actual_price = frame.closes[-1]

            # Get forecast median
            forecast_median = result.quantile_paths[7][0.5]

            # Calculate error
            error_pct = (forecast_median - actual_price) / actual_price * 100
            errors.append(error_pct)

            print(f"  {current.date()}: Forecast=${forecast_median:>7,.0f}, Actual=${actual_price:>7,.0f}, Error={error_pct:>+6.1f}%")

            test_count += 1

        except Exception as e:
            print(f"  {current.date()}: Failed - {e}")

        current += timedelta(days=10)

    if len(errors) > 0:
        mean_error = sum(errors) / len(errors)
        results[name] = mean_error
        print(f"\n  Mean Error: {mean_error:>+6.1f}%")
    else:
        results[name] = None
        print(f"\n  No forecasts generated")

# Summary
print("\n" + "=" * 70)
print("BIAS SUMMARY - Mean Forecast Error by Configuration")
print("=" * 70)

print(f"\n{'Configuration':<30s} {'Mean Error':>15s} {'Status':>20s}")
print("-" * 70)

for name, error in results.items():
    if error is None:
        print(f"{name:<30s} {'N/A':>15s} {'Failed':>20s}")
    else:
        if abs(error) < 2:
            status = "✅ No bias"
        elif error > 5:
            status = f"❌ {error:+.1f}% too high"
        elif error < -5:
            status = f"❌ {abs(error):.1f}% too low"
        else:
            status = f"⚠️ Small bias"

        print(f"{name:<30s} {error:>+14.1f}% {status:>20s}")

print("\n" + "=" * 70)
print("DIAGNOSIS")
print("=" * 70)

# Find where bias is introduced
prev_error = 0
for name, error in results.items():
    if error is not None:
        if prev_error != 0:
            delta = error - prev_error
            if abs(delta) > 3:
                print(f"\n⚠️ Adding {name} changed bias by {delta:+.1f}%")
                if delta > 3:
                    print(f"   → This component introduces UPWARD bias!")
                elif delta < -3:
                    print(f"   → This component introduces DOWNWARD bias!")
        prev_error = error

print("\n" + "=" * 70)
