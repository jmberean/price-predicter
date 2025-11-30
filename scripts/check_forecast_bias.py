"""
Check for systematic forecast bias.

Examines whether forecasts consistently over-predict or under-predict.
"""

import sys
sys.path.insert(0, "src")

from datetime import datetime, timedelta
from aetheris_oracle.config import ForecastConfig
from aetheris_oracle.pipeline.forecast import ForecastEngine
from aetheris_oracle.data.free_connectors import FreeDataConnector

print("=" * 70)
print("FORECAST BIAS ANALYSIS")
print("=" * 70)

connector = FreeDataConnector(enable_cache=True)

# Test with Full 5-SOTA
engine = ForecastEngine(
    connector=connector,
    use_ncc_calibration=True,
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

# Generate historical forecasts and check accuracy
print("\nGenerating historical 7-day forecasts...")
print("Testing last 30 days (every 3 days)")
print()

end_date = datetime.utcnow()
test_start = end_date - timedelta(days=30)

forecasts = []
actuals = []
errors = []

current = test_start
while current < end_date - timedelta(days=7):
    try:
        # Generate forecast
        config = ForecastConfig(
            asset_id="BTC-USD",
            horizon_days=7,
            as_of=current,
            num_paths=1000,
            seed=42,
        )
        result = engine.forecast(config)

        # Get actual price 7 days later
        actual_date = current + timedelta(days=7)
        frame = connector.fetch_window("BTC-USD", actual_date, timedelta(days=1))
        actual_price = frame.closes[-1]

        # Get forecast median
        forecast_median = result.quantile_paths[7][0.5]
        p10 = result.quantile_paths[7][0.1]
        p90 = result.quantile_paths[7][0.9]

        # Calculate error
        error_pct = (forecast_median - actual_price) / actual_price * 100

        # Check if actual fell in P10-P90
        in_range = p10 <= actual_price <= p90

        forecasts.append(forecast_median)
        actuals.append(actual_price)
        errors.append(error_pct)

        print(f"{current.date()}: Forecast=${forecast_median:>7,.0f}, Actual=${actual_price:>7,.0f}, "
              f"Error={error_pct:>+6.1f}%, In P10-P90: {'✓' if in_range else '✗'}")

    except Exception as e:
        print(f"{current.date()}: Failed - {e}")

    current += timedelta(days=3)

if len(errors) > 0:
    print("\n" + "=" * 70)
    print("BIAS ANALYSIS")
    print("=" * 70)

    mean_error = sum(errors) / len(errors)
    abs_errors = [abs(e) for e in errors]
    mean_abs_error = sum(abs_errors) / len(abs_errors)

    print(f"\nMean Forecast Error:     {mean_error:>+6.1f}%")
    print(f"Mean Absolute Error:     {mean_abs_error:>6.1f}%")
    print(f"Number of forecasts:     {len(errors)}")

    positive_errors = sum(1 for e in errors if e > 0)
    negative_errors = sum(1 for e in errors if e < 0)

    print(f"\nOver-predictions (too high):  {positive_errors}")
    print(f"Under-predictions (too low):  {negative_errors}")

    print(f"\nDiagnosis:")
    if abs(mean_error) < 2:
        print(f"  ✅ No significant bias")
    elif mean_error > 5:
        print(f"  ❌ SYSTEMATIC OVER-PREDICTION: Forecasts are {mean_error:.1f}% too high on average")
        print(f"     This explains poor coverage despite wide spreads!")
    elif mean_error < -5:
        print(f"  ❌ SYSTEMATIC UNDER-PREDICTION: Forecasts are {abs(mean_error):.1f}% too low on average")
        print(f"     This explains poor coverage despite wide spreads!")
    else:
        print(f"  ⚠️ Small bias detected ({mean_error:+.1f}%), but not the main issue")

    print("\n" + "=" * 70)
else:
    print("\n⚠️ No forecasts could be generated")
