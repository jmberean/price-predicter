"""
Simple forecast script - reads all config from .env file.

Usage:
    python run.py              # Run forecast with .env config
    python run.py --legacy     # Use legacy models instead of SOTA

Config is read from .env - see .env.example for all options.
"""

import os
import sys
sys.path.insert(0, "src")

from dotenv import load_dotenv
load_dotenv()

import argparse
from datetime import datetime, timedelta
from pathlib import Path


def get_env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def get_env_int(key: str, default: int) -> int:
    val = os.getenv(key, "")
    return int(val) if val.isdigit() else default


def get_env_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key, "").lower()
    if val in ("true", "1", "yes"):
        return True
    elif val in ("false", "0", "no"):
        return False
    return default


def main():
    parser = argparse.ArgumentParser(description="Run forecast (config from .env)")
    parser.add_argument("--legacy", action="store_true", help="Use legacy models instead of SOTA")
    args = parser.parse_args()

    # Read config from .env
    asset = get_env("FORECAST_ASSET", "BTC-USD")
    horizon = get_env_int("FORECAST_HORIZON", 7)
    paths = get_env_int("FORECAST_PATHS", 1000)
    show_chart = get_env_bool("FORECAST_SHOW_CHART", True)
    save_chart = get_env("FORECAST_SAVE_CHART", "")
    lookback = get_env_int("FORECAST_LOOKBACK_DAYS", 30)
    device = get_env("TORCH_DEVICE", "cpu")

    # SOTA flags from .env (or all False if --legacy)
    use_sota = not args.legacy

    print()
    print("=" * 70)
    print(f"  AETHERIS ORACLE - {asset} {horizon}-Day Probabilistic Forecast")
    if use_sota:
        print("  5-SOTA Configuration")
    else:
        print("  Legacy Configuration")
    print("=" * 70)
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Import after dotenv loaded
    from aetheris_oracle.config import ForecastConfig
    from aetheris_oracle.data.free_connectors import FreeDataConnector
    from aetheris_oracle.pipeline.forecast import ForecastEngine

    connector = FreeDataConnector(enable_cache=True)

    if use_sota:
        engine = ForecastEngine(
            connector=connector,
            seed=42,
            use_ncc_calibration=get_env_bool("USE_NCC_CALIBRATION", True),
            use_diff_greeks=get_env_bool("USE_DIFF_GREEKS", True),
            use_fm_gp_residuals=get_env_bool("USE_FM_GP_RESIDUALS", True),
            use_neural_rough_vol=get_env_bool("USE_NEURAL_ROUGH_VOL", True),
            use_neural_jumps=get_env_bool("USE_NEURAL_JUMPS", True),
            use_mamba_trend=get_env_bool("USE_MAMBA_TREND", False),
            ncc_artifact_path="artifacts/ncc_calibration.pt",
            diff_greeks_artifact_path="artifacts/diff_greeks.pt",
            fmgp_artifact_path="artifacts/fmgp_residuals.pt",
            neural_vol_artifact_path="artifacts/neural_rough_vol.pt",
            neural_jump_artifact_path="artifacts/neural_jump_sde.pt",
            device=device,
        )
    else:
        engine = ForecastEngine(
            connector=connector,
            seed=42,
            use_ncc_calibration=False,
            use_diff_greeks=False,
            use_fm_gp_residuals=False,
            use_neural_rough_vol=False,
            use_neural_jumps=False,
            use_mamba_trend=False,
        )

    config = ForecastConfig(
        asset_id=asset,
        horizon_days=horizon,
        num_paths=paths,
        seed=42,
    )

    print(f"Running forecast ({paths} paths)...")
    result = engine.forecast(config)

    # Print results
    print()
    print("Forecast Cone:")
    print("-" * 70)
    print(f"{'Day':>4} {'P05':>12} {'P10':>12} {'P50':>12} {'P90':>12} {'P95':>12}")
    print("-" * 70)

    # Determine which days to show
    if horizon <= 7:
        days_to_show = list(range(1, horizon + 1))
    elif horizon <= 14:
        days_to_show = [1, 3, 7, horizon]
    else:
        days_to_show = [1, 3, 7, 14]
        if horizon > 14:
            days_to_show.append(min(21, horizon))
        if horizon > 21:
            days_to_show.append(horizon)

    for day in days_to_show:
        if day <= horizon:
            q = result.quantile_paths[day]
            print(f"{day:>4} {q[0.05]:>12,.0f} {q[0.1]:>12,.0f} {q[0.5]:>12,.0f} {q[0.9]:>12,.0f} {q[0.95]:>12,.0f}")

    print("-" * 70)
    print()

    # Terminal stats
    p05 = result.quantile_paths[horizon][0.05]
    p10 = result.quantile_paths[horizon][0.1]
    p50 = result.quantile_paths[horizon][0.5]
    p90 = result.quantile_paths[horizon][0.9]
    p95 = result.quantile_paths[horizon][0.95]

    print(f"Terminal (Day {horizon}) Summary:")
    print(f"  Median forecast:     ${p50:>10,.0f}")
    print(f"  80% confidence:      ${p10:>10,.0f} - ${p90:>10,.0f}")
    print(f"  90% confidence:      ${p05:>10,.0f} - ${p95:>10,.0f}")
    print(f"  P10-P90 spread:      {(p90-p10)/p50*100:>10.1f}%")
    print()
    print("=" * 70)

    # Chart
    if show_chart or save_chart:
        print()
        print("Generating forecast chart...")

        from aetheris_oracle.visualization import plot_forecast_cone

        try:
            frame = connector.fetch_window(asset, datetime.now(), timedelta(days=lookback))
            historical_timestamps = frame.timestamps
            historical_prices = frame.closes
        except Exception as e:
            print(f"  Warning: Could not fetch historical data for chart: {e}")
            historical_timestamps = []
            historical_prices = []

        save_path = Path(save_chart) if save_chart else None

        chart_path = plot_forecast_cone(
            historical_timestamps=historical_timestamps,
            historical_prices=historical_prices,
            forecast_start=datetime.now(),
            quantiles_by_day=result.quantile_paths,
            asset_id=asset,
            save_path=save_path,
            show=show_chart,
        )

        if chart_path:
            print(f"  Chart saved to: {chart_path}")


if __name__ == "__main__":
    main()
