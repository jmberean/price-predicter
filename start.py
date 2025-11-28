"""
Convenience entrypoint to run the FastAPI service, CLI forecast, or offline evaluation.

Usage:
- Service: python start.py --mode service
- CLI:     python start.py --mode cli --asset BTC-USD --horizon 7 --paths 500
- Eval:    python start.py --mode offline_eval
- Forecast: python start.py --mode forecast --asset BTC-USD --horizon 30 --paths 2000

Note: Service configuration (API key, host, port) can be set in .env file
"""

import argparse
from datetime import datetime
from pprint import pprint

from dotenv import load_dotenv

from aetheris_oracle.cli import main as cli_main
from aetheris_oracle.config import ForecastConfig
from aetheris_oracle.pipeline.offline_evaluation import run_walk_forward
from aetheris_oracle.server import main as server_main

# Load environment variables from .env file
load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start Aetheris Oracle service, CLI forecast, or evaluation.")
    parser.add_argument("--mode", choices=["service", "cli", "offline_eval", "forecast"], default="service", help="Run mode")
    parser.add_argument("--api-key", help="API key for service mode")
    parser.add_argument("--host", default="0.0.0.0", help="Host for service mode")
    parser.add_argument("--port", type=int, default=8000, help="Port for service mode")
    # Forecast mode options
    parser.add_argument("--asset", default="BTC-USD", help="Asset to forecast")
    parser.add_argument("--horizon", type=int, default=7, help="Forecast horizon in days")
    parser.add_argument("--paths", type=int, default=2000, help="Number of Monte Carlo paths")
    parser.add_argument("--use-sota", action="store_true", default=True, help="Use 5-SOTA configuration (default: True)")
    parser.add_argument("--legacy", action="store_true", help="Use legacy models instead of SOTA")
    parser.add_argument("--plot", action="store_true", default=True, help="Show interactive chart (default: True)")
    parser.add_argument("--no-plot", action="store_true", help="Disable chart display")
    parser.add_argument("--save-plot", type=str, help="Path to save chart image")
    parser.add_argument("--lookback", type=int, default=30, help="Days of historical data to show in chart")
    return parser.parse_args()


def run_forecast(args: argparse.Namespace) -> None:
    """Run a full 5-SOTA forecast and display results."""
    from aetheris_oracle.data.free_connectors import FreeDataConnector
    from aetheris_oracle.pipeline.forecast import ForecastEngine

    use_sota = not args.legacy

    print()
    print("=" * 70)
    print(f"  AETHERIS ORACLE - {args.asset} {args.horizon}-Day Probabilistic Forecast")
    if use_sota:
        print("  5-SOTA Configuration (trained on 5 years of historical data)")
    else:
        print("  Legacy Configuration")
    print("=" * 70)
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    connector = FreeDataConnector(enable_cache=True)

    if use_sota:
        engine = ForecastEngine(
            connector=connector,
            seed=42,
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
            device="cpu",
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
        asset_id=args.asset,
        horizon_days=args.horizon,
        num_paths=args.paths,
        seed=42,
    )

    print(f"Running forecast ({args.paths} paths)...")
    result = engine.forecast(config)

    print()
    print("Forecast Cone:")
    print("-" * 70)
    print(f"{'Day':>4} {'P05':>12} {'P10':>12} {'P50':>12} {'P90':>12} {'P95':>12}")
    print("-" * 70)

    # Determine which days to show based on horizon
    if args.horizon <= 7:
        days_to_show = list(range(1, args.horizon + 1))
    elif args.horizon <= 14:
        days_to_show = [1, 3, 7] + [args.horizon]
    else:
        days_to_show = [1, 3, 7, 14]
        if args.horizon > 14:
            days_to_show.append(min(21, args.horizon))
        if args.horizon > 21:
            days_to_show.append(args.horizon)

    for day in days_to_show:
        if day <= args.horizon:
            q = result.quantile_paths[day]
            print(f"{day:>4} {q[0.05]:>12,.0f} {q[0.1]:>12,.0f} {q[0.5]:>12,.0f} {q[0.9]:>12,.0f} {q[0.95]:>12,.0f}")

    print("-" * 70)
    print()

    # Terminal stats
    terminal = args.horizon
    p05 = result.quantile_paths[terminal][0.05]
    p10 = result.quantile_paths[terminal][0.1]
    p50 = result.quantile_paths[terminal][0.5]
    p90 = result.quantile_paths[terminal][0.9]
    p95 = result.quantile_paths[terminal][0.95]

    print(f"Terminal (Day {terminal}) Summary:")
    print(f"  Median forecast:     ${p50:>10,.0f}")
    print(f"  80% confidence:      ${p10:>10,.0f} - ${p90:>10,.0f}")
    print(f"  90% confidence:      ${p05:>10,.0f} - ${p95:>10,.0f}")
    print(f"  P10-P90 spread:      {(p90-p10)/p50*100:>10.1f}%")
    print()

    # Week-by-week summary if horizon > 7
    if args.horizon >= 14:
        print("Week-by-Week Ranges (80% confidence):")
        print("-" * 50)
        weeks = [(1, 7), (2, 14), (3, 21), (4, 28), (5, 35)]
        for week, day in weeks:
            if day <= args.horizon:
                q = result.quantile_paths[day]
                spread = (q[0.9] - q[0.1]) / q[0.5] * 100
                print(f"  Week {week} (Day {day:>2}): ${q[0.1]:>10,.0f} - ${q[0.9]:>10,.0f}  ({spread:.0f}% spread)")
        print()

    print("=" * 70)

    # Generate chart
    show_plot = args.plot and not args.no_plot
    if show_plot or args.save_plot:
        print()
        print("Generating forecast chart...")

        from pathlib import Path
        from aetheris_oracle.visualization import plot_forecast_cone
        from datetime import timedelta

        # Fetch historical data for the chart
        try:
            frame = connector.fetch_window(
                args.asset,
                datetime.now(),
                timedelta(days=args.lookback)
            )
            historical_timestamps = frame.timestamps
            historical_prices = frame.closes
        except Exception as e:
            print(f"  Warning: Could not fetch historical data for chart: {e}")
            historical_timestamps = []
            historical_prices = []

        save_path = Path(args.save_plot) if args.save_plot else None

        chart_path = plot_forecast_cone(
            historical_timestamps=historical_timestamps,
            historical_prices=historical_prices,
            forecast_start=datetime.now(),
            quantiles_by_day=result.quantile_paths,
            asset_id=args.asset,
            save_path=save_path,
            show=show_plot,
        )

        print(f"  Chart saved to: {chart_path}")


def main() -> None:
    args = parse_args()
    if args.mode == "cli":
        cli_main()
    elif args.mode == "offline_eval":
        result = run_walk_forward([ForecastConfig(horizon_days=2, num_paths=50)])
        print("Offline evaluation summary:")
        pprint(result.coverage_summary())
    elif args.mode == "forecast":
        run_forecast(args)
    else:
        # server_main currently ignores host/port; could be extended to accept them
        server_main()


if __name__ == "__main__":
    main()
