import argparse
import os
from pprint import pprint
from typing import Dict

from dotenv import load_dotenv

from .config import ForecastConfig, ScenarioOverrides
from .data.csv_connector import CsvDataConnector
from .data.free_connectors import FreeDataConnector
from .pipeline.calibration import CalibrationEngine
from .pipeline.forecast import ForecastEngine
from .pipeline.train import TrainConfig, run_training

# Load environment variables from .env file
load_dotenv()


def _parse_narratives(pairs: list[str]) -> Dict[str, float]:
    parsed: Dict[str, float] = {}
    for p in pairs:
        if "=" not in p:
            continue
        k, v = p.split("=", maxsplit=1)
        try:
            parsed[k] = float(v)
        except ValueError:
            continue
    return parsed


def main() -> None:
    forecast_parent = argparse.ArgumentParser(add_help=False)
    forecast_parent.add_argument("--asset", default="BTC-USD", help="Asset identifier")
    forecast_parent.add_argument("--horizon", type=int, default=7, help="Forecast horizon in days")
    forecast_parent.add_argument("--paths", type=int, default=1000, help="Number of simulated paths")
    forecast_parent.add_argument("--thresholds", nargs="*", type=float, default=[20000.0, 50000.0])
    forecast_parent.add_argument("--seed", type=int, default=123, help="Random seed")
    forecast_parent.add_argument("--csv-path", help="Optional CSV file for input features")
    forecast_parent.add_argument(
        "--connector",
        choices=["synthetic", "csv", "free"],
        default="synthetic",
        help="Connector to use when csv-path not provided",
    )
    forecast_parent.add_argument("--iv-multiplier", type=float, default=1.0)
    forecast_parent.add_argument("--funding-shift", type=float, default=0.0)
    forecast_parent.add_argument("--basis-shift", type=float, default=0.0)
    forecast_parent.add_argument(
        "--narrative",
        action="append",
        default=[],
        help="Narrative override pairs key=value (repeatable)",
    )
    forecast_parent.add_argument(
        "--calibration-path",
        help="Optional path to load/save calibration state JSON",
    )
    forecast_parent.add_argument(
        "--realized-price",
        type=float,
        help="Optional realized price to update calibration post-forecast (uses final horizon)",
    )
    forecast_parent.add_argument(
        "--plot",
        action="store_true",
        help="Generate and display forecast visualization chart",
    )
    forecast_parent.add_argument(
        "--plot-save",
        help="Path to save chart image (default: auto-generated filename)",
    )
    forecast_parent.add_argument(
        "--plot-lookback",
        type=int,
        default=30,
        help="Days of historical data to show in chart (default: 30)",
    )

    # SOTA model flags
    forecast_parent.add_argument(
        "--use-ncc",
        action="store_true",
        help="Use Neural Conformal Control calibration (trained model)",
    )
    forecast_parent.add_argument(
        "--use-diff-greeks",
        action="store_true",
        help="Use Differentiable Greeks for market maker state",
    )
    forecast_parent.add_argument(
        "--ncc-path",
        default="artifacts/ncc_calibration.pt",
        help="Path to NCC model artifact (default: artifacts/ncc_calibration.pt)",
    )
    forecast_parent.add_argument(
        "--diff-greeks-path",
        default="artifacts/diff_greeks.pt",
        help="Path to Diff Greeks model artifact (default: artifacts/diff_greeks.pt)",
    )

    parser = argparse.ArgumentParser(
        description="Run Aetheris Oracle forecast or training.",
        parents=[forecast_parent],
    )
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Run offline training on synthetic data")
    train_parser.add_argument("--asset", default="BTC-USD")
    train_parser.add_argument("--horizon", type=int, default=7)
    train_parser.add_argument("--samples", type=int, default=24)
    train_parser.add_argument("--trailing-window", type=int, default=90)
    train_parser.add_argument("--artifact-root", default="artifacts")
    train_parser.add_argument("--seed", type=int, default=123)

    args = parser.parse_args()

    if args.command == "train":
        train_cfg = TrainConfig(
            asset_id=args.asset,
            horizon_days=args.horizon,
            trailing_window_days=args.trailing_window,
            num_samples=args.samples,
            artifact_root=args.artifact_root,
            seed=args.seed,
        )
        artifacts = run_training(train_cfg)
        print("Training complete. Saved artifacts to", args.artifact_root)
        print("Trend weights version:", artifacts.trend.version)
        return

    # Default to forecast path when no subcommand provided
    scenario = None
    narratives = _parse_narratives(args.narrative if hasattr(args, "narrative") else [])
    if getattr(args, "iv_multiplier", 1.0) != 1.0 or getattr(args, "funding_shift", 0.0) or getattr(args, "basis_shift", 0.0) or narratives:
        scenario = ScenarioOverrides(
            iv_multiplier=args.iv_multiplier,
            funding_shift=args.funding_shift,
            basis_shift=args.basis_shift,
            narrative_overrides=narratives,
            description="cli scenario",
        )

    cfg = ForecastConfig(
        asset_id=args.asset,
        horizon_days=args.horizon,
        num_paths=args.paths,
        thresholds=tuple(args.thresholds),
        seed=args.seed,
        scenario=scenario,
    )
    connector = None
    if getattr(args, "csv_path", None):
        connector = CsvDataConnector(args.csv_path)
    elif getattr(args, "connector", "synthetic") == "free":
        connector = FreeDataConnector()

    calibration = None
    if getattr(args, "calibration_path", None):
        try:
            calibration = CalibrationEngine.load(args.calibration_path)
        except FileNotFoundError:
            calibration = CalibrationEngine()

    # Create engine with optional SOTA models
    engine_kwargs = {
        "seed": args.seed,
        "connector": connector,
        "calibration": calibration,
    }

    if getattr(args, "use_ncc", False):
        engine_kwargs["use_ncc_calibration"] = True
        engine_kwargs["ncc_artifact_path"] = args.ncc_path

    if getattr(args, "use_diff_greeks", False):
        engine_kwargs["use_diff_greeks"] = True
        engine_kwargs["diff_greeks_artifact_path"] = args.diff_greeks_path

    engine = ForecastEngine(**engine_kwargs)
    result = engine.forecast(cfg)

    if getattr(args, "calibration_path", None):
        if getattr(args, "realized_price", None) is not None:
            engine.update_calibration_with_realized(result, args.realized_price)
        engine.calibration.save(args.calibration_path)

    print(f"Forecast cone for {cfg.asset_id} (as of {cfg.as_of.isoformat()})")
    print("Quantiles by day:")
    for t, qs in result.quantile_paths.items():
        print(f"  +{t}d:", {round(k, 2): round(v, 2) for k, v in qs.items()})
    print("Threshold probabilities (terminal):")
    pprint(result.threshold_probabilities)
    print("Top drivers:")
    pprint(result.drivers)

    # Generate visualization if requested
    if getattr(args, "plot", False):
        from .visualization import quick_plot_from_cli_result

        save_path = getattr(args, "plot_save", None)
        lookback = getattr(args, "plot_lookback", 30)

        print(f"\nGenerating forecast visualization...")
        try:
            chart_path = quick_plot_from_cli_result(
                result_dict=result.__dict__,
                asset_id=cfg.asset_id,
                lookback_days=lookback,
                save_path=save_path,
                show=True,
            )
            print(f"✅ Chart saved to: {chart_path}")
        except Exception as e:
            print(f"⚠️  Failed to generate chart: {e}")


if __name__ == "__main__":
    main()
