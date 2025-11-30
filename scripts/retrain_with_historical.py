"""
Retrain all SOTA components using 5-year historical data.

Uses HistoricalParquetConnector to read from pre-collected data,
avoiding API rate limits and enabling training on full history.

Usage:
    python scripts/retrain_with_historical.py
    python scripts/retrain_with_historical.py --component fmgp --epochs 100
    python scripts/retrain_with_historical.py --skip-mamba  # Recommended
"""

import sys
sys.path.insert(0, "src")

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

# Set environment variables BEFORE importing training modules
# Use full historical data (holdout uses .env TRAINING_HOLDOUT_DAYS)
os.environ["TRAINING_LOOKBACK_DAYS"] = "1700"  # ~5 years
# TRAINING_HOLDOUT_DAYS from .env (default 30 days)
os.environ["TRAINING_SAMPLES_NCC"] = "500"
os.environ["TRAINING_SAMPLES_FMGP"] = "500"
os.environ["TRAINING_SAMPLES_NEURAL_JUMP"] = "500"
os.environ["TRAINING_SAMPLES_DIFF_GREEKS"] = "300"
os.environ["TRAINING_SAMPLES_NEURAL_VOL"] = "500"
os.environ["TRAINING_SAMPLES_MAMBA"] = "500"

from aetheris_oracle.data.historical_connector import HistoricalParquetConnector
from aetheris_oracle.pipeline.train_sota import (
    train_ncc_calibration,
    train_fmgp_residuals,
    train_neural_jump_sde,
    train_differentiable_greeks,
    train_neural_rough_vol,
    train_mamba_trend,
)


def main():
    parser = argparse.ArgumentParser(description="Retrain SOTA with historical data")
    parser.add_argument("--component", type=str, default="all",
                        choices=["all", "ncc", "fmgp", "neural_jump", "diff_greeks", "neural_vol", "mamba"],
                        help="Component to train")
    parser.add_argument("--asset", type=str, default="BTC-USD", help="Asset ID")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--data-dir", type=str, default="data/historical", help="Historical data directory")
    parser.add_argument("--artifact-dir", type=str, default="artifacts", help="Output directory")
    parser.add_argument("--skip-mamba", action="store_true", help="Skip Mamba (recommended due to bias)")

    args = parser.parse_args()

    print("=" * 70)
    print("RETRAINING SOTA COMPONENTS WITH 5-YEAR HISTORICAL DATA")
    print("=" * 70)

    # Initialize connector with historical data
    print(f"\nLoading historical data from {args.data_dir}...")
    connector = HistoricalParquetConnector(
        data_dir=args.data_dir,
        asset_id=args.asset,
    )

    start_date, end_date = connector.get_date_range()
    total_days = connector.get_total_days()
    print(f"Available data: {total_days} days ({start_date.date()} to {end_date.date()})")

    # Create artifact directory
    artifact_path = Path(args.artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)

    # Training results
    results = {}
    training_start = time.time()

    # Define components to train
    if args.component == "all":
        components = ["diff_greeks", "fmgp", "neural_vol", "neural_jump", "ncc"]
        if not args.skip_mamba:
            components.append("mamba")
        else:
            print("\nSkipping Mamba (--skip-mamba flag set)")
    else:
        components = [args.component]

    # Default epochs
    default_epochs = {
        "ncc": 30,
        "fmgp": 80,
        "neural_jump": 80,
        "diff_greeks": 50,
        "neural_vol": 50,
        "mamba": 60,
    }

    for component in components:
        epochs = args.epochs or default_epochs.get(component, 50)

        print(f"\n{'='*70}")
        print(f"Training: {component.upper()} ({epochs} epochs)")
        print(f"{'='*70}")

        comp_start = time.time()

        try:
            if component == "ncc":
                # NCC needs special handling - train on base forecasts
                metrics = train_ncc_calibration(
                    connector=connector,
                    asset_id=args.asset,
                    epochs=epochs,
                    device=args.device,
                    artifact_path=str(artifact_path / "ncc_calibration.pt"),
                )
            elif component == "fmgp":
                metrics = train_fmgp_residuals(
                    connector=connector,
                    asset_id=args.asset,
                    epochs=epochs,
                    device=args.device,
                    artifact_path=str(artifact_path / "fmgp_residuals.pt"),
                )
            elif component == "neural_jump":
                metrics = train_neural_jump_sde(
                    connector=connector,
                    asset_id=args.asset,
                    epochs=epochs,
                    device=args.device,
                    artifact_path=str(artifact_path / "neural_jump_sde.pt"),
                )
            elif component == "diff_greeks":
                metrics = train_differentiable_greeks(
                    connector=connector,
                    asset_id=args.asset,
                    epochs=epochs,
                    device=args.device,
                    artifact_path=str(artifact_path / "diff_greeks.pt"),
                )
            elif component == "neural_vol":
                metrics = train_neural_rough_vol(
                    connector=connector,
                    asset_id=args.asset,
                    epochs=epochs,
                    device=args.device,
                    artifact_path=str(artifact_path / "neural_rough_vol.pt"),
                )
            elif component == "mamba":
                metrics = train_mamba_trend(
                    connector=connector,
                    asset_id=args.asset,
                    epochs=epochs,
                    device=args.device,
                    artifact_path=str(artifact_path / "mamba_trend.pt"),
                )

            comp_time = time.time() - comp_start
            results[component] = {
                "status": "success",
                "epochs": epochs,
                "time_seconds": comp_time,
                "metrics": metrics if isinstance(metrics, dict) else {"result": str(metrics)},
            }
            print(f"\n{component.upper()}: Completed in {comp_time:.1f}s")

        except Exception as e:
            comp_time = time.time() - comp_start
            results[component] = {
                "status": "failed",
                "error": str(e),
                "time_seconds": comp_time,
            }
            print(f"\n{component.upper()}: FAILED - {e}")

    # Summary
    total_time = time.time() - training_start

    print(f"\n{'='*70}")
    print("TRAINING SUMMARY")
    print(f"{'='*70}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Data used: {total_days} days of {args.asset}")
    print()

    for comp, result in results.items():
        status = "OK" if result["status"] == "success" else "FAILED"
        time_s = result["time_seconds"]
        print(f"  {comp:<15} [{status}] {time_s:.1f}s")

    # Save results
    results_file = artifact_path / f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "asset": args.asset,
            "data_days": total_days,
            "total_time_seconds": total_time,
            "results": results,
        }, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")
    print(f"Models saved to: {artifact_path}/")


if __name__ == "__main__":
    main()
