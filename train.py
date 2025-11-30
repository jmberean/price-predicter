"""
Simple training script - reads all config from .env file.

Usage:
    python train.py              # Train all SOTA components
    python train.py --quick      # Quick training (fewer samples/epochs)
    python train.py -w 4         # Parallel training with 4 workers

Config is read from .env - see .env.example for all options.
"""

import os
import sys
sys.path.insert(0, "src")

from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
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


def get_env_float(key: str, default: float) -> float:
    val = os.getenv(key, "")
    try:
        return float(val) if val else default
    except ValueError:
        return default


def get_tuned_hyperparams() -> dict:
    """Read tuned hyperparameters from .env (set by hyperparameter_tuning.py)."""
    return {
        "ncc": {
            "learning_rate": get_env_float("TUNING_NCC_LR", 0.001),
            "epochs": get_env_int("TUNING_NCC_EPOCHS", 30),
            "hidden_dim": get_env_int("TUNING_NCC_HIDDEN_DIM", 64),
            "smoothness_weight": get_env_float("TUNING_NCC_SMOOTHNESS", 0.1),
        },
        "fmgp": {
            "learning_rate": get_env_float("TUNING_FMGP_LR", 0.001),
            "epochs": get_env_int("TUNING_FMGP_EPOCHS", 80),
            "time_embed_dim": get_env_int("TUNING_FMGP_TIME_EMBED", 64),
            "cfg_dropout": get_env_float("TUNING_FMGP_DROPOUT", 0.1),
        },
        "neural_jump": {
            "learning_rate": get_env_float("TUNING_JUMP_LR", 0.001),
            "epochs": get_env_int("TUNING_JUMP_EPOCHS", 80),
            "hidden_dim": get_env_int("TUNING_JUMP_HIDDEN_DIM", 128),
            "n_layers": get_env_int("TUNING_JUMP_LAYERS", 3),
        },
        "diff_greeks": {
            "learning_rate": get_env_float("TUNING_GREEKS_LR", 0.001),
            "epochs": get_env_int("TUNING_GREEKS_EPOCHS", 50),
            "n_heads": get_env_int("TUNING_GREEKS_HEADS", 4),
            "embed_dim": get_env_int("TUNING_GREEKS_EMBED_DIM", 64),
        },
        "neural_vol": {
            "learning_rate": get_env_float("TUNING_VOL_LR", 0.001),
            "epochs": get_env_int("TUNING_VOL_EPOCHS", 50),
            "hidden_dim": get_env_int("TUNING_VOL_HIDDEN_DIM", 128),
            "hurst": get_env_float("TUNING_VOL_HURST", 0.1),
        },
    }


def update_module_horizons(horizon: int) -> None:
    """Update horizon in SOTA module config files."""
    import re

    files_to_update = [
        ("src/aetheris_oracle/modules/fm_gp_residual.py", "horizon: int = "),
        ("src/aetheris_oracle/modules/neural_rough_vol.py", "horizon: int = "),
        ("src/aetheris_oracle/modules/mamba_trend.py", "horizon: int = "),
    ]

    for filepath, pattern in files_to_update:
        if not Path(filepath).exists():
            continue

        with open(filepath, "r") as f:
            content = f.read()

        # Replace horizon value in dataclass
        new_content = re.sub(
            rf"({pattern})\d+",
            rf"\g<1>{horizon}",
            content
        )

        if new_content != content:
            with open(filepath, "w") as f:
                f.write(new_content)
            print(f"  Updated horizon to {horizon} in {filepath}")


DEFAULT_WORKERS = max(1, multiprocessing.cpu_count() // 2)


def _train_single_component(
    component: str,
    epochs: int,
    asset: str,
    device: str,
    data_dir: str,
    artifact_dir: str,
) -> dict:
    """Worker function to train a single component."""
    # Re-import in worker process
    import sys
    sys.path.insert(0, "src")
    from dotenv import load_dotenv
    load_dotenv()

    from pathlib import Path
    import time

    from aetheris_oracle.data.historical_connector import HistoricalParquetConnector
    from aetheris_oracle.pipeline.train_sota import (
        train_ncc_calibration,
        train_fmgp_residuals,
        train_neural_jump_sde,
        train_differentiable_greeks,
        train_neural_rough_vol,
        train_mamba_trend,
    )

    train_funcs = {
        "ncc": (train_ncc_calibration, "ncc_calibration.pt"),
        "fmgp": (train_fmgp_residuals, "fmgp_residuals.pt"),
        "neural_jump": (train_neural_jump_sde, "neural_jump_sde.pt"),
        "diff_greeks": (train_differentiable_greeks, "diff_greeks.pt"),
        "neural_vol": (train_neural_rough_vol, "neural_rough_vol.pt"),
        "mamba": (train_mamba_trend, "mamba_trend.pt"),
    }

    try:
        connector = HistoricalParquetConnector(data_dir=data_dir, asset_id=asset)
        train_func, artifact_name = train_funcs[component]
        artifact_path = Path(artifact_dir)

        comp_start = time.time()
        metrics = train_func(
            connector=connector,
            asset_id=asset,
            epochs=epochs,
            device=device,
            artifact_path=str(artifact_path / artifact_name),
        )
        comp_time = time.time() - comp_start
        return {"component": component, "status": "success", "time": comp_time}
    except Exception as e:
        return {"component": component, "status": "failed", "error": str(e), "time": 0}


def main():
    parser = argparse.ArgumentParser(description="Train SOTA models (config from .env)")
    parser.add_argument("--quick", action="store_true", help="Quick training (fewer samples/epochs)")
    parser.add_argument("--component", type=str, default="all",
                        choices=["all", "ncc", "fmgp", "neural_jump", "diff_greeks", "neural_vol", "mamba"],
                        help="Component to train (default: all)")
    parser.add_argument("-w", "--workers", type=int, default=1,
                        help=f"Number of parallel workers (default: 1, max recommended: {DEFAULT_WORKERS})")
    args = parser.parse_args()

    # Read config from .env
    asset = get_env("FORECAST_ASSET", "BTC-USD")
    horizon = get_env_int("TRAINING_HORIZON", 90)
    device = get_env("TORCH_DEVICE", "cpu")
    data_dir = get_env("TRAINING_DATA_DIR", "data/historical")
    artifact_dir = get_env("ARTIFACT_ROOT", "artifacts")
    skip_mamba = not get_env_bool("USE_MAMBA_TREND", False)

    # Quick mode overrides
    if args.quick:
        os.environ["TRAINING_LOOKBACK_DAYS"] = "90"
        os.environ["TRAINING_SAMPLES_NCC"] = "100"
        os.environ["TRAINING_SAMPLES_FMGP"] = "100"
        os.environ["TRAINING_SAMPLES_NEURAL_JUMP"] = "100"
        os.environ["TRAINING_SAMPLES_DIFF_GREEKS"] = "100"
        os.environ["TRAINING_SAMPLES_NEURAL_VOL"] = "100"
        os.environ["TRAINING_SAMPLES_MAMBA"] = "100"
        quick_epochs = {"ncc": 10, "fmgp": 20, "neural_jump": 20, "diff_greeks": 15, "neural_vol": 15, "mamba": 15}
    else:
        quick_epochs = None

    # Read tuned hyperparameters from .env
    tuned = get_tuned_hyperparams()

    print("=" * 70)
    print("AETHERIS ORACLE - SOTA MODEL TRAINING")
    print("=" * 70)
    print(f"  Asset:    {asset}")
    print(f"  Horizon:  {horizon} days")
    print(f"  Device:   {device}")
    print(f"  Mode:     {'Quick' if args.quick else 'Full'}")
    print(f"  Workers:  {args.workers} {'(parallel)' if args.workers > 1 else '(sequential)'}")
    print(f"  Mamba:    {'Skipped (disabled in .env)' if skip_mamba else 'Enabled'}")

    # Check if tuned hyperparameters exist
    has_tuned = os.getenv("TUNING_NCC_LR") is not None
    if has_tuned:
        print(f"  Tuned:    Using hyperparameters from .env")
    else:
        print(f"  Tuned:    Using defaults (run hyperparameter_tuning.py for better results)")
    print("=" * 70)

    # Update module horizons
    print("\nUpdating model horizons...")
    update_module_horizons(horizon)

    # Import training functions (after horizon update)
    from aetheris_oracle.data.historical_connector import HistoricalParquetConnector
    from aetheris_oracle.pipeline.train_sota import (
        train_ncc_calibration,
        train_fmgp_residuals,
        train_neural_jump_sde,
        train_differentiable_greeks,
        train_neural_rough_vol,
        train_mamba_trend,
    )

    # Initialize connector
    print(f"\nLoading historical data from {data_dir}...")
    try:
        connector = HistoricalParquetConnector(data_dir=data_dir, asset_id=asset)
        start_date, end_date = connector.get_date_range()
        total_days = connector.get_total_days()
        print(f"Available: {total_days} days ({start_date.date()} to {end_date.date()})")
    except Exception as e:
        print(f"ERROR: Could not load historical data: {e}")
        print(f"Run 'python scripts/collect_historical_data.py --asset {asset}' first")
        sys.exit(1)

    # Create artifact directory
    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)

    # Define components
    if args.component == "all":
        components = ["diff_greeks", "fmgp", "neural_vol", "neural_jump", "ncc"]
        if not skip_mamba:
            components.append("mamba")
    else:
        components = [args.component]

    # Get epochs from tuned params or defaults
    def get_epochs(comp):
        if args.quick:
            quick_map = {"ncc": 10, "fmgp": 20, "neural_jump": 20, "diff_greeks": 15, "neural_vol": 15, "mamba": 15}
            return quick_map.get(comp, 15)
        # Use tuned epochs if available
        return tuned.get(comp, {}).get("epochs", {"ncc": 30, "fmgp": 80, "neural_jump": 80, "diff_greeks": 50, "neural_vol": 50, "mamba": 60}.get(comp, 50))

    # Train functions mapping
    train_funcs = {
        "ncc": (train_ncc_calibration, "ncc_calibration.pt"),
        "fmgp": (train_fmgp_residuals, "fmgp_residuals.pt"),
        "neural_jump": (train_neural_jump_sde, "neural_jump_sde.pt"),
        "diff_greeks": (train_differentiable_greeks, "diff_greeks.pt"),
        "neural_vol": (train_neural_rough_vol, "neural_rough_vol.pt"),
        "mamba": (train_mamba_trend, "mamba_trend.pt"),
    }

    results = {}
    training_start = time.time()

    # Parallel training support
    n_workers = args.workers
    if n_workers > 1 and args.component == "all":
        print(f"\nParallel training enabled: {n_workers} workers")

        # NCC must run last (depends on other models), so train base models in parallel first
        base_components = [c for c in components if c != "ncc"]
        ncc_component = "ncc" if "ncc" in components else None

        # Train base components in parallel
        if base_components:
            print(f"\n{'='*70}")
            print(f"Training {len(base_components)} base components in parallel...")
            print(f"{'='*70}")

            with ProcessPoolExecutor(max_workers=min(n_workers, len(base_components))) as executor:
                futures = {}
                for comp in base_components:
                    epochs = get_epochs(comp)
                    future = executor.submit(
                        _train_single_component,
                        comp, epochs, asset, device, data_dir, artifact_dir
                    )
                    futures[future] = comp
                    print(f"  Submitted: {comp.upper()} ({epochs} epochs)")

                for future in as_completed(futures):
                    result = future.result()
                    comp = result["component"]
                    if result["status"] == "success":
                        results[comp] = {"status": "success", "time": result["time"]}
                        print(f"  Completed: {comp.upper()} ({result['time']:.1f}s)")
                    else:
                        results[comp] = {"status": "failed", "error": result.get("error", "unknown"), "time": 0}
                        print(f"  Failed: {comp.upper()} - {result.get('error', 'unknown')}")

        # Train NCC last (needs other models to exist)
        if ncc_component:
            epochs = get_epochs("ncc")
            print(f"\n{'='*70}")
            print(f"Training: NCC ({epochs} epochs) - runs last, needs other models")
            print(f"{'='*70}")

            comp_start = time.time()
            try:
                train_func, artifact_name = train_funcs["ncc"]
                metrics = train_func(
                    connector=connector,
                    asset_id=asset,
                    epochs=epochs,
                    device=device,
                    artifact_path=str(artifact_path / artifact_name),
                )
                comp_time = time.time() - comp_start
                results["ncc"] = {"status": "success", "time": comp_time}
                print(f"\nNCC: OK ({comp_time:.1f}s)")
            except Exception as e:
                comp_time = time.time() - comp_start
                results["ncc"] = {"status": "failed", "error": str(e), "time": comp_time}
                print(f"\nNCC: FAILED - {e}")
    else:
        # Sequential training (original behavior)
        for component in components:
            epochs = get_epochs(component)
            train_func, artifact_name = train_funcs[component]
            comp_tuned = tuned.get(component, {})

            print(f"\n{'='*70}")
            print(f"Training: {component.upper()} ({epochs} epochs)")
            if comp_tuned and not args.quick:
                print(f"  Hyperparams: lr={comp_tuned.get('learning_rate', 'default')}, hidden={comp_tuned.get('hidden_dim', comp_tuned.get('embed_dim', comp_tuned.get('time_embed_dim', 'default')))}")
            print(f"{'='*70}")

            comp_start = time.time()
            try:
                metrics = train_func(
                    connector=connector,
                    asset_id=asset,
                    epochs=epochs,
                    device=device,
                    artifact_path=str(artifact_path / artifact_name),
                )
                comp_time = time.time() - comp_start
                results[component] = {"status": "success", "time": comp_time}
                print(f"\n{component.upper()}: OK ({comp_time:.1f}s)")
            except Exception as e:
                comp_time = time.time() - comp_start
                results[component] = {"status": "failed", "error": str(e), "time": comp_time}
                print(f"\n{component.upper()}: FAILED - {e}")

    # Summary
    total_time = time.time() - training_start
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print()
    for comp, result in results.items():
        status = "OK" if result["status"] == "success" else "FAILED"
        print(f"  {comp:<15} [{status}] {result['time']:.1f}s")
    print(f"\nModels saved to: {artifact_path}/")


if __name__ == "__main__":
    main()
