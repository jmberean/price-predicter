"""
Hyperparameter tuning for SOTA neural components.

Supports:
- Grid search for systematic exploration
- Random search for efficient sampling
- Results tracking and best model selection

Usage:
    python scripts/hyperparameter_tuning.py --component ncc --method grid
    python scripts/hyperparameter_tuning.py --component fmgp --method random --trials 20
    python scripts/hyperparameter_tuning.py --component all --method grid --quick
"""

import sys
sys.path.insert(0, "src")

import argparse
import json
import itertools
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from aetheris_oracle.data.free_connectors import FreeDataConnector
from aetheris_oracle.data.connectors import SyntheticDataConnector
from aetheris_oracle.config import ForecastConfig
from aetheris_oracle.pipeline.forecast import ForecastEngine


# Hyperparameter search spaces for each component
# Based on actual config fields from each module
SEARCH_SPACES = {
    "ncc": {
        "learning_rate": [0.0001, 0.0005, 0.001, 0.005],
        "epochs": [10, 20, 30],
        "hidden_dim": [32, 64, 128],
        "smoothness_weight": [0.05, 0.1, 0.2],
    },
    "fmgp": {
        "learning_rate": [0.0001, 0.0005, 0.001],
        "epochs": [30, 50, 70],
        "time_embed_dim": [32, 64, 128],
        "cfg_dropout": [0.0, 0.1, 0.2],
    },
    "neural_jump": {
        "learning_rate": [0.0001, 0.0005, 0.001],
        "epochs": [30, 50, 70],
        "hidden_dim": [64, 128, 256],
        "n_layers": [2, 3, 4],
    },
    "diff_greeks": {
        "learning_rate": [0.0001, 0.0005, 0.001],
        "epochs": [20, 30, 40],
        "n_heads": [2, 4, 8],
        "embed_dim": [32, 64, 128],
    },
    "neural_vol": {
        "learning_rate": [0.0001, 0.0005, 0.001],
        "epochs": [20, 30, 40],
        "hidden_dim": [64, 128, 256],
        "hurst": [0.05, 0.1, 0.15],
    },
}

# Quick search spaces (fewer options for faster iteration)
QUICK_SEARCH_SPACES = {
    "ncc": {
        "learning_rate": [0.0005, 0.001],
        "epochs": [10, 20],
        "hidden_dim": [64],
        "smoothness_weight": [0.1],
    },
    "fmgp": {
        "learning_rate": [0.0005, 0.001],
        "epochs": [30, 50],
        "time_embed_dim": [64],
        "cfg_dropout": [0.1],
    },
    "neural_jump": {
        "learning_rate": [0.0005, 0.001],
        "epochs": [30, 50],
        "hidden_dim": [128],
        "n_layers": [3],
    },
    "diff_greeks": {
        "learning_rate": [0.0005, 0.001],
        "epochs": [20, 30],
        "n_heads": [4],
        "embed_dim": [64],
    },
    "neural_vol": {
        "learning_rate": [0.0005, 0.001],
        "epochs": [20, 30],
        "hidden_dim": [128],
        "hurst": [0.1],
    },
}


def grid_search_configs(search_space: Dict[str, List]) -> List[Dict]:
    """Generate all combinations from search space."""
    keys = list(search_space.keys())
    values = list(search_space.values())
    configs = []
    for combo in itertools.product(*values):
        configs.append(dict(zip(keys, combo)))
    return configs


def random_search_configs(search_space: Dict[str, List], n_trials: int) -> List[Dict]:
    """Generate random configurations from search space."""
    configs = []
    for _ in range(n_trials):
        config = {}
        for key, values in search_space.items():
            config[key] = np.random.choice(values)
        configs.append(config)
    return configs


def train_and_evaluate_ncc(
    connector,
    config: Dict,
    asset_id: str = "BTC-USD",
    device: str = "cpu",
) -> Tuple[float, Dict]:
    """Train NCC with given hyperparameters and evaluate."""
    from aetheris_oracle.pipeline.neural_conformal_control import NCCCalibrationEngine, NCCConfig
    from aetheris_oracle.pipeline.training_data_prep import prepare_ncc_training_data

    # Create config with hyperparameters (only use valid NCCConfig fields)
    ncc_config = NCCConfig(
        hidden_dim=config.get("hidden_dim", 64),
        learning_rate=config.get("learning_rate", 0.001),
        smoothness_weight=config.get("smoothness_weight", 0.1),
    )

    engine = NCCCalibrationEngine(config=ncc_config, device=device)

    # Prepare training data (smaller sample for tuning)
    horizons = [1, 3, 7, 14]
    base_quantiles_list, actuals_list, features_list, horizon_indices_list = prepare_ncc_training_data(
        connector=connector,
        asset_id=asset_id,
        n_samples=30,  # Smaller for faster tuning
        horizons=horizons,
    )

    if len(base_quantiles_list) == 0:
        return float("inf"), {"error": "No training data"}

    # Convert data
    quantile_keys = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    base_quantiles_dicts = [
        {q: float(base_quantiles_list[i][j]) for j, q in enumerate(quantile_keys)}
        for i in range(len(base_quantiles_list))
    ]
    features_dicts = [
        {f"f{j}": float(features_list[i][j]) for j in range(len(features_list[i]))}
        for i in range(len(features_list))
    ]
    horizons_list = [horizons[idx] for idx in horizon_indices_list]

    # Train
    metrics = engine.train_on_historical(
        base_quantiles=base_quantiles_dicts,
        features_list=features_dicts,
        actuals=actuals_list,
        horizons=horizons_list,
        epochs=config.get("epochs", 20),
        batch_size=16,
    )

    # Evaluate: use final loss as score (lower is better)
    final_loss = metrics.get("loss", [1.0])[-1] if isinstance(metrics.get("loss"), list) else 1.0

    return final_loss, metrics


def train_and_evaluate_fmgp(
    connector,
    config: Dict,
    asset_id: str = "BTC-USD",
    device: str = "cpu",
) -> Tuple[float, Dict]:
    """Train FM-GP with given hyperparameters and evaluate."""
    from aetheris_oracle.modules.fm_gp_residual import FMGPResidualEngine, FMGPConfig
    from aetheris_oracle.pipeline.training_data_prep import prepare_fmgp_residual_training_data

    fmgp_config = FMGPConfig(
        time_embed_dim=config.get("time_embed_dim", 64),
        cfg_dropout=config.get("cfg_dropout", 0.1),
        learning_rate=config.get("learning_rate", 0.001),
    )

    engine = FMGPResidualEngine(config=fmgp_config, device=device)

    # Prepare training data
    conditioning_list, residual_paths_list = prepare_fmgp_residual_training_data(
        connector=connector,
        asset_id=asset_id,
        n_samples=30,
    )

    if len(residual_paths_list) == 0:
        return float("inf"), {"error": "No training data"}

    # Train
    metrics = engine.train_on_historical(
        residual_sequences=residual_paths_list,
        conditioning_sequences=conditioning_list,
        epochs=config.get("epochs", 50),
        batch_size=16,
    )

    final_loss = metrics.get("loss", [1.0])[-1] if isinstance(metrics.get("loss"), list) else 1.0
    return final_loss, metrics


def train_and_evaluate_neural_jump(
    connector,
    config: Dict,
    asset_id: str = "BTC-USD",
    device: str = "cpu",
) -> Tuple[float, Dict]:
    """Train Neural Jump SDE with given hyperparameters and evaluate."""
    from aetheris_oracle.modules.neural_jump_sde import NeuralJumpSDEEngine, NeuralJumpSDEConfig
    from aetheris_oracle.pipeline.training_data_prep import prepare_jump_sde_training_data

    jump_config = NeuralJumpSDEConfig(
        hidden_dim=config.get("hidden_dim", 128),
        n_layers=config.get("n_layers", 3),
        learning_rate=config.get("learning_rate", 0.001),
    )

    engine = NeuralJumpSDEEngine(config=jump_config, device=device)

    # Prepare training data
    x0_list, conditioning_list, target_paths_list = prepare_jump_sde_training_data(
        connector=connector,
        asset_id=asset_id,
        n_samples=30,
        horizon=14,
    )

    if len(target_paths_list) == 0:
        return float("inf"), {"error": "No training data"}

    # Train
    metrics = engine.train_on_historical(
        paths=target_paths_list,
        conditioning_sequences=conditioning_list,
        epochs=config.get("epochs", 50),
        batch_size=16,
    )

    final_loss = metrics.get("loss", [1.0])[-1] if isinstance(metrics.get("loss"), list) else 1.0
    return final_loss, metrics


def train_and_evaluate_diff_greeks(
    connector,
    config: Dict,
    asset_id: str = "BTC-USD",
    device: str = "cpu",
) -> Tuple[float, Dict]:
    """Train Differentiable Greeks with given hyperparameters and evaluate."""
    import torch
    from aetheris_oracle.modules.differentiable_greeks import DifferentiableMMEngine, DifferentiableGreeksConfig

    greeks_config = DifferentiableGreeksConfig(
        n_heads=config.get("n_heads", 4),
        embed_dim=config.get("embed_dim", 64),
        learning_rate=config.get("learning_rate", 0.001),
    )

    engine = DifferentiableMMEngine(greeks_config).to(device)
    engine.optimizer = torch.optim.AdamW(
        engine.parameters(),
        lr=config.get("learning_rate", 0.001)
    )

    # Train on synthetic data
    epochs = config.get("epochs", 30)
    batch_size = 16
    n_samples = 80

    epoch_losses = []
    for epoch in range(epochs):
        batch_losses = []
        for _ in range(n_samples // batch_size):
            spot = torch.rand(batch_size, device=device) * 50000 + 30000
            strikes = spot.unsqueeze(1) * torch.linspace(0.8, 1.2, greeks_config.n_strikes, device=device).unsqueeze(0)
            taus = torch.rand(batch_size, greeks_config.n_strikes, device=device) * 30 / 365
            ivs = torch.rand(batch_size, greeks_config.n_strikes, device=device) * 0.5 + 0.3
            ois = torch.rand(batch_size, greeks_config.n_strikes, device=device)
            funding = torch.randn(batch_size, device=device) * 0.01
            basis = torch.randn(batch_size, device=device) * 100
            skew = torch.randn(batch_size, device=device) * 0.1
            imbalance = torch.randn(batch_size, device=device) * 0.2
            target_returns = torch.randn(batch_size, device=device) * 0.02

            metrics = engine.train_step(
                spot, strikes, taus, ivs, ois, funding, basis, skew, imbalance, target_returns
            )
            batch_losses.append(metrics["loss"])

        epoch_losses.append(np.mean(batch_losses))

    final_loss = epoch_losses[-1]
    return final_loss, {"loss": epoch_losses}


def train_and_evaluate_neural_vol(
    connector,
    config: Dict,
    asset_id: str = "BTC-USD",
    device: str = "cpu",
) -> Tuple[float, Dict]:
    """Train Neural Rough Vol with given hyperparameters and evaluate."""
    from aetheris_oracle.modules.neural_rough_vol import NeuralRoughVolWrapper, NeuralRoughVolConfig
    from aetheris_oracle.pipeline.training_data_prep import prepare_neural_vol_training_data

    vol_config = NeuralRoughVolConfig(
        hidden_dim=config.get("hidden_dim", 128),
        hurst=config.get("hurst", 0.1),
        learning_rate=config.get("learning_rate", 0.001),
    )

    wrapper = NeuralRoughVolWrapper(config=vol_config, device=device)

    # Prepare training data
    past_vols, conditioning_list, target_vol_paths = prepare_neural_vol_training_data(
        connector=connector,
        asset_id=asset_id,
        n_samples=30,
    )

    if len(target_vol_paths) == 0:
        return float("inf"), {"error": "No training data"}

    # Train
    metrics = wrapper.train_on_historical(
        past_vols=past_vols,
        conditioning_sequences=conditioning_list,
        target_vol_paths=target_vol_paths,
        epochs=config.get("epochs", 30),
        batch_size=16,
    )

    final_loss = metrics.get("loss", [1.0])[-1] if isinstance(metrics.get("loss"), list) else 1.0
    return final_loss, metrics


TRAIN_FUNCTIONS = {
    "ncc": train_and_evaluate_ncc,
    "fmgp": train_and_evaluate_fmgp,
    "neural_jump": train_and_evaluate_neural_jump,
    "diff_greeks": train_and_evaluate_diff_greeks,
    "neural_vol": train_and_evaluate_neural_vol,
}


def run_hyperparameter_search(
    component: str,
    method: str = "grid",
    n_trials: int = 10,
    quick: bool = False,
    asset_id: str = "BTC-USD",
    device: str = "cpu",
    output_dir: str = "artifacts/tuning",
) -> Dict[str, Any]:
    """
    Run hyperparameter search for a component.

    Args:
        component: Component to tune (ncc, fmgp, etc.)
        method: Search method (grid, random)
        n_trials: Number of trials for random search
        quick: Use quick search space
        asset_id: Asset to train on
        device: Device (cpu/cuda)
        output_dir: Directory to save results

    Returns:
        Results dictionary with best config and all trials
    """
    print(f"\n{'='*60}")
    print(f"Hyperparameter Tuning: {component.upper()}")
    print(f"Method: {method}, Quick: {quick}")
    print(f"{'='*60}")

    # Get search space
    search_spaces = QUICK_SEARCH_SPACES if quick else SEARCH_SPACES
    if component not in search_spaces:
        raise ValueError(f"Unknown component: {component}")

    search_space = search_spaces[component]

    # Generate configs
    if method == "grid":
        configs = grid_search_configs(search_space)
    else:
        configs = random_search_configs(search_space, n_trials)

    print(f"Testing {len(configs)} configurations...")

    # Initialize connector
    connector = FreeDataConnector(enable_cache=True, cache_ttl_seconds=3600)

    # Get training function
    train_fn = TRAIN_FUNCTIONS[component]

    # Run trials
    results = []
    best_loss = float("inf")
    best_config = None

    for i, config in enumerate(configs):
        print(f"\nTrial {i+1}/{len(configs)}: {config}")

        try:
            start_time = time.time()
            loss, metrics = train_fn(connector, config, asset_id, device)
            duration = time.time() - start_time

            trial_result = {
                "trial": i + 1,
                "config": config,
                "loss": float(loss),
                "duration_seconds": duration,
                "success": True,
            }

            print(f"  Loss: {loss:.6f}, Time: {duration:.1f}s")

            if loss < best_loss:
                best_loss = loss
                best_config = config.copy()
                print(f"  ** New best! **")

        except Exception as e:
            trial_result = {
                "trial": i + 1,
                "config": config,
                "loss": float("inf"),
                "error": str(e),
                "success": False,
            }
            print(f"  Error: {e}")

        results.append(trial_result)

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f"tuning_{component}_{timestamp}.json"

    summary = {
        "component": component,
        "method": method,
        "quick": quick,
        "n_trials": len(configs),
        "best_config": best_config,
        "best_loss": best_loss,
        "timestamp": timestamp,
        "trials": results,
    }

    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"BEST CONFIG for {component.upper()}:")
    print(f"{'='*60}")
    if best_config is not None:
        print(f"  Loss: {best_loss:.6f}")
        for k, v in best_config.items():
            print(f"  {k}: {v}")
    else:
        print(f"  No successful trials - all configurations failed")
    print(f"\nResults saved to: {results_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for SOTA components")
    parser.add_argument("--component", type=str, default="ncc",
                        choices=["ncc", "fmgp", "neural_jump", "diff_greeks", "neural_vol", "all"],
                        help="Component to tune")
    parser.add_argument("--method", type=str, default="grid",
                        choices=["grid", "random"],
                        help="Search method")
    parser.add_argument("--trials", type=int, default=10,
                        help="Number of trials for random search")
    parser.add_argument("--quick", action="store_true",
                        help="Use quick (smaller) search space")
    parser.add_argument("--asset", type=str, default="BTC-USD",
                        help="Asset to train on")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Device")
    parser.add_argument("--output-dir", type=str, default="artifacts/tuning",
                        help="Output directory")

    args = parser.parse_args()

    if args.component == "all":
        components = ["ncc", "fmgp", "neural_jump", "diff_greeks", "neural_vol"]
        all_results = {}
        for comp in components:
            results = run_hyperparameter_search(
                component=comp,
                method=args.method,
                n_trials=args.trials,
                quick=args.quick,
                asset_id=args.asset,
                device=args.device,
                output_dir=args.output_dir,
            )
            all_results[comp] = results

        # Save combined summary
        output_path = Path(args.output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = output_path / f"tuning_all_{timestamp}.json"

        with open(summary_file, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"\n{'='*60}")
        print("ALL COMPONENTS TUNED")
        print(f"{'='*60}")
        for comp, results in all_results.items():
            print(f"\n{comp.upper()}: loss={results['best_loss']:.6f}")
            for k, v in results['best_config'].items():
                print(f"  {k}: {v}")
        print(f"\nCombined results saved to: {summary_file}")
    else:
        run_hyperparameter_search(
            component=args.component,
            method=args.method,
            n_trials=args.trials,
            quick=args.quick,
            asset_id=args.asset,
            device=args.device,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
