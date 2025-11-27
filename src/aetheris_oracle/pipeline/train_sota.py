"""
Training scripts for SOTA neural components.

Provides CLI and programmatic interfaces for training:
- Neural Conformal Control (NCC)
- FM-GP Residual Generator
- Neural Jump SDE
- Differentiable Greeks MM Engine
- Neural Rough Volatility
- MambaTS Trend Backbone
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..config import ForecastConfig
from ..data.connectors import SyntheticDataConnector
from ..data.free_connectors import FreeDataConnector
from ..data.interfaces import DataConnector
from ..features.regime import compute_regime_vector
from ..features.stationarity import StationarityNormalizer
from .training_data_prep import (
    prepare_ncc_training_data,
    prepare_jump_sde_training_data,
    prepare_diff_greeks_training_data,
    prepare_fmgp_residual_training_data,
    prepare_neural_vol_training_data,
    prepare_mamba_trend_training_data,
)


def train_ncc_calibration(
    connector: DataConnector,
    asset_id: str,
    epochs: int = 20,
    batch_size: int = 32,
    device: str = "cpu",
    artifact_path: str = "artifacts/ncc_calibration.pt",
) -> Dict[str, List[float]]:
    """
    Train Neural Conformal Control calibration engine.

    Args:
        connector: Data connector for fetching historical data
        asset_id: Asset to train on
        epochs: Training epochs
        batch_size: Batch size
        device: Device (cpu/cuda)
        artifact_path: Where to save trained model

    Returns:
        Training metrics history
    """
    from .neural_conformal_control import NCCCalibrationEngine, NCCConfig

    print(f"Training NCC Calibration on {asset_id}...")

    # Initialize engine
    config = NCCConfig()
    engine = NCCCalibrationEngine(config=config, device=device)

    # Prepare real historical training data (respects .env TRAINING_SAMPLES_NCC)
    horizons = [1, 3, 7, 14]

    base_quantiles_list, actuals_list, features_list, horizon_indices_list = prepare_ncc_training_data(
        connector=connector,
        asset_id=asset_id,
        n_samples=None,  # Will use TRAINING_SAMPLES_NCC from .env
        horizons=horizons,
    )

    # Convert data to expected format
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
    metrics_history = engine.train_on_historical(
        base_quantiles=base_quantiles_dicts,
        features_list=features_dicts,
        actuals=actuals_list,
        horizons=horizons_list,
        epochs=epochs,
        batch_size=batch_size,
    )

    # Save
    save_path = Path(artifact_path)
    engine.save(save_path)
    print(f"Saved NCC model to {save_path}")

    return metrics_history


def train_fmgp_residuals(
    connector: DataConnector,
    asset_id: str,
    epochs: int = 50,
    batch_size: int = 64,
    device: str = "cpu",
    artifact_path: str = "artifacts/fmgp_residuals.pt",
) -> Dict[str, List[float]]:
    """
    Train FM-GP Residual Generator.

    Args:
        connector: Data connector
        asset_id: Asset to train on
        epochs: Training epochs
        batch_size: Batch size
        device: Device
        artifact_path: Save path

    Returns:
        Training metrics history
    """
    from ..modules.fm_gp_residual import FMGPResidualEngine, FMGPConfig

    print(f"Training FM-GP Residuals on {asset_id}...")

    config = FMGPConfig()
    engine = FMGPResidualEngine(config=config, device=device)

    # Prepare REAL historical training data
    conditioning_list, residual_paths_list = prepare_fmgp_residual_training_data(
        connector=connector,
        asset_id=asset_id,
    )

    print(f"Training on {len(residual_paths_list)} real historical residual paths...")

    # Train
    metrics_history = engine.train_on_historical(
        residual_sequences=residual_paths_list,
        conditioning_sequences=conditioning_list,
        epochs=epochs,
        batch_size=batch_size,
    )

    # Save
    save_path = Path(artifact_path)
    engine.save(save_path)
    print(f"Saved FM-GP model to {save_path}")

    return metrics_history


def train_neural_jump_sde(
    connector: DataConnector,
    asset_id: str,
    epochs: int = 50,
    batch_size: int = 32,
    device: str = "cpu",
    artifact_path: str = "artifacts/neural_jump_sde.pt",
) -> Dict[str, List[float]]:
    """
    Train Neural Jump SDE.

    Args:
        connector: Data connector
        asset_id: Asset to train on
        epochs: Training epochs
        batch_size: Batch size
        device: Device
        artifact_path: Save path

    Returns:
        Training metrics history
    """
    from ..modules.neural_jump_sde import NeuralJumpSDEEngine, NeuralJumpSDEConfig

    print(f"Training Neural Jump SDE on {asset_id}...")

    config = NeuralJumpSDEConfig()
    engine = NeuralJumpSDEEngine(config=config, device=device)

    # Prepare real historical training data
    n_samples = 300
    horizon = 14

    x0_list, conditioning_list, target_paths_list = prepare_jump_sde_training_data(
        connector=connector,
        asset_id=asset_id,
        n_samples=n_samples,
        horizon=horizon,
    )

    # Train
    metrics_history = engine.train_on_historical(
        paths=target_paths_list,
        conditioning_sequences=conditioning_list,
        epochs=epochs,
        batch_size=batch_size,
    )

    # Save
    save_path = Path(artifact_path)
    engine.save(save_path)
    print(f"Saved Neural Jump SDE model to {save_path}")

    return metrics_history


def train_differentiable_greeks(
    connector: DataConnector,
    asset_id: str,
    epochs: int = 30,
    batch_size: int = 16,
    device: str = "cpu",
    artifact_path: str = "artifacts/diff_greeks.pt",
) -> Dict[str, float]:
    """
    Train Differentiable Greeks MM Engine.

    Args:
        connector: Data connector
        asset_id: Asset to train on
        epochs: Training epochs
        batch_size: Batch size
        device: Device
        artifact_path: Save path

    Returns:
        Final training metrics
    """
    from ..modules.differentiable_greeks import DifferentiableMMEngine, DifferentiableGreeksConfig

    print(f"Training Differentiable Greeks on {asset_id}...")

    config = DifferentiableGreeksConfig()
    engine = DifferentiableMMEngine(config).to(device)

    # Generate synthetic training data
    n_samples = 200

    for epoch in range(epochs):
        epoch_losses = []

        for _ in range(n_samples // batch_size):
            # Synthetic inputs
            spot = torch.rand(batch_size, device=device) * 50000 + 30000
            strikes = spot.unsqueeze(1) * torch.linspace(0.8, 1.2, config.n_strikes, device=device).unsqueeze(0)
            taus = torch.rand(batch_size, config.n_strikes, device=device) * 30 / 365
            ivs = torch.rand(batch_size, config.n_strikes, device=device) * 0.5 + 0.3
            ois = torch.rand(batch_size, config.n_strikes, device=device)
            funding = torch.randn(batch_size, device=device) * 0.01
            basis = torch.randn(batch_size, device=device) * 100
            skew = torch.randn(batch_size, device=device) * 0.1
            imbalance = torch.randn(batch_size, device=device) * 0.2

            # Target: future returns (synthetic)
            target_returns = torch.randn(batch_size, device=device) * 0.02

            # Train step
            metrics = engine.train_step(
                spot, strikes, taus, ivs, ois, funding, basis, skew, imbalance, target_returns
            )

            epoch_losses.append(metrics["loss"])

        avg_loss = np.mean(epoch_losses)
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss:.6f}")

    # Save
    save_path = Path(artifact_path)
    engine.save(save_path)
    print(f"Saved Differentiable Greeks model to {save_path}")

    return {"final_loss": avg_loss}


def train_neural_rough_vol(
    connector: DataConnector,
    asset_id: str,
    epochs: int = 30,
    batch_size: int = 32,
    device: str = "cpu",
    artifact_path: str = "artifacts/neural_rough_vol.pt",
) -> Dict[str, List[float]]:
    """
    Train Neural Rough Volatility.

    Args:
        connector: Data connector
        asset_id: Asset to train on
        epochs: Training epochs
        batch_size: Batch size
        device: Device
        artifact_path: Save path

    Returns:
        Training metrics history
    """
    from ..modules.neural_rough_vol import NeuralRoughVolWrapper

    print(f"Training Neural Rough Vol on {asset_id}...")

    wrapper = NeuralRoughVolWrapper(device=device)

    # Prepare REAL historical training data
    past_vols, conditioning_list, target_vol_paths = prepare_neural_vol_training_data(
        connector=connector,
        asset_id=asset_id,
    )

    print(f"Training on {len(target_vol_paths)} real historical volatility paths...")

    # Train
    metrics_history = wrapper.train_on_historical(
        past_vols=past_vols,
        conditioning_sequences=conditioning_list,
        target_vol_paths=target_vol_paths,
        epochs=epochs,
        batch_size=batch_size,
    )

    # Save
    save_path = Path(artifact_path)
    wrapper.save(save_path)
    print(f"Saved Neural Rough Vol model to {save_path}")

    return metrics_history


def train_mamba_trend(
    connector: DataConnector,
    asset_id: str,
    epochs: int = 40,
    batch_size: int = 32,
    device: str = "cpu",
    artifact_path: str = "artifacts/mamba_trend.pt",
) -> Dict[str, float]:
    """
    Train MambaTS Trend Backbone.

    Args:
        connector: Data connector
        asset_id: Asset to train on
        epochs: Training epochs
        batch_size: Batch size
        device: Device
        artifact_path: Save path

    Returns:
        Final training metrics
    """
    from ..modules.mamba_trend import MambaTrendBackbone, MambaTrendConfig

    print(f"Training MambaTS Trend on {asset_id}...")

    config = MambaTrendConfig()
    model = MambaTrendBackbone(config).to(device)

    # Prepare REAL historical training data
    input_sequences, target_sequences = prepare_mamba_trend_training_data(
        connector=connector,
        asset_id=asset_id,
    )

    print(f"Training on {len(input_sequences)} real historical sequences...")

    n_samples = len(input_sequences)

    for epoch in range(epochs):
        epoch_losses = []

        # Shuffle indices
        indices = np.random.permutation(n_samples)

        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]

            batch_inputs = [input_sequences[i] for i in batch_indices]
            batch_targets = [target_sequences[i] for i in batch_indices]

            x = torch.tensor(batch_inputs, device=device, dtype=torch.float32)
            targets = torch.tensor(batch_targets, device=device, dtype=torch.float32)

            # Train step
            metrics = model.train_step(x, targets)
            epoch_losses.append(metrics["loss"])

        avg_loss = np.mean(epoch_losses)
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss:.6f}")

    # Save
    save_path = Path(artifact_path)
    model.save(save_path)
    print(f"Saved MambaTS model to {save_path}")

    return {"final_loss": avg_loss}


def train_all_components(
    asset_id: str = "BTC-USD",
    device: str = "cpu",
    artifact_root: str = "artifacts",
    epochs_dict: Optional[Dict[str, int]] = None,
) -> Dict[str, Dict]:
    """
    Train all SOTA components.

    Args:
        asset_id: Asset to train on
        device: Device
        artifact_root: Root directory for artifacts
        epochs_dict: Custom epochs per component

    Returns:
        Combined training metrics
    """
    artifact_path = Path(artifact_root)
    artifact_path.mkdir(parents=True, exist_ok=True)

    epochs = epochs_dict or {
        "ncc": 20,
        "fmgp": 50,
        "neural_jump": 50,
        "diff_greeks": 30,
        "neural_vol": 30,
        "mamba": 40,
    }

    # Use REAL data connector
    connector = FreeDataConnector(enable_cache=True, cache_ttl_seconds=3600)

    results = {}

    print("=" * 60)
    print("Training SOTA Components")
    print("=" * 60)

    # NCC Calibration
    results["ncc"] = train_ncc_calibration(
        connector, asset_id, epochs=epochs["ncc"], device=device,
        artifact_path=str(artifact_path / "ncc_calibration.pt")
    )

    # FM-GP Residuals
    results["fmgp"] = train_fmgp_residuals(
        connector, asset_id, epochs=epochs["fmgp"], device=device,
        artifact_path=str(artifact_path / "fmgp_residuals.pt")
    )

    # Neural Jump SDE
    results["neural_jump"] = train_neural_jump_sde(
        connector, asset_id, epochs=epochs["neural_jump"], device=device,
        artifact_path=str(artifact_path / "neural_jump_sde.pt")
    )

    # Differentiable Greeks
    results["diff_greeks"] = train_differentiable_greeks(
        connector, asset_id, epochs=epochs["diff_greeks"], device=device,
        artifact_path=str(artifact_path / "diff_greeks.pt")
    )

    # Neural Rough Vol
    results["neural_vol"] = train_neural_rough_vol(
        connector, asset_id, epochs=epochs["neural_vol"], device=device,
        artifact_path=str(artifact_path / "neural_rough_vol.pt")
    )

    # MambaTS
    results["mamba"] = train_mamba_trend(
        connector, asset_id, epochs=epochs["mamba"], device=device,
        artifact_path=str(artifact_path / "mamba_trend.pt")
    )

    print("=" * 60)
    print("Training Complete!")
    print(f"Models saved to: {artifact_path}")
    print("=" * 60)

    # Save metrics summary
    metrics_path = artifact_path / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    return results


def main():
    """CLI entrypoint for training SOTA components."""
    parser = argparse.ArgumentParser(description="Train SOTA neural components")
    parser.add_argument("--component", type=str, default="all",
                        choices=["all", "ncc", "fmgp", "neural_jump", "diff_greeks", "neural_vol", "mamba"],
                        help="Component to train")
    parser.add_argument("--asset", type=str, default="BTC-USD", help="Asset ID")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs (overrides defaults)")
    parser.add_argument("--artifact-root", type=str, default="artifacts", help="Artifact directory")

    args = parser.parse_args()

    # Use real data connector for training
    print(f"Initializing FreeDataConnector for {args.asset}...")
    connector = FreeDataConnector(enable_cache=True, cache_ttl_seconds=3600)

    if args.component == "all":
        epochs_dict = None
        if args.epochs:
            epochs_dict = {k: args.epochs for k in ["ncc", "fmgp", "neural_jump", "diff_greeks", "neural_vol", "mamba"]}
        train_all_components(
            asset_id=args.asset,
            device=args.device,
            artifact_root=args.artifact_root,
            epochs_dict=epochs_dict,
        )
    else:
        epochs = args.epochs or 30

        if args.component == "ncc":
            train_ncc_calibration(connector, args.asset, epochs=epochs, device=args.device,
                                  artifact_path=f"{args.artifact_root}/ncc_calibration.pt")
        elif args.component == "fmgp":
            train_fmgp_residuals(connector, args.asset, epochs=epochs, device=args.device,
                                 artifact_path=f"{args.artifact_root}/fmgp_residuals.pt")
        elif args.component == "neural_jump":
            train_neural_jump_sde(connector, args.asset, epochs=epochs, device=args.device,
                                  artifact_path=f"{args.artifact_root}/neural_jump_sde.pt")
        elif args.component == "diff_greeks":
            train_differentiable_greeks(connector, args.asset, epochs=epochs, device=args.device,
                                        artifact_path=f"{args.artifact_root}/diff_greeks.pt")
        elif args.component == "neural_vol":
            train_neural_rough_vol(connector, args.asset, epochs=epochs, device=args.device,
                                   artifact_path=f"{args.artifact_root}/neural_rough_vol.pt")
        elif args.component == "mamba":
            train_mamba_trend(connector, args.asset, epochs=epochs, device=args.device,
                              artifact_path=f"{args.artifact_root}/mamba_trend.pt")


if __name__ == "__main__":
    main()
