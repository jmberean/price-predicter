"""
Hyperparameter tuning for SOTA neural components.

Uses train/validation split to prevent overfitting.
Automatically updates .env with best hyperparameters.
Supports parallel execution for faster tuning.

Usage:
    python scripts/hyperparameter_tuning.py --component ncc --method grid
    python scripts/hyperparameter_tuning.py --component all --quick
    python scripts/hyperparameter_tuning.py --component all --quick -w 4   # 4 parallel workers
    python scripts/hyperparameter_tuning.py --component all --quick -w 0   # auto-detect workers
    python scripts/hyperparameter_tuning.py --component all --thorough
    python scripts/hyperparameter_tuning.py --component all --validate

Tuning Modes:
    | Mode               | Samples | Time Estimate                   |
    |--------------------|---------|--------------------------------|
    | --validate         | 5       | ~1 min (just verifies it works) |
    | Default (standard) | 80      | ~1-2 hours                      |
    | --quick            | 20      | ~15-20 min                      |
    | --thorough         | 150     | ~3-4 hours                      |

Parallel Options:
    -w, --workers N     Number of parallel workers (default: 1 = sequential)
                        Use -w 0 to auto-detect based on CPU count

The tuned hyperparameters are written to .env as TUNING_* variables,
which are automatically read by model configs during training.
"""

import os
import sys
sys.path.insert(0, "src")

import argparse
import json
import itertools
import time
import re
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np

from aetheris_oracle.data.free_connectors import FreeDataConnector
from aetheris_oracle.pipeline.train_sota import (
    train_fmgp_residuals,
    train_neural_jump_sde,
    train_differentiable_greeks,
    train_neural_rough_vol,
)

# Default number of parallel workers (auto-detect CPU count)
DEFAULT_WORKERS = max(1, multiprocessing.cpu_count() - 2)  # Leave 2 cores free


# Thorough search spaces (more options, prioritize quality)
THOROUGH_SEARCH_SPACES = {
    "ncc": {
        "learning_rate": [0.00005, 0.0001, 0.0005, 0.001, 0.005],
        "epochs": [20, 30, 40, 50],
        "hidden_dim": [32, 64, 128, 256],
        "smoothness_weight": [0.01, 0.05, 0.1, 0.2],
    },
    "fmgp": {
        "learning_rate": [0.00005, 0.0001, 0.0005, 0.001],
        "epochs": [40, 60, 80, 100],
        "time_embed_dim": [32, 64, 128, 256],
        "cfg_dropout": [0.0, 0.1, 0.2, 0.3],
    },
    "neural_jump": {
        "learning_rate": [0.00005, 0.0001, 0.0005, 0.001],
        "epochs": [40, 60, 80, 100],
        "hidden_dim": [64, 128, 256, 512],
        "n_layers": [2, 3, 4, 5],
    },
    "diff_greeks": {
        "learning_rate": [0.00005, 0.0001, 0.0005, 0.001],
        "epochs": [30, 40, 50, 60],
        "n_heads": [2, 4, 8, 16],
        "embed_dim": [32, 64, 128, 256],
    },
    "neural_vol": {
        "learning_rate": [0.00005, 0.0001, 0.0005, 0.001],
        "epochs": [30, 40, 50, 60],
        "hidden_dim": [64, 128, 256, 512],
        "hurst": [0.05, 0.08, 0.1, 0.12, 0.15],
    },
}

# Standard search spaces
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

# Validate search spaces (minimal for quick validation)
VALIDATE_SEARCH_SPACES = {
    "ncc": {
        "learning_rate": [0.001],
        "epochs": [5],
        "hidden_dim": [64],
        "smoothness_weight": [0.1],
    },
    "fmgp": {
        "learning_rate": [0.001],
        "epochs": [10],
        "time_embed_dim": [64],
        "cfg_dropout": [0.1],
    },
    "neural_jump": {
        "learning_rate": [0.001],
        "epochs": [10],
        "hidden_dim": [128],
        "n_layers": [3],
    },
    "diff_greeks": {
        "learning_rate": [0.001],
        "epochs": [10],
        "n_heads": [4],
        "embed_dim": [64],
    },
    "neural_vol": {
        "learning_rate": [0.001],
        "epochs": [10],
        "hidden_dim": [128],
        "hurst": [0.1],
    },
}

# Quick search spaces (fewer options for faster iteration)
QUICK_SEARCH_SPACES = {
    "ncc": {
        "learning_rate": [0.0005, 0.001],
        "epochs": [15, 25],
        "hidden_dim": [64, 128],
        "smoothness_weight": [0.1],
    },
    "fmgp": {
        "learning_rate": [0.0005, 0.001],
        "epochs": [40, 60],
        "time_embed_dim": [64, 128],
        "cfg_dropout": [0.1],
    },
    "neural_jump": {
        "learning_rate": [0.0005, 0.001],
        "epochs": [40, 60],
        "hidden_dim": [128, 256],
        "n_layers": [3],
    },
    "diff_greeks": {
        "learning_rate": [0.0005, 0.001],
        "epochs": [25, 35],
        "n_heads": [4, 8],
        "embed_dim": [64],
    },
    "neural_vol": {
        "learning_rate": [0.0005, 0.001],
        "epochs": [25, 35],
        "hidden_dim": [128, 256],
        "hurst": [0.1],
    },
}

# Number of training samples per mode
TUNING_SAMPLES = {
    "validate": 5,
    "quick": 25,  # NCC can produce fewer samples than requested due to forecast failures
    "standard": 80,
    "thorough": 150,
}

# Train/validation split ratio
VALIDATION_SPLIT = 0.2


def get_training_horizon() -> int:
    """Get training horizon from environment or use default of 90."""
    return int(os.getenv("TRAINING_HORIZON", "90"))


def get_tuning_horizons() -> List[int]:
    """
    Get horizon values for tuning based on TRAINING_HORIZON.

    Returns appropriate sub-horizons for calibration training.
    E.g., if TRAINING_HORIZON=7, returns [1, 3, 7]
          if TRAINING_HORIZON=30, returns [1, 7, 14, 30]
          if TRAINING_HORIZON=90, returns [1, 7, 14, 30, 90]
    """
    max_horizon = get_training_horizon()

    if max_horizon <= 7:
        horizons = [1, 3, max_horizon]
    elif max_horizon <= 14:
        horizons = [1, 3, 7, max_horizon]
    elif max_horizon <= 30:
        horizons = [1, 7, 14, max_horizon]
    else:
        horizons = [1, 7, 14, 30, max_horizon]

    # Remove duplicates and sort
    return sorted(list(set(horizons)))


# Mapping from component hyperparams to .env variable names
ENV_VAR_MAPPING = {
    "ncc": {
        "learning_rate": "TUNING_NCC_LR",
        "epochs": "TUNING_NCC_EPOCHS",
        "hidden_dim": "TUNING_NCC_HIDDEN_DIM",
        "smoothness_weight": "TUNING_NCC_SMOOTHNESS",
    },
    "fmgp": {
        "learning_rate": "TUNING_FMGP_LR",
        "epochs": "TUNING_FMGP_EPOCHS",
        "time_embed_dim": "TUNING_FMGP_TIME_EMBED",
        "cfg_dropout": "TUNING_FMGP_DROPOUT",
    },
    "neural_jump": {
        "learning_rate": "TUNING_JUMP_LR",
        "epochs": "TUNING_JUMP_EPOCHS",
        "hidden_dim": "TUNING_JUMP_HIDDEN_DIM",
        "n_layers": "TUNING_JUMP_LAYERS",
    },
    "diff_greeks": {
        "learning_rate": "TUNING_GREEKS_LR",
        "epochs": "TUNING_GREEKS_EPOCHS",
        "n_heads": "TUNING_GREEKS_HEADS",
        "embed_dim": "TUNING_GREEKS_EMBED_DIM",
    },
    "neural_vol": {
        "learning_rate": "TUNING_VOL_LR",
        "epochs": "TUNING_VOL_EPOCHS",
        "hidden_dim": "TUNING_VOL_HIDDEN_DIM",
        "hurst": "TUNING_VOL_HURST",
    },
}


def update_env_file(component: str, best_config: Dict, env_path: str = ".env") -> None:
    """Update .env file with best hyperparameters for a component."""
    if best_config is None:
        return

    env_file = Path(env_path)
    if not env_file.exists():
        print(f"  Warning: {env_path} not found, skipping .env update")
        return

    content = env_file.read_text()
    mapping = ENV_VAR_MAPPING.get(component, {})
    updated = False

    for param, value in best_config.items():
        env_var = mapping.get(param)
        if not env_var:
            continue

        # Format value
        if isinstance(value, float):
            if value < 0.01:
                formatted = f"{value:.6f}"
            else:
                formatted = f"{value:.4f}"
        else:
            formatted = str(int(value) if isinstance(value, float) and value.is_integer() else value)

        # Check if variable exists
        pattern = rf"^{env_var}=.*$"
        if re.search(pattern, content, re.MULTILINE):
            # Update existing
            content = re.sub(pattern, f"{env_var}={formatted}", content, flags=re.MULTILINE)
            updated = True
        else:
            # Add new variable
            # Find the TUNING section or add at end
            if "# TUNED HYPERPARAMETERS" not in content:
                content += "\n# ============================================================\n"
                content += "# TUNED HYPERPARAMETERS (auto-generated by hyperparameter_tuning.py)\n"
                content += "# ============================================================\n"
            content += f"{env_var}={formatted}\n"
            updated = True

    if updated:
        env_file.write_text(content)
        print(f"  Updated {env_path} with {component} hyperparameters")


def train_val_split(data_list: List, val_ratio: float = VALIDATION_SPLIT) -> Tuple[List, List]:
    """Split data into train and validation sets."""
    n = len(data_list)
    n_val = max(1, int(n * val_ratio))
    n_train = n - n_val
    return data_list[:n_train], data_list[n_train:]


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


def compute_validation_loss(predictions: List[float], actuals: List[float]) -> float:
    """Compute MSE validation loss."""
    if len(predictions) == 0 or len(actuals) == 0:
        return float("inf")
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    return float(np.mean((predictions - actuals) ** 2))


def train_and_evaluate_ncc(
    connector,
    config: Dict,
    asset_id: str = "BTC-USD",
    device: str = "cpu",
    n_samples: int = 50,
) -> Tuple[float, Dict]:
    """Train NCC with given hyperparameters and evaluate on validation set."""
    from aetheris_oracle.pipeline.neural_conformal_control import NCCCalibrationEngine, NCCConfig
    from aetheris_oracle.pipeline.training_data_prep import prepare_ncc_training_data

    ncc_config = NCCConfig(
        hidden_dim=config.get("hidden_dim", 64),
        learning_rate=config.get("learning_rate", 0.001),
        smoothness_weight=config.get("smoothness_weight", 0.1),
    )

    engine = NCCCalibrationEngine(config=ncc_config, device=device)

    # Prepare data (more samples for proper split)
    horizons = get_tuning_horizons()
    base_quantiles_list, actuals_list, features_list, horizon_indices_list = prepare_ncc_training_data(
        connector=connector,
        asset_id=asset_id,
        n_samples=n_samples,
        horizons=horizons,
    )

    min_samples_required = int(os.getenv("TRAINING_MIN_SAMPLES", "10"))
    if len(base_quantiles_list) < min_samples_required:
        return float("inf"), {"error": "Insufficient training data"}

    # Split into train/validation
    n = len(base_quantiles_list)
    indices = list(range(n))
    np.random.shuffle(indices)
    n_val = max(5, int(n * VALIDATION_SPLIT))
    train_idx, val_idx = indices[:-n_val], indices[-n_val:]

    # Convert data
    quantile_keys = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]

    def make_dicts(idx_list):
        bq = [{q: float(base_quantiles_list[i][j]) for j, q in enumerate(quantile_keys)} for i in idx_list]
        feat = [{f"f{j}": float(features_list[i][j]) for j in range(len(features_list[i]))} for i in idx_list]
        act = [actuals_list[i] for i in idx_list]
        hor = [horizons[horizon_indices_list[i]] for i in idx_list]
        return bq, feat, act, hor

    train_bq, train_feat, train_act, train_hor = make_dicts(train_idx)
    val_bq, val_feat, val_act, val_hor = make_dicts(val_idx)

    # Train on training set only
    train_metrics = engine.train_on_historical(
        base_quantiles=train_bq,
        features_list=train_feat,
        actuals=train_act,
        horizons=train_hor,
        epochs=config.get("epochs", 20),
        batch_size=16,
    )

    # Evaluate on validation set
    val_predictions = []
    for i in range(len(val_bq)):
        adjusted = engine.calibrate_quantiles(val_bq[i], features=val_feat[i], horizon=val_hor[i])
        val_predictions.append(adjusted.get(0.5, val_bq[i].get(0.5, 0)))

    val_loss = compute_validation_loss(val_predictions, val_act)
    train_loss = train_metrics.get("loss", [1.0])[-1] if isinstance(train_metrics.get("loss"), list) else 1.0

    return val_loss, {"train_loss": train_loss, "val_loss": val_loss}


def train_and_evaluate_fmgp(
    connector,
    config: Dict,
    asset_id: str = "BTC-USD",
    device: str = "cpu",
    n_samples: int = 50,
) -> Tuple[float, Dict]:
    """Train FM-GP with given hyperparameters and evaluate on validation set."""
    from aetheris_oracle.modules.fm_gp_residual import FMGPResidualEngine, FMGPConfig
    from aetheris_oracle.pipeline.training_data_prep import prepare_fmgp_residual_training_data
    import torch

    fmgp_config = FMGPConfig(
        time_embed_dim=config.get("time_embed_dim", 64),
        cfg_dropout=config.get("cfg_dropout", 0.1),
        learning_rate=config.get("learning_rate", 0.001),
    )

    engine = FMGPResidualEngine(config=fmgp_config, device=device)

    # Prepare data
    conditioning_list, residual_paths_list = prepare_fmgp_residual_training_data(
        connector=connector,
        asset_id=asset_id,
        n_samples=n_samples,
    )

    if len(residual_paths_list) < 10:
        return float("inf"), {"error": "Insufficient training data"}

    # Split into train/validation
    n = len(residual_paths_list)
    n_val = max(5, int(n * VALIDATION_SPLIT))
    train_res, val_res = residual_paths_list[:-n_val], residual_paths_list[-n_val:]
    train_cond, val_cond = conditioning_list[:-n_val], conditioning_list[-n_val:]

    # Train on training set
    train_metrics = engine.train_on_historical(
        residual_sequences=train_res,
        conditioning_sequences=train_cond,
        epochs=config.get("epochs", 50),
        batch_size=16,
    )

    # Evaluate on validation set - compute reconstruction loss
    engine.model.eval()
    val_losses = []
    with torch.no_grad():
        for i in range(len(val_res)):
            target = torch.tensor(val_res[i], dtype=torch.float32, device=device).unsqueeze(0)
            cond = torch.tensor(val_cond[i], dtype=torch.float32, device=device).unsqueeze(0)
            # Sample and compare to target (sample is on engine.model, not engine)
            generated = engine.model.sample(cond, n_paths=1, device=device)
            mse = float(((generated - target) ** 2).mean())
            val_losses.append(mse)

    val_loss = float(np.mean(val_losses))
    train_loss = train_metrics.get("loss", [1.0])[-1] if isinstance(train_metrics.get("loss"), list) else 1.0

    return val_loss, {"train_loss": train_loss, "val_loss": val_loss}


def train_and_evaluate_neural_jump(
    connector,
    config: Dict,
    asset_id: str = "BTC-USD",
    device: str = "cpu",
    n_samples: int = 50,
) -> Tuple[float, Dict]:
    """Train Neural Jump SDE with given hyperparameters and evaluate on validation set."""
    from aetheris_oracle.modules.neural_jump_sde import NeuralJumpSDEEngine, NeuralJumpSDEConfig
    from aetheris_oracle.pipeline.training_data_prep import prepare_jump_sde_training_data
    import torch

    jump_config = NeuralJumpSDEConfig(
        hidden_dim=config.get("hidden_dim", 128),
        n_layers=config.get("n_layers", 3),
        learning_rate=config.get("learning_rate", 0.001),
    )

    engine = NeuralJumpSDEEngine(config=jump_config, device=device)

    # Prepare data
    x0_list, conditioning_list, target_paths_list = prepare_jump_sde_training_data(
        connector=connector,
        asset_id=asset_id,
        n_samples=n_samples,
        horizon=get_training_horizon(),
    )

    if len(target_paths_list) < 10:
        return float("inf"), {"error": "Insufficient training data"}

    # Split into train/validation
    n = len(target_paths_list)
    n_val = max(5, int(n * VALIDATION_SPLIT))
    train_paths, val_paths = target_paths_list[:-n_val], target_paths_list[-n_val:]
    train_cond, val_cond = conditioning_list[:-n_val], conditioning_list[-n_val:]

    # Train on training set
    train_metrics = engine.train_on_historical(
        paths=train_paths,
        conditioning_sequences=train_cond,
        epochs=config.get("epochs", 50),
        batch_size=16,
    )

    # Evaluate on validation set
    engine.model.eval()
    val_losses = []
    with torch.no_grad():
        for i in range(len(val_paths)):
            target = torch.tensor(val_paths[i], dtype=torch.float32, device=device)
            # Sample and compare
            x0_tensor = torch.tensor([target[0].item()], device=device, dtype=torch.float32)
            cond_list = list(val_cond[i])
            vol_path = [0.2] * len(target)  # dummy constant volatility
            generated = engine.sample_sde_paths(x0_tensor, cond_list, len(target), vol_path)
            mse = float(((generated.squeeze() - target) ** 2).mean())
            val_losses.append(mse)

    val_loss = float(np.mean(val_losses))
    train_loss = train_metrics.get("loss", [1.0])[-1] if isinstance(train_metrics.get("loss"), list) else 1.0

    return val_loss, {"train_loss": train_loss, "val_loss": val_loss}


def train_and_evaluate_diff_greeks(
    connector,
    config: Dict,
    asset_id: str = "BTC-USD",
    device: str = "cpu",
    n_samples: int = 50,
) -> Tuple[float, Dict]:
    """Train Differentiable Greeks with given hyperparameters and evaluate on validation set."""
    import torch
    from aetheris_oracle.modules.differentiable_greeks import DifferentiableMMEngine, DifferentiableGreeksConfig

    greeks_config = DifferentiableGreeksConfig(
        n_heads=config.get("n_heads", 4),
        embed_dim=config.get("embed_dim", 64),
        learning_rate=config.get("learning_rate", 0.001),
    )

    engine = DifferentiableMMEngine(greeks_config).to(device)
    engine.optimizer = torch.optim.AdamW(engine.parameters(), lr=config.get("learning_rate", 0.001))

    # Generate synthetic data
    total_samples = max(n_samples, 100)
    n_val = int(total_samples * VALIDATION_SPLIT)

    def generate_batch(batch_size):
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
        return spot, strikes, taus, ivs, ois, funding, basis, skew, imbalance, target_returns

    # Generate validation set once
    val_data = generate_batch(n_val)

    # Train
    epochs = config.get("epochs", 30)
    batch_size = 16
    train_losses = []

    for epoch in range(epochs):
        batch_losses = []
        for _ in range((total_samples - n_val) // batch_size):
            train_batch = generate_batch(batch_size)
            metrics = engine.train_step(*train_batch)
            batch_losses.append(metrics["loss"])
        train_losses.append(np.mean(batch_losses))

    # Evaluate on validation set
    engine.eval()
    with torch.no_grad():
        spot, strikes, taus, ivs, ois, funding, basis, skew, imbalance, target_returns = val_data
        # forward returns (mm_state, attention_weights) tuple
        mm_state, _ = engine(spot, strikes, taus, ivs, ois, funding, basis, skew, imbalance)
        # Use mean of mm_state as prediction proxy
        predictions = mm_state.mean(dim=-1)
        val_loss = float(((predictions - target_returns) ** 2).mean())

    train_loss = train_losses[-1]
    return val_loss, {"train_loss": train_loss, "val_loss": val_loss}


def train_and_evaluate_neural_vol(
    connector,
    config: Dict,
    asset_id: str = "BTC-USD",
    device: str = "cpu",
    n_samples: int = 50,
) -> Tuple[float, Dict]:
    """Train Neural Rough Vol with given hyperparameters and evaluate on validation set."""
    from aetheris_oracle.modules.neural_rough_vol import NeuralRoughVolWrapper, NeuralRoughVolConfig
    from aetheris_oracle.pipeline.training_data_prep import prepare_neural_vol_training_data
    import torch

    vol_config = NeuralRoughVolConfig(
        hidden_dim=config.get("hidden_dim", 128),
        hurst=config.get("hurst", 0.1),
        learning_rate=config.get("learning_rate", 0.001),
    )

    wrapper = NeuralRoughVolWrapper(config=vol_config, device=device)

    # Prepare data
    past_vols, conditioning_list, target_vol_paths = prepare_neural_vol_training_data(
        connector=connector,
        asset_id=asset_id,
        n_samples=n_samples,
    )

    if len(target_vol_paths) < 10:
        return float("inf"), {"error": "Insufficient training data"}

    # Split into train/validation
    n = len(target_vol_paths)
    n_val = max(5, int(n * VALIDATION_SPLIT))
    train_vols, val_vols = past_vols[:-n_val], past_vols[-n_val:]
    train_cond, val_cond = conditioning_list[:-n_val], conditioning_list[-n_val:]
    train_targets, val_targets = target_vol_paths[:-n_val], target_vol_paths[-n_val:]

    # Train on training set
    train_metrics = wrapper.train_on_historical(
        past_vols=train_vols,
        conditioning_sequences=train_cond,
        target_vol_paths=train_targets,
        epochs=config.get("epochs", 30),
        batch_size=16,
    )

    # Evaluate on validation set
    wrapper.model.eval()
    val_losses = []
    with torch.no_grad():
        for i in range(len(val_vols)):
            target = torch.tensor(val_targets[i], dtype=torch.float32, device=device)
            past_vol = max(val_vols[i], 1e-6)  # Ensure positive volatility
            cond = torch.tensor(val_cond[i], dtype=torch.float32, device=device).unsqueeze(0)
            # Generate and compare
            try:
                generated = wrapper.generate_vol_path(past_vol, cond.squeeze(0).tolist(), horizon=len(target))
                gen_tensor = torch.tensor(generated, dtype=torch.float32, device=device)
                mse = float(((gen_tensor - target) ** 2).mean())
                if not np.isnan(mse) and not np.isinf(mse):
                    val_losses.append(mse)
            except Exception:
                pass  # Skip problematic samples

    if not val_losses:
        return float("inf"), {"error": "All validation samples failed"}
    val_loss = float(np.mean(val_losses))
    train_loss = train_metrics.get("loss", [1.0])[-1] if isinstance(train_metrics.get("loss"), list) else 1.0

    return val_loss, {"train_loss": train_loss, "val_loss": val_loss}


TRAIN_FUNCTIONS = {
    "ncc": train_and_evaluate_ncc,
    "fmgp": train_and_evaluate_fmgp,
    "neural_jump": train_and_evaluate_neural_jump,
    "diff_greeks": train_and_evaluate_diff_greeks,
    "neural_vol": train_and_evaluate_neural_vol,
}


def prefetch_training_data(component: str, connector, asset_id: str, n_samples: int) -> Dict[str, Any]:
    """
    Pre-fetch training data for a component. Called once in main process.
    Returns data that can be passed to worker processes.
    """
    from aetheris_oracle.pipeline.training_data_prep import (
        prepare_ncc_training_data,
        prepare_fmgp_residual_training_data,
        prepare_jump_sde_training_data,
        prepare_neural_vol_training_data,
    )

    if component == "ncc":
        horizons = get_tuning_horizons()
        base_quantiles_list, actuals_list, features_list, horizon_indices_list = prepare_ncc_training_data(
            connector=connector,
            asset_id=asset_id,
            n_samples=n_samples,
            horizons=horizons,
        )
        return {
            "type": "ncc",
            "horizons": horizons,
            "base_quantiles_list": base_quantiles_list,
            "actuals_list": actuals_list,
            "features_list": features_list,
            "horizon_indices_list": horizon_indices_list,
        }
    elif component == "fmgp":
        conditioning_list, residual_paths_list = prepare_fmgp_residual_training_data(
            connector=connector,
            asset_id=asset_id,
            n_samples=n_samples,
        )
        return {
            "type": "fmgp",
            "conditioning_list": conditioning_list,
            "residual_paths_list": residual_paths_list,
        }
    elif component == "neural_jump":
        x0_list, conditioning_list, target_paths_list = prepare_jump_sde_training_data(
            connector=connector,
            asset_id=asset_id,
            n_samples=n_samples,
            horizon=get_training_horizon(),
        )
        return {
            "type": "neural_jump",
            "x0_list": x0_list,
            "conditioning_list": conditioning_list,
            "target_paths_list": target_paths_list,
        }
    elif component == "neural_vol":
        past_vols, conditioning_list, target_vol_paths = prepare_neural_vol_training_data(
            connector=connector,
            asset_id=asset_id,
            n_samples=n_samples,
        )
        return {
            "type": "neural_vol",
            "past_vols": past_vols,
            "conditioning_list": conditioning_list,
            "target_vol_paths": target_vol_paths,
        }
    elif component == "diff_greeks":
        # diff_greeks uses synthetic data, no pre-fetch needed
        return {"type": "diff_greeks", "n_samples": n_samples}
    else:
        return {"type": "unknown"}


def _train_with_prefetched_data(component: str, config: Dict, prefetched_data: Dict, device: str) -> Tuple[float, Dict]:
    """Train a component using pre-fetched data (no API calls needed)."""
    import torch

    if component == "ncc":
        from aetheris_oracle.pipeline.neural_conformal_control import NCCCalibrationEngine, NCCConfig

        ncc_config = NCCConfig(
            hidden_dim=config.get("hidden_dim", 64),
            learning_rate=config.get("learning_rate", 0.001),
            smoothness_weight=config.get("smoothness_weight", 0.1),
        )
        engine = NCCCalibrationEngine(config=ncc_config, device=device)

        horizons = prefetched_data["horizons"]
        base_quantiles_list = prefetched_data["base_quantiles_list"]
        actuals_list = prefetched_data["actuals_list"]
        features_list = prefetched_data["features_list"]
        horizon_indices_list = prefetched_data["horizon_indices_list"]

        min_samples_required = int(os.getenv("TRAINING_MIN_SAMPLES", "10"))
        if len(base_quantiles_list) < min_samples_required:
            return float("inf"), {"error": "Insufficient training data"}

        # Split into train/validation
        n = len(base_quantiles_list)
        indices = list(range(n))
        np.random.shuffle(indices)
        n_val = max(5, int(n * VALIDATION_SPLIT))
        train_idx, val_idx = indices[:-n_val], indices[-n_val:]

        quantile_keys = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]

        def make_dicts(idx_list):
            bq = [{q: float(base_quantiles_list[i][j]) for j, q in enumerate(quantile_keys)} for i in idx_list]
            feat = [{f"f{j}": float(features_list[i][j]) for j in range(len(features_list[i]))} for i in idx_list]
            act = [actuals_list[i] for i in idx_list]
            hor = [horizons[horizon_indices_list[i]] for i in idx_list]
            return bq, feat, act, hor

        train_bq, train_feat, train_act, train_hor = make_dicts(train_idx)
        val_bq, val_feat, val_act, val_hor = make_dicts(val_idx)

        train_metrics = engine.train_on_historical(
            base_quantiles=train_bq,
            features_list=train_feat,
            actuals=train_act,
            horizons=train_hor,
            epochs=config.get("epochs", 20),
            batch_size=16,
        )

        val_predictions = []
        for i in range(len(val_bq)):
            adjusted = engine.calibrate_quantiles(val_bq[i], features=val_feat[i], horizon=val_hor[i])
            val_predictions.append(adjusted.get(0.5, val_bq[i].get(0.5, 0)))

        val_loss = compute_validation_loss(val_predictions, val_act)
        train_loss = train_metrics.get("loss", [1.0])[-1] if isinstance(train_metrics.get("loss"), list) else 1.0
        return val_loss, {"train_loss": train_loss, "val_loss": val_loss}

    elif component == "fmgp":
        from aetheris_oracle.modules.fm_gp_residual import FMGPResidualEngine, FMGPConfig

        fmgp_config = FMGPConfig(
            time_embed_dim=config.get("time_embed_dim", 64),
            cfg_dropout=config.get("cfg_dropout", 0.1),
            learning_rate=config.get("learning_rate", 0.001),
        )
        engine = FMGPResidualEngine(config=fmgp_config, device=device)

        conditioning_list = prefetched_data["conditioning_list"]
        residual_paths_list = prefetched_data["residual_paths_list"]

        if len(residual_paths_list) < 10:
            return float("inf"), {"error": "Insufficient training data"}

        n = len(residual_paths_list)
        n_val = max(5, int(n * VALIDATION_SPLIT))
        train_res, val_res = residual_paths_list[:-n_val], residual_paths_list[-n_val:]
        train_cond, val_cond = conditioning_list[:-n_val], conditioning_list[-n_val:]

        train_metrics = engine.train_on_historical(
            residual_sequences=train_res,
            conditioning_sequences=train_cond,
            epochs=config.get("epochs", 50),
            batch_size=16,
        )

        engine.model.eval()
        val_losses = []
        with torch.no_grad():
            for i in range(len(val_res)):
                target = torch.tensor(val_res[i], dtype=torch.float32, device=device).unsqueeze(0)
                cond = torch.tensor(val_cond[i], dtype=torch.float32, device=device).unsqueeze(0)
                generated = engine.model.sample(cond, n_paths=1, device=device)
                mse = float(((generated - target) ** 2).mean())
                val_losses.append(mse)

        val_loss = float(np.mean(val_losses))
        train_loss = train_metrics.get("loss", [1.0])[-1] if isinstance(train_metrics.get("loss"), list) else 1.0
        return val_loss, {"train_loss": train_loss, "val_loss": val_loss}

    elif component == "neural_jump":
        from aetheris_oracle.modules.neural_jump_sde import NeuralJumpSDEEngine, NeuralJumpSDEConfig

        jump_config = NeuralJumpSDEConfig(
            hidden_dim=config.get("hidden_dim", 128),
            n_layers=config.get("n_layers", 3),
            learning_rate=config.get("learning_rate", 0.001),
        )
        engine = NeuralJumpSDEEngine(config=jump_config, device=device)

        conditioning_list = prefetched_data["conditioning_list"]
        target_paths_list = prefetched_data["target_paths_list"]

        if len(target_paths_list) < 10:
            return float("inf"), {"error": "Insufficient training data"}

        n = len(target_paths_list)
        n_val = max(5, int(n * VALIDATION_SPLIT))
        train_paths, val_paths = target_paths_list[:-n_val], target_paths_list[-n_val:]
        train_cond, val_cond = conditioning_list[:-n_val], conditioning_list[-n_val:]

        train_metrics = engine.train_on_historical(
            paths=train_paths,
            conditioning_sequences=train_cond,
            epochs=config.get("epochs", 50),
            batch_size=16,
        )

        engine.model.eval()
        val_losses = []
        with torch.no_grad():
            for i in range(len(val_paths)):
                target = torch.tensor(val_paths[i], dtype=torch.float32, device=device)
                # Sample and compare
                x0_tensor = torch.tensor([target[0].item()], device=device, dtype=torch.float32)
                cond_list = list(val_cond[i])
                vol_path = [0.2] * len(target)  # dummy constant volatility
                generated = engine.sample_sde_paths(x0_tensor, cond_list, len(target), vol_path)
                mse = float(((generated.squeeze() - target) ** 2).mean())
                val_losses.append(mse)

        val_loss = float(np.mean(val_losses))
        train_loss = train_metrics.get("loss", [1.0])[-1] if isinstance(train_metrics.get("loss"), list) else 1.0
        return val_loss, {"train_loss": train_loss, "val_loss": val_loss}

    elif component == "neural_vol":
        from aetheris_oracle.modules.neural_rough_vol import NeuralRoughVolWrapper, NeuralRoughVolConfig

        vol_config = NeuralRoughVolConfig(
            hidden_dim=config.get("hidden_dim", 128),
            hurst=config.get("hurst", 0.1),
            learning_rate=config.get("learning_rate", 0.001),
        )
        wrapper = NeuralRoughVolWrapper(config=vol_config, device=device)

        past_vols = prefetched_data["past_vols"]
        conditioning_list = prefetched_data["conditioning_list"]
        target_vol_paths = prefetched_data["target_vol_paths"]

        if len(target_vol_paths) < 10:
            return float("inf"), {"error": "Insufficient training data"}

        n = len(target_vol_paths)
        n_val = max(5, int(n * VALIDATION_SPLIT))
        train_vols, val_vols = past_vols[:-n_val], past_vols[-n_val:]
        train_cond, val_cond = conditioning_list[:-n_val], conditioning_list[-n_val:]
        train_targets, val_targets = target_vol_paths[:-n_val], target_vol_paths[-n_val:]

        train_metrics = wrapper.train_on_historical(
            past_vols=train_vols,
            conditioning_sequences=train_cond,
            target_vol_paths=train_targets,
            epochs=config.get("epochs", 30),
            batch_size=16,
        )

        wrapper.model.eval()
        val_losses = []
        with torch.no_grad():
            for i in range(len(val_vols)):
                target = torch.tensor(val_targets[i], dtype=torch.float32, device=device)
                past_vol = max(val_vols[i], 1e-6)  # Ensure positive volatility
                cond = torch.tensor(val_cond[i], dtype=torch.float32, device=device).unsqueeze(0)
                try:
                    generated = wrapper.generate_vol_path(past_vol, cond.squeeze(0).tolist(), horizon=len(target))
                    gen_tensor = torch.tensor(generated, dtype=torch.float32, device=device)
                    mse = float(((gen_tensor - target) ** 2).mean())
                    if not np.isnan(mse) and not np.isinf(mse):
                        val_losses.append(mse)
                except Exception:
                    pass  # Skip problematic samples

        if not val_losses:
            return float("inf"), {"error": "All validation samples failed"}
        val_loss = float(np.mean(val_losses))
        train_loss = train_metrics.get("loss", [1.0])[-1] if isinstance(train_metrics.get("loss"), list) else 1.0
        return val_loss, {"train_loss": train_loss, "val_loss": val_loss}

    elif component == "diff_greeks":
        # diff_greeks uses synthetic data, call original function
        from aetheris_oracle.modules.differentiable_greeks import DifferentiableMMEngine, DifferentiableGreeksConfig

        greeks_config = DifferentiableGreeksConfig(
            n_heads=config.get("n_heads", 4),
            embed_dim=config.get("embed_dim", 64),
            learning_rate=config.get("learning_rate", 0.001),
        )

        engine = DifferentiableMMEngine(greeks_config).to(device)
        engine.optimizer = torch.optim.AdamW(engine.parameters(), lr=config.get("learning_rate", 0.001))

        n_samples = prefetched_data.get("n_samples", 100)
        total_samples = max(n_samples, 100)
        n_val = int(total_samples * VALIDATION_SPLIT)

        def generate_batch(batch_size):
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
            return spot, strikes, taus, ivs, ois, funding, basis, skew, imbalance, target_returns

        val_data = generate_batch(n_val)

        epochs = config.get("epochs", 30)
        batch_size = 16
        train_losses = []

        for epoch in range(epochs):
            batch_losses = []
            for _ in range((total_samples - n_val) // batch_size):
                train_batch = generate_batch(batch_size)
                metrics = engine.train_step(*train_batch)
                batch_losses.append(metrics["loss"])
            train_losses.append(np.mean(batch_losses))

        engine.eval()
        with torch.no_grad():
            spot, strikes, taus, ivs, ois, funding, basis, skew, imbalance, target_returns = val_data
            mm_state, _ = engine(spot, strikes, taus, ivs, ois, funding, basis, skew, imbalance)
            predictions = mm_state.mean(dim=-1)
            val_loss = float(((predictions - target_returns) ** 2).mean())

        train_loss = train_losses[-1]
        return val_loss, {"train_loss": train_loss, "val_loss": val_loss}

    else:
        return float("inf"), {"error": f"Unknown component: {component}"}


def _run_single_trial(args: Tuple) -> Dict[str, Any]:
    """
    Worker function for parallel trial execution.

    Must be a top-level function (not a method) to be picklable for multiprocessing.
    Uses pre-fetched data to avoid duplicate API calls.
    """
    trial_idx, config, component, device, prefetched_data = args

    try:
        start_time = time.time()
        val_loss, metrics = _train_with_prefetched_data(component, config, prefetched_data, device)
        duration = time.time() - start_time

        return {
            "trial": trial_idx + 1,
            "config": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in config.items()},
            "val_loss": float(val_loss),
            "train_loss": float(metrics.get("train_loss", 0)),
            "duration_seconds": duration,
            "success": True,
        }
    except Exception as e:
        import traceback
        return {
            "trial": trial_idx + 1,
            "config": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in config.items()},
            "val_loss": float("inf"),
            "error": str(e),
            "traceback": traceback.format_exc(),
            "success": False,
        }


def run_hyperparameter_search(
    component: str,
    method: str = "grid",
    n_trials: int = 10,
    mode: str = "standard",  # "quick", "standard", or "thorough"
    asset_id: str = "BTC-USD",
    device: str = "cpu",
    output_dir: str = "artifacts/tuning",
    update_env: bool = True,
    n_workers: int = 1,
) -> Dict[str, Any]:
    """
    Run hyperparameter search for a component using validation loss.

    Args:
        component: Component to tune (ncc, fmgp, etc.)
        method: Search method (grid, random)
        n_trials: Number of trials for random search
        mode: "quick", "standard", or "thorough"
        asset_id: Asset to train on
        device: Device (cpu/cuda)
        output_dir: Directory to save results
        update_env: Whether to auto-update .env with best config
        n_workers: Number of parallel workers (1 = sequential)

    Returns:
        Results dictionary with best config and all trials
    """
    print(f"\n{'='*60}")
    print(f"Hyperparameter Tuning: {component.upper()}")
    print(f"Mode: {mode.upper()}, Method: {method}, Workers: {n_workers}")
    print(f"Optimizing: VALIDATION LOSS (prevents overfitting)")
    print(f"{'='*60}")

    # Get search space based on mode
    if mode == "validate":
        search_spaces = VALIDATE_SEARCH_SPACES
    elif mode == "thorough":
        search_spaces = THOROUGH_SEARCH_SPACES
    elif mode == "quick":
        search_spaces = QUICK_SEARCH_SPACES
    else:
        search_spaces = SEARCH_SPACES

    if component not in search_spaces:
        raise ValueError(f"Unknown component: {component}")

    search_space = search_spaces[component]
    n_samples = TUNING_SAMPLES.get(mode, 80)

    # Generate configs
    if method == "grid":
        configs = grid_search_configs(search_space)
    else:
        configs = random_search_configs(search_space, n_trials)

    print(f"Testing {len(configs)} configurations with {n_samples} samples each...")

    # Run trials
    results = []
    best_val_loss = float("inf")
    best_config = None

    if n_workers > 1:
        # Parallel execution using ProcessPoolExecutor
        print(f"\nPre-fetching training data (single API call)...")

        # Fetch data once in main process
        connector = FreeDataConnector(enable_cache=True, cache_ttl_seconds=3600)
        prefetched_data = prefetch_training_data(component, connector, asset_id, n_samples)

        print(f"Running {len(configs)} trials in parallel with {n_workers} workers...")

        # Prepare arguments for each trial (pass prefetched data, not connector)
        trial_args = [
            (i, config, component, device, prefetched_data)
            for i, config in enumerate(configs)
        ]

        # Use ProcessPoolExecutor for parallel execution
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all trials
            futures = {executor.submit(_run_single_trial, args): args[0] for args in trial_args}

            # Collect results as they complete
            for future in as_completed(futures):
                trial_idx = futures[future]
                trial_result = future.result()
                results.append(trial_result)

                # Print progress
                if trial_result["success"]:
                    print(f"Trial {trial_result['trial']}/{len(configs)}: Val={trial_result['val_loss']:.4f}, Time={trial_result['duration_seconds']:.1f}s")
                    if trial_result["val_loss"] < best_val_loss:
                        best_val_loss = trial_result["val_loss"]
                        best_config = trial_result["config"]
                        print(f"  ** New best! **")
                else:
                    print(f"Trial {trial_result['trial']}/{len(configs)}: Error - {trial_result.get('error', 'Unknown')}")

        # Sort results by trial number for consistent output
        results.sort(key=lambda x: x["trial"])
    else:
        # Sequential execution (original behavior)
        connector = FreeDataConnector(enable_cache=True, cache_ttl_seconds=3600)
        train_fn = TRAIN_FUNCTIONS[component]

        for i, config in enumerate(configs):
            print(f"\nTrial {i+1}/{len(configs)}: {config}")

            try:
                start_time = time.time()
                val_loss, metrics = train_fn(connector, config, asset_id, device, n_samples)
                duration = time.time() - start_time

                trial_result = {
                    "trial": i + 1,
                    "config": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in config.items()},
                    "val_loss": float(val_loss),
                    "train_loss": float(metrics.get("train_loss", 0)),
                    "duration_seconds": duration,
                    "success": True,
                }

                train_loss_val = metrics.get('train_loss', 'N/A')
                if isinstance(train_loss_val, (int, float)):
                    print(f"  Train Loss: {train_loss_val:.6f}")
                else:
                    print(f"  Train Loss: {train_loss_val}")
                print(f"  Val Loss:   {val_loss:.6f}")
                print(f"  Time: {duration:.1f}s")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_config = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in config.items()}
                    print(f"  ** New best! **")

            except Exception as e:
                trial_result = {
                    "trial": i + 1,
                    "config": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in config.items()},
                    "val_loss": float("inf"),
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
        "mode": mode,
        "n_samples": n_samples,
        "validation_split": VALIDATION_SPLIT,
        "n_trials": len(configs),
        "best_config": best_config,
        "best_val_loss": float(best_val_loss) if best_val_loss != float("inf") else None,
        "timestamp": timestamp,
        "trials": results,
    }

    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"BEST CONFIG for {component.upper()}:")
    print(f"{'='*60}")
    if best_config is not None:
        print(f"  Validation Loss: {best_val_loss:.6f}")
        for k, v in best_config.items():
            print(f"  {k}: {v}")

        # Auto-update .env file
        if update_env:
            update_env_file(component, best_config)
    else:
        print(f"  No successful trials")

    print(f"\nResults saved to: {results_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for SOTA components (uses validation loss)")
    parser.add_argument("--component", type=str, default="ncc",
                        choices=["ncc", "fmgp", "neural_jump", "diff_greeks", "neural_vol", "all"],
                        help="Component to tune")
    parser.add_argument("--method", type=str, default="grid",
                        choices=["grid", "random"],
                        help="Search method")
    parser.add_argument("--trials", type=int, default=10,
                        help="Number of trials for random search")

    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--validate", action="store_true",
                           help="Validate mode: minimal test, 5 samples (~1 min)")
    mode_group.add_argument("--quick", action="store_true",
                           help="Quick mode: fewer options, 50 samples (~30 min)")
    mode_group.add_argument("--thorough", action="store_true",
                           help="Thorough mode: more options, 150 samples, prioritize quality (~3-4 hrs)")

    parser.add_argument("--asset", type=str, default="BTC-USD",
                        help="Asset to train on")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Device")
    parser.add_argument("--output-dir", type=str, default="artifacts/tuning",
                        help="Output directory")
    parser.add_argument("--no-env-update", action="store_true",
                        help="Don't auto-update .env with best hyperparameters")
    parser.add_argument("--workers", "-w", type=int, default=1,
                        help=f"Number of parallel workers (default: 1, auto: {DEFAULT_WORKERS}). "
                             "Use -w 0 for auto-detect, -w 1 for sequential.")

    args = parser.parse_args()

    # Handle workers auto-detect
    n_workers = args.workers if args.workers > 0 else DEFAULT_WORKERS
    if n_workers > 1:
        print(f"Parallel mode enabled: {n_workers} workers")

    # Determine mode
    if args.validate:
        mode = "validate"
        # Set low sample threshold and faster settings for validate mode
        os.environ["TRAINING_MIN_SAMPLES"] = "3"
        os.environ["TRAINING_HOLDOUT_DAYS"] = "7"
        os.environ["TRAINING_LOOKBACK_DAYS"] = "30"
        os.environ["FORECAST_NUM_PATHS"] = "100"
    elif args.thorough:
        mode = "thorough"
    elif args.quick:
        mode = "quick"
    else:
        mode = "standard"

    update_env = not args.no_env_update

    if args.component == "all":
        # NCC must run LAST because it needs other SOTA models trained first
        # Order: base components first, then NCC calibration
        components = ["fmgp", "neural_jump", "diff_greeks", "neural_vol", "ncc"]
        all_results = {}

        print("\n" + "="*60)
        print("TUNING ORDER: Base SOTA components first, then NCC")
        print("NCC calibrates forecasts from other components, so they must exist.")
        print("After tuning each base component, it will be trained and saved.")
        print("="*60 + "\n")

        # Mapping from component name to training function
        train_functions = {
            "fmgp": train_fmgp_residuals,
            "neural_jump": train_neural_jump_sde,
            "diff_greeks": train_differentiable_greeks,
            "neural_vol": train_neural_rough_vol,
        }

        for comp in components:
            results = run_hyperparameter_search(
                component=comp,
                method=args.method,
                n_trials=args.trials,
                mode=mode,
                asset_id=args.asset,
                device=args.device,
                output_dir=args.output_dir,
                update_env=update_env,
                n_workers=n_workers,
            )
            all_results[comp] = results

            # Train base components after tuning (NCC comes last and doesn't need this)
            if comp in train_functions and results.get('best_config'):
                print(f"\n>>> Training {comp.upper()} with best hyperparameters...")
                try:
                    # Create connector for training
                    train_connector = FreeDataConnector(enable_cache=True, cache_ttl_seconds=3600)
                    train_functions[comp](
                        connector=train_connector,
                        asset_id=args.asset,
                        device=args.device,
                    )
                    print(f">>> {comp.upper()} trained and saved to artifacts/")
                except Exception as e:
                    print(f">>> Warning: Failed to train {comp}: {e}")

        # Save combined summary
        output_path = Path(args.output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = output_path / f"tuning_all_{timestamp}.json"

        with open(summary_file, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"\n{'='*60}")
        print(f"ALL COMPONENTS TUNED ({mode.upper()} mode)")
        print(f"{'='*60}")
        for comp, results in all_results.items():
            if results.get('best_config'):
                print(f"\n{comp.upper()}: val_loss={results['best_val_loss']:.6f}")
                for k, v in results['best_config'].items():
                    print(f"  {k}: {v}")
            else:
                print(f"\n{comp.upper()}: No successful trials")

        if update_env:
            print(f"\n.env has been updated with best hyperparameters")
        print(f"Combined results saved to: {summary_file}")
    else:
        run_hyperparameter_search(
            component=args.component,
            method=args.method,
            n_trials=args.trials,
            mode=mode,
            asset_id=args.asset,
            device=args.device,
            output_dir=args.output_dir,
            update_env=update_env,
            n_workers=n_workers,
        )


if __name__ == "__main__":
    main()
