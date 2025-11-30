"""
Neural Rough Volatility Engine.

Implements rough volatility dynamics based on:
- "Volatility is Rough" (Gatheral, Jaisson & Rosenbaum, 2018)
- "Deep Learning Volatility" (Horvath et al., 2021)

Key insight: Log-volatility behaves like fractional Brownian motion with H ≈ 0.1,
explaining volatility clustering, term structure, and skew dynamics better than
classical models (Heston, SABR) which assume H = 0.5.
"""

import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_default_horizon() -> int:
    """Get default horizon from environment or use 90."""
    return int(os.getenv("TRAINING_HORIZON", "90"))


def _get_vol_hidden_dim() -> int:
    """Get Neural Vol hidden_dim from tuned env var or use default."""
    return int(os.getenv("TUNING_VOL_HIDDEN_DIM", "128"))


def _get_vol_hurst() -> float:
    """Get Neural Vol hurst from tuned env var or use default."""
    return float(os.getenv("TUNING_VOL_HURST", "0.1"))


def _get_vol_learning_rate() -> float:
    """Get Neural Vol learning_rate from tuned env var or use default."""
    return float(os.getenv("TUNING_VOL_LR", "0.001"))


def _get_vol_epochs() -> int:
    """Get Neural Vol training epochs from tuned env var or use default."""
    return int(os.getenv("TUNING_VOL_EPOCHS", "30"))


@dataclass
class NeuralRoughVolConfig:
    """Configuration for Neural Rough Volatility."""

    hurst: float = field(default_factory=_get_vol_hurst)
    horizon: int = field(default_factory=_get_default_horizon)
    cond_dim: int = 10  # Conditioning features
    hidden_dim: int = field(default_factory=_get_vol_hidden_dim)
    n_vol_params: int = 3  # xi (vol-of-vol), rho (correlation), fwd_var
    learning_rate: float = field(default_factory=_get_vol_learning_rate)
    default_epochs: int = field(default_factory=_get_vol_epochs)
    artifact_path: str = "artifacts/neural_rough_vol_state.pt"


class FractionalKernel(nn.Module):
    """
    Fractional covariance kernel for rough volatility.

    Cov(B_H(s), B_H(t)) = 0.5 * (s^(2H) + t^(2H) - |t-s|^(2H))

    This is the integrated kernel for fractional Brownian motion,
    guaranteed to be positive definite for H ∈ (0, 1).
    """

    def __init__(self, hurst: float = 0.1):
        super().__init__()
        self.hurst = hurst

    def forward(self, time_grid: torch.Tensor) -> torch.Tensor:
        """
        Compute fractional covariance matrix.

        Args:
            time_grid: (n,) time points

        Returns:
            (n, n) covariance matrix
        """
        n = time_grid.shape[0]

        # Pairwise time differences
        t_i = time_grid.unsqueeze(1)  # (n, 1)
        t_j = time_grid.unsqueeze(0)  # (1, n)

        # Fractional Brownian motion covariance (integrated kernel)
        # Cov(B_H(s), B_H(t)) = 0.5 * (s^(2H) + t^(2H) - |t-s|^(2H))
        # This is ALWAYS positive definite for H ∈ (0, 1)
        H = self.hurst

        s_2H = torch.pow(t_i, 2 * H)  # (n, 1)
        t_2H = torch.pow(t_j, 2 * H)  # (1, n)
        diff_2H = torch.pow(torch.abs(t_i - t_j) + 1e-10, 2 * H)  # (n, n)

        kernel = 0.5 * (s_2H + t_2H - diff_2H)

        # Add jitter for numerical stability in Cholesky
        kernel = kernel + 1e-6 * torch.eye(n, device=time_grid.device)

        return kernel


class ConditionalRoughVolNet(nn.Module):
    """
    Neural network for rough vol parameters conditioned on regime and MM state.
    """

    def __init__(self, cond_dim: int, hidden_dim: int, n_params: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, n_params),
        )

    def forward(self, conditioning: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict rough vol parameters.

        Args:
            conditioning: (batch, cond_dim)

        Returns:
            xi: (batch,) vol-of-vol
            rho: (batch,) correlation with price
            fwd_var: (batch,) forward variance level
        """
        params = self.net(conditioning)  # (batch, 3)

        # xi: positive (vol-of-vol)
        xi = F.softplus(params[:, 0])

        # rho: [-1, 1] (correlation)
        rho = torch.tanh(params[:, 1])

        # fwd_var: positive (forward variance)
        fwd_var = F.softplus(params[:, 2])

        return xi, rho, fwd_var


class NeuralRoughVolEngine(nn.Module):
    """
    Neural Rough Volatility Engine.

    Combines fractional kernel (rough dynamics) with neural parameterization.
    """

    def __init__(self, config: NeuralRoughVolConfig):
        super().__init__()
        self.config = config

        # Fractional kernel
        self.frac_kernel = FractionalKernel(hurst=config.hurst)

        # Neural network for parameters
        self.param_net = ConditionalRoughVolNet(
            cond_dim=config.cond_dim,
            hidden_dim=config.hidden_dim,
            n_params=config.n_vol_params,
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=config.learning_rate)

    def forward(
        self,
        past_vol: torch.Tensor,
        conditioning: torch.Tensor,
        horizon: int,
    ) -> torch.Tensor:
        """
        Forecast rough volatility path.

        Args:
            past_vol: (batch,) current volatility level
            conditioning: (batch, cond_dim) regime + MM features
            horizon: Forecast horizon

        Returns:
            vol_paths: (batch, horizon) volatility forecasts
        """
        batch_size = past_vol.shape[0]
        device = past_vol.device

        # Get parameters from neural network
        xi, rho, fwd_var = self.param_net(conditioning)

        # Time grid
        time_grid = torch.linspace(0, horizon / 365.0, horizon, device=device)

        # Fractional covariance
        frac_cov = self.frac_kernel(time_grid)  # (horizon, horizon)

        # Cholesky decomposition for sampling
        # With the corrected kernel, this should always succeed
        # But we keep fallback for numerical edge cases
        try:
            L = torch.linalg.cholesky(frac_cov)
        except RuntimeError as e:
            # Add more jitter if Cholesky fails (should be rare now)
            jitter_scale = 1e-4
            for _ in range(3):  # Try up to 3 times with increasing jitter
                try:
                    frac_cov = frac_cov + jitter_scale * torch.eye(horizon, device=device)
                    L = torch.linalg.cholesky(frac_cov)
                    break
                except RuntimeError:
                    jitter_scale *= 10
            else:
                # Last resort: use eigenvalue decomposition
                eigenvalues, eigenvectors = torch.linalg.eigh(frac_cov)
                eigenvalues = torch.clamp(eigenvalues, min=1e-6)
                L = eigenvectors @ torch.diag(torch.sqrt(eigenvalues))

        # Sample fractional Brownian motion
        z = torch.randn(batch_size, horizon, device=device)
        fBm = (L @ z.T).T  # (batch, horizon)

        # Rough Bergomi: V(t) = V(0) * exp(xi * fBm(t) - 0.5 * xi^2 * t^(2H))
        t_2H = time_grid.pow(2 * self.config.hurst)

        # Numerical stability: clamp past_vol to avoid log(0)
        past_vol_clamped = torch.clamp(past_vol, min=1e-6)

        log_vol = (
            torch.log(past_vol_clamped).unsqueeze(1)
            + xi.unsqueeze(1) * fBm
            - 0.5 * xi.unsqueeze(1).pow(2) * t_2H.unsqueeze(0)
        )

        # Clamp log_vol to avoid extreme values
        log_vol = torch.clamp(log_vol, min=-20, max=5)

        vol_paths = torch.exp(log_vol)

        # Apply forward variance adjustment
        vol_paths = vol_paths * torch.sqrt(torch.clamp(fwd_var.unsqueeze(1), min=1e-6) / (past_vol_clamped.unsqueeze(1) + 1e-8))

        return vol_paths

    def training_loss(
        self,
        vol_paths: torch.Tensor,
        target_vol_paths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Training loss.

        Args:
            vol_paths: (batch, horizon) predicted vol paths
            target_vol_paths: (batch, horizon) target vol paths

        Returns:
            loss, metrics dict
        """
        # Clamp values for numerical stability
        vol_paths_clamped = torch.clamp(vol_paths, min=1e-6)
        target_clamped = torch.clamp(target_vol_paths, min=1e-6)

        # Log-space MSE (more stable)
        log_pred = torch.log(vol_paths_clamped)
        log_target = torch.log(target_clamped)

        loss = F.mse_loss(log_pred, log_target)

        # Handle NaN loss
        if torch.isnan(loss):
            loss = torch.tensor(1.0, device=vol_paths.device, requires_grad=True)

        metrics = {
            "loss": loss.item(),
            "mean_pred_vol": vol_paths.mean().item(),
            "mean_target_vol": target_vol_paths.mean().item(),
        }

        return loss, metrics

    def train_step(
        self,
        past_vol: torch.Tensor,
        conditioning: torch.Tensor,
        target_vol_paths: torch.Tensor,
    ) -> Dict[str, float]:
        """Single training step."""
        self.optimizer.zero_grad()

        horizon = target_vol_paths.shape[1]
        vol_paths = self.forward(past_vol, conditioning, horizon)

        loss, metrics = self.training_loss(vol_paths, target_vol_paths)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

        return metrics

    def save(self, path: Path) -> None:
        """Save model state."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": self.state_dict(), "config": self.config}, path)

    @classmethod
    def load(cls, path: Path) -> "NeuralRoughVolEngine":
        """Load model state."""
        checkpoint = torch.load(path, weights_only=False)
        model = cls(checkpoint["config"])
        model.load_state_dict(checkpoint["model_state"])
        return model


class NeuralRoughVolWrapper:
    """
    Wrapper for Neural Rough Vol compatible with existing interface.
    """

    def __init__(self, config: Optional[NeuralRoughVolConfig] = None, device: str = "cpu"):
        self.config = config or NeuralRoughVolConfig()
        self.device = device
        self.model = NeuralRoughVolEngine(self.config).to(device)

    def forecast(
        self,
        iv_points: Dict[str, float],
        horizon: int,
        regime_strength: float = 1.0,
        mm_indices: Optional[Tuple[float, float, float]] = None,
    ) -> List[float]:
        """
        Forecast volatility path (compatible with existing interface).

        Args:
            iv_points: {'iv_7d_atm': ..., 'iv_14d_atm': ..., 'iv_30d_atm': ...}
            horizon: Forecast horizon
            regime_strength: Regime strength
            mm_indices: (gamma_squeeze, inventory_unwind, basis_pressure)

        Returns:
            List of volatility forecasts
        """
        self.model.eval()

        # Current vol level - handle NaN values
        base_iv = iv_points.get("iv_7d_atm", 0.5)
        if base_iv is None or (isinstance(base_iv, float) and math.isnan(base_iv)):
            base_iv = 0.5

        # Build conditioning - handle NaN values from mm_indices
        gsi, inventory, basis_pressure = mm_indices or (0.0, 0.0, 0.0)
        safe_gsi = 0.0 if (gsi is None or (isinstance(gsi, float) and math.isnan(gsi))) else gsi
        safe_inventory = 0.0 if (inventory is None or (isinstance(inventory, float) and math.isnan(inventory))) else inventory
        safe_basis_pressure = 0.0 if (basis_pressure is None or (isinstance(basis_pressure, float) and math.isnan(basis_pressure))) else basis_pressure

        iv_30 = iv_points.get("iv_30d_atm", base_iv)
        if iv_30 is None or (isinstance(iv_30, float) and math.isnan(iv_30)):
            iv_30 = base_iv
        skew = iv_points.get("skew_25d", 0.0)
        if skew is None or (isinstance(skew, float) and math.isnan(skew)):
            skew = 0.0

        conditioning = [
            base_iv,
            iv_30,
            skew,
            regime_strength,
            safe_gsi,
            safe_inventory,
            safe_basis_pressure,
        ]

        # Pad to cond_dim
        while len(conditioning) < self.config.cond_dim:
            conditioning.append(0.0)
        conditioning = conditioning[: self.config.cond_dim]

        # Convert to tensors
        past_vol_t = torch.tensor([base_iv], device=self.device, dtype=torch.float32)
        cond_t = torch.tensor([conditioning], device=self.device, dtype=torch.float32)

        # Forecast
        with torch.no_grad():
            vol_paths = self.model(past_vol_t, cond_t, horizon)

        # Convert to list
        vol_forecast = vol_paths[0].cpu().numpy().tolist()

        return vol_forecast

    def generate_vol_path(
        self,
        past_vol: float,
        conditioning: Sequence[float],
        horizon: int,
    ) -> List[float]:
        """
        Generate a volatility path from past vol and conditioning.

        This is a simpler interface for hyperparameter tuning validation.

        Args:
            past_vol: Starting volatility level
            conditioning: Conditioning features
            horizon: Number of steps to forecast

        Returns:
            List of volatility forecasts
        """
        import math
        self.model.eval()

        # Pad conditioning to cond_dim and replace NaN with 0
        cond_list = list(conditioning) + [0.0] * (self.config.cond_dim - len(conditioning))
        cond_list = cond_list[:self.config.cond_dim]
        cond_list = [0.0 if (math.isnan(x) or math.isinf(x)) else x for x in cond_list]

        # Ensure past_vol is valid
        if math.isnan(past_vol) or math.isinf(past_vol) or past_vol <= 0:
            past_vol = 0.2  # Default volatility

        # Convert to tensors
        past_vol_t = torch.tensor([past_vol], device=self.device, dtype=torch.float32)
        cond_t = torch.tensor([cond_list], device=self.device, dtype=torch.float32)

        # Forecast
        with torch.no_grad():
            vol_paths = self.model(past_vol_t, cond_t, horizon)

        return vol_paths[0].cpu().numpy().tolist()

    def train_on_historical(
        self,
        past_vols: Sequence[float],
        conditioning_sequences: Sequence[Sequence[float]],
        target_vol_paths: Sequence[Sequence[float]],
        epochs: int = 10,
        batch_size: int = 32,
    ) -> Dict[str, List[float]]:
        """
        Train on historical volatility paths.

        Args:
            past_vols: List of starting volatilities
            conditioning_sequences: List of conditioning features
            target_vol_paths: List of realized vol paths
            epochs: Training epochs
            batch_size: Batch size

        Returns:
            Training metrics history
        """
        self.model.train()

        n_samples = len(past_vols)
        indices = list(range(n_samples))

        import math

        # Convert to tensors, clamping to avoid NaN/inf
        past_vols_clean = [max(v, 1e-6) if not (math.isnan(v) or math.isinf(v)) else 0.2 for v in past_vols]
        past_vols_t = torch.tensor(past_vols_clean, device=self.device, dtype=torch.float32)

        # Pad conditioning and clean NaN/inf values
        cond_list = []
        for cond in conditioning_sequences:
            padded = list(cond) + [0.0] * (self.config.cond_dim - len(cond))
            cleaned = [0.0 if (math.isnan(x) or math.isinf(x)) else x for x in padded[: self.config.cond_dim]]
            cond_list.append(cleaned)
        cond_t = torch.tensor(cond_list, device=self.device, dtype=torch.float32)

        # Pad target paths
        max_len = max(len(p) for p in target_vol_paths)
        targets_list = []
        for path in target_vol_paths:
            padded = list(path) + [path[-1]] * (max_len - len(path))
            targets_list.append(padded)
        targets_t = torch.tensor(targets_list, device=self.device, dtype=torch.float32)

        # Training loop
        metrics_history = {"loss": [], "mean_pred_vol": [], "mean_target_vol": []}

        for epoch in range(epochs):
            import random
            random.shuffle(indices)

            epoch_metrics = {k: [] for k in metrics_history.keys()}

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_indices = indices[start:end]

                batch_past_vols = past_vols_t[batch_indices]
                batch_cond = cond_t[batch_indices]
                batch_targets = targets_t[batch_indices]

                metrics = self.model.train_step(batch_past_vols, batch_cond, batch_targets)

                for k in metrics_history.keys():
                    if k in metrics:
                        epoch_metrics[k].append(metrics[k])

            # Average epoch metrics
            for k in metrics_history.keys():
                if epoch_metrics[k]:
                    metrics_history[k].append(np.mean(epoch_metrics[k]))

        return metrics_history

    def save(self, path: Path) -> None:
        """Save model."""
        self.model.save(path)

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "NeuralRoughVolWrapper":
        """Load model."""
        model = NeuralRoughVolEngine.load(path)
        wrapper = cls(config=model.config, device=device)
        wrapper.model = model
        return wrapper
