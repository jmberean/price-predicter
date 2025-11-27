"""
Flow Matching with Gaussian Process Priors for Residual Generation.

Implements the approach from "Flow Matching with Gaussian Process Priors for
Probabilistic Time Series Forecasting" (Kollovieh et al., ICLR 2025).

Key innovation: Uses GP as base distribution instead of standard Gaussian,
encoding temporal correlation structure into the prior.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint


@dataclass
class FMGPConfig:
    """Configuration for FM-GP residual generator."""

    horizon: int = 14
    cond_dim: int = 10  # regime + vol + MM features
    hidden_dims: List[int] = None
    time_embed_dim: int = 64
    n_mixtures: int = 4  # Spectral mixture kernel components
    cfg_dropout: float = 0.1  # Classifier-free guidance dropout
    learning_rate: float = 0.001
    artifact_path: str = "artifacts/fm_gp_state.pt"

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256, 256]


class SpectralMixtureKernel(nn.Module):
    """
    Spectral Mixture Kernel for GP prior.

    k(τ) = Σ_q w_q * exp(-2π²τ²/s_q²) * cos(2πτμ_q)

    This kernel can represent any stationary covariance through spectral density.
    """

    def __init__(self, n_mixtures: int = 4):
        super().__init__()
        self.n_mixtures = n_mixtures

        # Learnable parameters (in log space for positivity)
        self.log_weights = nn.Parameter(torch.randn(n_mixtures))
        self.log_scales = nn.Parameter(torch.randn(n_mixtures))
        self.means = nn.Parameter(torch.randn(n_mixtures))

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """
        Compute kernel values for time lags tau.

        Args:
            tau: (n, n) matrix of time lags

        Returns:
            Covariance matrix
        """
        weights = F.softmax(self.log_weights, dim=0)
        scales = torch.exp(self.log_scales)

        # Compute kernel as sum of components
        cov = 0.0
        for q in range(self.n_mixtures):
            w_q = weights[q]
            s_q = scales[q]
            mu_q = self.means[q]

            # Spectral mixture component
            exp_term = torch.exp(-2 * (math.pi ** 2) * (tau ** 2) / (s_q ** 2))
            cos_term = torch.cos(2 * math.pi * tau * mu_q)

            cov = cov + w_q * exp_term * cos_term

        return cov


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding like in Transformers."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (batch,) time values in [0, 1]

        Returns:
            (batch, embed_dim) time embeddings
        """
        half_dim = self.embed_dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb_scale)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ConditionalVectorField(nn.Module):
    """
    Neural network that learns the vector field for flow matching.

    v_θ(x_t, t, c) where:
    - x_t: current state
    - t: time in [0, 1]
    - c: conditioning (regime, vol, MM)
    """

    def __init__(
        self, data_dim: int, cond_dim: int, hidden_dims: List[int], time_embed_dim: int
    ):
        super().__init__()
        self.data_dim = data_dim
        self.cond_dim = cond_dim

        # Time embedding
        self.time_embed = TimeEmbedding(time_embed_dim)

        # Input layer
        input_dim = data_dim + time_embed_dim + cond_dim
        layers = [nn.Linear(input_dim, hidden_dims[0]), nn.GELU()]

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.GELU(),
                nn.LayerNorm(hidden_dims[i + 1]),
            ])

        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], data_dim))

        self.net = nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, conditioning: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict velocity at (x, t, c).

        Args:
            x: (batch, data_dim) current state
            t: (batch,) time
            conditioning: (batch, cond_dim) or None

        Returns:
            (batch, data_dim) velocity
        """
        # Time embedding
        t_emb = self.time_embed(t)

        # Concatenate inputs
        if conditioning is None:
            conditioning = torch.zeros(x.shape[0], self.cond_dim, device=x.device)

        inp = torch.cat([x, t_emb, conditioning], dim=-1)

        return self.net(inp)


class FMGPResidualGenerator(nn.Module):
    """
    Flow Matching with GP prior for time-series residuals.

    Key insight: Use GP as the base distribution instead of standard Gaussian.
    This encodes temporal correlation structure into the prior.
    """

    def __init__(self, config: FMGPConfig):
        super().__init__()
        self.config = config

        # GP kernel for prior (learns temporal correlation)
        self.gp_kernel = SpectralMixtureKernel(n_mixtures=config.n_mixtures)

        # Conditional flow network
        self.flow_net = ConditionalVectorField(
            data_dim=config.horizon,
            cond_dim=config.cond_dim,
            hidden_dims=config.hidden_dims,
            time_embed_dim=config.time_embed_dim,
        )

        # Classifier-free guidance dropout
        self.cfg_dropout = config.cfg_dropout

        # Optimizer
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=config.learning_rate)

    def sample_gp_prior(self, n_samples: int, device: str = "cpu") -> torch.Tensor:
        """
        Sample from GP prior.

        Args:
            n_samples: Number of samples
            device: Device for tensors

        Returns:
            (n_samples, horizon) samples from GP
        """
        horizon = self.config.horizon

        # Time points
        time_points = torch.linspace(0, 1, horizon, device=device)

        # Compute covariance matrix
        tau = time_points[:, None] - time_points[None, :]  # (horizon, horizon)
        cov = self.gp_kernel(tau)

        # Add jitter for numerical stability
        cov = cov + 1e-5 * torch.eye(horizon, device=device)

        # Sample from multivariate normal
        dist = torch.distributions.MultivariateNormal(
            torch.zeros(horizon, device=device), cov
        )
        samples = dist.sample((n_samples,))

        return samples

    def sample(
        self, conditioning: torch.Tensor, n_paths: int = 1000, device: str = "cpu"
    ) -> torch.Tensor:
        """
        Generate residual paths via ODE integration.

        Args:
            conditioning: (batch, cond_dim) conditioning features
            n_paths: Number of paths to sample
            device: Device for computation

        Returns:
            (batch, n_paths, horizon) residual paths
        """
        self.eval()
        batch_size = conditioning.shape[0]

        with torch.no_grad():
            # Sample from GP prior
            x0 = self.sample_gp_prior(batch_size * n_paths, device=device)

            # Expand conditioning
            cond_expanded = conditioning.repeat_interleave(n_paths, dim=0)

            # ODE integration from t=0 to t=1
            def ode_func(t_scalar, x):
                t = torch.full((x.shape[0],), t_scalar, device=x.device)
                return self.flow_net(x, t, cond_expanded)

            t_span = torch.linspace(0, 1, 10, device=device)  # Integration steps
            paths = odeint(ode_func, x0, t_span)[-1]  # Take final state

            # Reshape to (batch, n_paths, horizon)
            paths = paths.view(batch_size, n_paths, self.config.horizon)

            # Zero-center each path
            paths = paths - paths.mean(dim=-1, keepdim=True)

        return paths

    def training_loss(
        self, x: torch.Tensor, conditioning: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Conditional flow matching loss with GP prior.

        Args:
            x: (batch, horizon) target residual paths
            conditioning: (batch, cond_dim) conditioning features

        Returns:
            loss, metrics dict
        """
        batch_size = x.shape[0]
        device = x.device

        # Sample time uniformly
        t = torch.rand(batch_size, device=device)

        # Sample from GP prior
        x0 = self.sample_gp_prior(batch_size, device=device)

        # Linear interpolation between GP sample and data
        xt = t[:, None] * x + (1 - t[:, None]) * x0

        # Target velocity (straight path)
        v_target = x - x0

        # Classifier-free guidance: randomly drop conditioning
        if self.training:
            mask = torch.rand(batch_size, device=device) > self.cfg_dropout
            cond_masked = conditioning * mask[:, None]
        else:
            cond_masked = conditioning

        # Predicted velocity
        v_pred = self.flow_net(xt, t, cond_masked)

        # MSE loss
        loss = F.mse_loss(v_pred, v_target)

        metrics = {
            "loss": loss.item(),
            "v_pred_norm": v_pred.norm().item(),
            "v_target_norm": v_target.norm().item(),
        }

        return loss, metrics

    def train_step(
        self, x: torch.Tensor, conditioning: torch.Tensor
    ) -> dict:
        """Single training step."""
        self.optimizer.zero_grad()
        loss, metrics = self.training_loss(x, conditioning)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        return metrics

    def save(self, path: Path) -> None:
        """Save model state."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_state": self.state_dict(), "config": self.config}, path
        )

    @classmethod
    def load(cls, path: Path) -> "FMGPResidualGenerator":
        """Load model state."""
        checkpoint = torch.load(path, weights_only=False)
        model = cls(checkpoint["config"])
        model.load_state_dict(checkpoint["model_state"])
        return model


class FMGPResidualEngine:
    """
    Wrapper for FM-GP that integrates with the existing residual generator interface.
    """

    def __init__(self, config: Optional[FMGPConfig] = None, device: str = "cpu"):
        self.config = config or FMGPConfig()
        self.device = device
        self.model = FMGPResidualGenerator(self.config).to(device)

    def sample_paths(
        self,
        horizon: int,
        num_paths: int,
        vol_path: Sequence[float],
        regime_strength: float,
        mm_features: Optional[Sequence[float]] = None,
    ) -> List[List[float]]:
        """
        Sample residual paths (compatible with existing interface).

        Args:
            horizon: Forecast horizon
            num_paths: Number of paths to sample
            vol_path: Volatility path forecast
            regime_strength: Regime strength scalar
            mm_features: (gamma_squeeze, inventory_unwind, basis_pressure)

        Returns:
            List of residual paths
        """
        self.model.eval()

        # Build conditioning vector
        mm_gamma, mm_inventory, mm_basis = mm_features or (0.0, 0.0, 0.0)

        # Pad vol_path to horizon length
        vol_padded = list(vol_path) + [vol_path[-1]] * max(0, horizon - len(vol_path))
        vol_padded = vol_padded[:horizon]

        # Conditioning: [mean_vol, regime_strength, mm_gamma, mm_inventory, mm_basis, ...]
        mean_vol = sum(vol_padded) / len(vol_padded)
        conditioning = [
            mean_vol,
            regime_strength,
            mm_gamma,
            mm_inventory,
            mm_basis,
        ]

        # Pad to cond_dim
        while len(conditioning) < self.config.cond_dim:
            conditioning.append(0.0)
        conditioning = conditioning[: self.config.cond_dim]

        # Convert to tensor
        cond_tensor = torch.tensor([conditioning], device=self.device, dtype=torch.float32)

        # Sample
        with torch.no_grad():
            paths_tensor = self.model.sample(
                cond_tensor, n_paths=num_paths, device=self.device
            )  # (1, num_paths, horizon)

        # Convert back to list
        paths_array = paths_tensor[0].cpu().numpy()  # (num_paths, horizon)

        # IMPORTANT: Scale residuals by volatility path (like legacy ResidualGenerator)
        # Legacy scales by: std + vol_path * 0.05
        # FM-GP residuals are in normalized space, so we need similar scaling
        paths_list = []
        for path in paths_array:
            scaled_path = []
            for t in range(horizon):
                vol_scale = vol_padded[min(t, len(vol_padded) - 1)] * 0.05
                # Scale the FM-GP residual by volatility to match legacy magnitude
                # Increased multiplier from 20x to 50x for better spread matching
                scaled_residual = path[t] * (1.0 + vol_scale * 50.0)
                scaled_path.append(scaled_residual)
            paths_list.append(scaled_path)

        return paths_list

    def train_on_historical(
        self,
        residual_sequences: Sequence[Sequence[float]],
        conditioning_sequences: Sequence[Sequence[float]],
        epochs: int = 10,
        batch_size: int = 32,
    ) -> dict:
        """
        Train FM-GP on historical residual sequences.

        Args:
            residual_sequences: List of historical residual paths
            conditioning_sequences: List of conditioning feature vectors
            epochs: Number of training epochs
            batch_size: Batch size

        Returns:
            Training metrics history
        """
        self.model.train()

        # Prepare data
        n_samples = len(residual_sequences)
        indices = list(range(n_samples))

        # Convert to tensors
        residuals_list = []
        cond_list = []

        for res_seq, cond_seq in zip(residual_sequences, conditioning_sequences):
            # Pad/truncate to horizon
            res_padded = list(res_seq) + [0.0] * max(0, self.config.horizon - len(res_seq))
            res_padded = res_padded[: self.config.horizon]
            residuals_list.append(res_padded)

            # Pad conditioning
            cond_padded = list(cond_seq) + [0.0] * max(0, self.config.cond_dim - len(cond_seq))
            cond_padded = cond_padded[: self.config.cond_dim]
            cond_list.append(cond_padded)

        residuals_tensor = torch.tensor(residuals_list, device=self.device, dtype=torch.float32)
        cond_tensor = torch.tensor(cond_list, device=self.device, dtype=torch.float32)

        # Training loop
        metrics_history = {"loss": [], "v_pred_norm": [], "v_target_norm": []}

        for epoch in range(epochs):
            import random

            random.shuffle(indices)

            epoch_metrics = {k: [] for k in metrics_history.keys()}

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_indices = indices[start:end]

                batch_res = residuals_tensor[batch_indices]
                batch_cond = cond_tensor[batch_indices]

                metrics = self.model.train_step(batch_res, batch_cond)

                for k, v in metrics.items():
                    epoch_metrics[k].append(v)

            # Average epoch metrics
            for k in metrics_history.keys():
                avg = sum(epoch_metrics[k]) / max(len(epoch_metrics[k]), 1)
                metrics_history[k].append(avg)

        return metrics_history

    def save(self, path: Path) -> None:
        """Save model."""
        self.model.save(path)

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "FMGPResidualEngine":
        """Load model."""
        model = FMGPResidualGenerator.load(path)
        engine = cls(config=model.config, device=device)
        engine.model = model
        return engine
