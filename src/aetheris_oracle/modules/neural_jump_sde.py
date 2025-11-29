"""
Neural Jump Stochastic Differential Equations.

Implements end-to-end learned jump-diffusion processes from:
"Neural Jump SDEs" (Jia & Benson, NeurIPS 2019)
"Neural SDEs as Infinite-Dimensional GANs" (Kidger et al., NeurIPS 2021)

Learns: intensity function λ(t), jump size distribution, diffusion coefficient.
"""

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_jump_hidden_dim() -> int:
    """Get Neural Jump hidden_dim from tuned env var or use default."""
    return int(os.getenv("TUNING_JUMP_HIDDEN_DIM", "128"))


def _get_jump_n_layers() -> int:
    """Get Neural Jump n_layers from tuned env var or use default."""
    return int(os.getenv("TUNING_JUMP_LAYERS", "3"))


def _get_jump_learning_rate() -> float:
    """Get Neural Jump learning_rate from tuned env var or use default."""
    return float(os.getenv("TUNING_JUMP_LR", "0.001"))


def _get_jump_epochs() -> int:
    """Get Neural Jump training epochs from tuned env var or use default."""
    return int(os.getenv("TUNING_JUMP_EPOCHS", "50"))


@dataclass
class NeuralJumpSDEConfig:
    """Configuration for Neural Jump SDE."""

    state_dim: int = 1  # Price (or log-price)
    cond_dim: int = 10  # Conditioning features
    hidden_dim: int = field(default_factory=_get_jump_hidden_dim)
    n_layers: int = field(default_factory=_get_jump_n_layers)
    dt: float = 1.0  # Daily steps (matching training data resolution)
    learning_rate: float = field(default_factory=_get_jump_learning_rate)
    default_epochs: int = field(default_factory=_get_jump_epochs)
    artifact_path: str = "artifacts/neural_jump_sde_state.pt"


class IntensityNetwork(nn.Module):
    """
    Learns jump intensity λ(x, c, t).

    Intensity must be positive, so we use softplus activation.
    """

    def __init__(self, state_dim: int, cond_dim: int, hidden_dim: int, n_layers: int):
        super().__init__()

        layers = [nn.Linear(state_dim + cond_dim, hidden_dim), nn.Softplus()]

        for _ in range(n_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.Softplus(),
                nn.LayerNorm(hidden_dim),
            ])

        layers.extend([nn.Linear(hidden_dim, 1), nn.Softplus()])  # Output intensity

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """
        Compute intensity.

        Args:
            x: (batch, state_dim) current state
            conditioning: (batch, cond_dim) conditioning features

        Returns:
            (batch, 1) intensity values
        """
        inp = torch.cat([x, conditioning], dim=-1)
        return self.net(inp)


class JumpSizeNetwork(nn.Module):
    """
    Learns jump size distribution parameters.

    Outputs (mean, log_std) for Gaussian jump distribution.
    """

    def __init__(self, state_dim: int, cond_dim: int, hidden_dim: int, n_layers: int):
        super().__init__()

        layers = [nn.Linear(state_dim + cond_dim, hidden_dim), nn.GELU()]

        for _ in range(n_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            ])

        layers.append(nn.Linear(hidden_dim, 2))  # mean, log_std

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute jump size distribution parameters.

        Args:
            x: (batch, state_dim) current state
            conditioning: (batch, cond_dim) conditioning features

        Returns:
            mean, log_std tensors (batch, 1)
        """
        inp = torch.cat([x, conditioning], dim=-1)
        out = self.net(inp)
        mean, log_std = out[:, :1], out[:, 1:]
        return mean, log_std


class DiffusionNetwork(nn.Module):
    """
    Learns continuous diffusion dynamics.

    Outputs (drift, diffusion_coef).
    """

    def __init__(self, state_dim: int, cond_dim: int, hidden_dim: int, n_layers: int):
        super().__init__()

        layers = [nn.Linear(state_dim + cond_dim, hidden_dim), nn.GELU()]

        for _ in range(n_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            ])

        layers.append(nn.Linear(hidden_dim, 2 * state_dim))  # drift + diffusion

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute drift and diffusion coefficient.

        Args:
            x: (batch, state_dim) current state
            conditioning: (batch, cond_dim) conditioning features

        Returns:
            drift, diffusion_coef tensors (batch, state_dim)
        """
        inp = torch.cat([x, conditioning], dim=-1)
        out = self.net(inp)
        drift = out[:, : x.shape[-1]]
        diffusion = F.softplus(out[:, x.shape[-1] :])  # Positive diffusion
        return drift, diffusion


class NeuralJumpSDE(nn.Module):
    """
    Neural Jump Stochastic Differential Equation.

    dx = μ(x,c) dt + σ(x,c) dW + J(x,c) dN

    where:
    - μ: drift (learned)
    - σ: diffusion (learned)
    - J: jump size (learned)
    - N: Poisson process with intensity λ(x,c) (learned)
    """

    def __init__(self, config: NeuralJumpSDEConfig):
        super().__init__()
        self.config = config

        # Neural networks
        self.intensity_net = IntensityNetwork(
            config.state_dim, config.cond_dim, config.hidden_dim, config.n_layers
        )

        self.jump_size_net = JumpSizeNetwork(
            config.state_dim, config.cond_dim, config.hidden_dim, config.n_layers
        )

        self.diffusion_net = DiffusionNetwork(
            config.state_dim, config.cond_dim, config.hidden_dim, config.n_layers
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=config.learning_rate)

    def forward(
        self, x0: torch.Tensor, conditioning: torch.Tensor, horizon: int
    ) -> Tuple[torch.Tensor, List, List]:
        """
        Simulate jump-diffusion paths.

        Args:
            x0: (batch, state_dim) initial state
            conditioning: (batch, cond_dim) conditioning features
            horizon: Forecast horizon (in days)

        Returns:
            paths: (batch, n_steps, state_dim)
            jump_times: List of (step, jump_occurred_mask)
            jump_sizes: List of jump magnitudes
        """
        batch_size, state_dim = x0.shape
        n_steps = int(horizon / self.config.dt)
        dt = self.config.dt
        device = x0.device

        paths = [x0]
        jump_times = []
        jump_sizes = []

        x = x0

        for step in range(n_steps):
            # Compute intensity
            intensity = self.intensity_net(x, conditioning).squeeze(-1)  # (batch,)

            # Sample jump occurrence (thinning algorithm)
            jump_prob = intensity * dt
            jump_occurred = torch.rand(batch_size, device=device) < jump_prob

            # Sample jump size where jumps occurred
            jump_mean, jump_log_std = self.jump_size_net(x, conditioning)
            jump_std = torch.exp(jump_log_std)
            jump_noise = torch.randn_like(jump_mean)
            jump = jump_mean + jump_std * jump_noise
            jump = jump * jump_occurred.float().unsqueeze(-1)  # Mask non-jumps

            # Diffusion dynamics
            drift, diff_coef = self.diffusion_net(x, conditioning)
            diffusion_noise = torch.randn_like(x) * math.sqrt(dt)
            diffusion_term = diff_coef * diffusion_noise

            # Update state: dx = drift*dt + diffusion + jump
            dx = drift * dt + diffusion_term + jump
            x = x + dx

            paths.append(x)

            # Record jumps
            if jump_occurred.any():
                jump_times.append((step, jump_occurred))
                jump_sizes.append(jump[jump_occurred])

        paths_tensor = torch.stack(paths, dim=1)  # (batch, n_steps+1, state_dim)

        return paths_tensor, jump_times, jump_sizes

    def training_loss(
        self,
        observed_paths: torch.Tensor,
        conditioning: torch.Tensor,
        jump_indicators: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Training loss via simulation-based inference.

        Uses a combination of:
        1. Trajectory matching loss
        2. Jump detection loss (if jump_indicators provided)

        Args:
            observed_paths: (batch, n_steps, state_dim) historical paths
            conditioning: (batch, cond_dim) conditioning features
            jump_indicators: (batch, n_steps) binary indicators of jumps

        Returns:
            loss, metrics dict
        """
        batch_size, n_steps, state_dim = observed_paths.shape
        device = observed_paths.device

        # Simulate paths
        x0 = observed_paths[:, 0, :]
        simulated, jump_times, _ = self.forward(
            x0, conditioning, horizon=n_steps * self.config.dt
        )

        # Trajectory matching loss (simplified - using endpoint + intermediate points)
        simulated_trimmed = simulated[:, : n_steps + 1, :]
        trajectory_loss = F.mse_loss(simulated_trimmed[:, 1:, :], observed_paths[:, 1:, :])

        total_loss = trajectory_loss
        metrics = {"trajectory_loss": trajectory_loss.item()}

        # Jump detection loss (if provided)
        if jump_indicators is not None:
            # Compute intensity at each step
            jump_loss = 0.0
            for step in range(n_steps):
                x_step = observed_paths[:, step, :]
                intensity = self.intensity_net(x_step, conditioning).squeeze(-1)

                # Binary cross-entropy with jump indicators
                jump_prob = torch.sigmoid(intensity * self.config.dt - 0.5)
                jump_target = jump_indicators[:, step].float()

                jump_loss += F.binary_cross_entropy(jump_prob, jump_target)

            jump_loss /= n_steps
            total_loss = total_loss + 0.1 * jump_loss
            metrics["jump_loss"] = jump_loss.item()

        metrics["total_loss"] = total_loss.item()

        return total_loss, metrics

    def train_step(
        self,
        observed_paths: torch.Tensor,
        conditioning: torch.Tensor,
        jump_indicators: Optional[torch.Tensor] = None,
    ) -> dict:
        """Single training step."""
        self.optimizer.zero_grad()
        loss, metrics = self.training_loss(observed_paths, conditioning, jump_indicators)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        return metrics

    def save(self, path: Path) -> None:
        """Save model state."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": self.state_dict(), "config": self.config}, path)

    @classmethod
    def load(cls, path: Path) -> "NeuralJumpSDE":
        """Load model state."""
        checkpoint = torch.load(path, weights_only=False)
        model = cls(checkpoint["config"])
        model.load_state_dict(checkpoint["model_state"])
        return model


class NeuralJumpSDEEngine:
    """
    Wrapper for Neural Jump SDE that integrates with existing interface.
    """

    def __init__(self, config: Optional[NeuralJumpSDEConfig] = None, device: str = "cpu"):
        self.config = config or NeuralJumpSDEConfig()
        self.device = device
        self.model = NeuralJumpSDE(self.config).to(device)

    def sample_path(
        self,
        horizon: int,
        vol_path: Sequence[float],
        narrative_score: float,
        gamma_squeeze: float,
        regime_strength: float = 1.0,
        basis_pressure: float = 0.0,
    ) -> List[float]:
        """
        Sample jump path (compatible with existing interface).

        Args:
            horizon: Forecast horizon in days
            vol_path: Volatility path
            narrative_score: Narrative intensity
            gamma_squeeze: MM gamma squeeze index
            regime_strength: Regime strength
            basis_pressure: Basis pressure

        Returns:
            Jump path (cumulative jumps per step)
        """
        self.model.eval()

        # Build conditioning
        mean_vol = sum(vol_path) / len(vol_path) if vol_path else 0.5
        conditioning = [
            mean_vol,
            narrative_score,
            gamma_squeeze,
            regime_strength,
            basis_pressure,
        ]

        # Pad to cond_dim
        while len(conditioning) < self.config.cond_dim:
            conditioning.append(0.0)
        conditioning = conditioning[: self.config.cond_dim]

        cond_tensor = torch.tensor(
            [conditioning], device=self.device, dtype=torch.float32
        )

        # Initial state (zero)
        x0 = torch.zeros(1, self.config.state_dim, device=self.device)

        # Simulate
        with torch.no_grad():
            paths, _, _ = self.model.forward(x0, cond_tensor, horizon)

        # Extract path
        path_array = paths[0, :, 0].cpu().numpy()  # (n_steps+1,)

        # Convert to per-step jumps (differences)
        jumps = np.diff(path_array, prepend=0.0)

        # Aggregate to daily jumps
        n_steps_per_day = int(1.0 / self.config.dt)
        daily_jumps = []

        for day in range(horizon):
            start_idx = day * n_steps_per_day
            end_idx = (day + 1) * n_steps_per_day
            day_jump = np.sum(jumps[start_idx:end_idx])
            daily_jumps.append(float(day_jump))

        return daily_jumps

    def sample_sde_paths(
        self,
        x0: torch.Tensor,
        conditioning: List[float],
        horizon: int,
        vol_path: Sequence[float],
    ) -> torch.Tensor:
        """
        Sample multiple jump paths at once (batch sampling for efficiency).

        Args:
            x0: Initial states (num_paths,) or (num_paths, state_dim)
            conditioning: Feature vector [regime_vol, iv_level, gamma, inventory, basis, narrative]
            horizon: Forecast horizon in days
            vol_path: Volatility path

        Returns:
            Jump paths tensor of shape (num_paths, horizon)
        """
        self.model.eval()

        num_paths = x0.shape[0]

        # Pad conditioning to cond_dim
        cond_list = list(conditioning)
        while len(cond_list) < self.config.cond_dim:
            cond_list.append(0.0)
        cond_list = cond_list[: self.config.cond_dim]

        # Broadcast conditioning to all paths
        cond_tensor = torch.tensor(
            [cond_list] * num_paths, device=self.device, dtype=torch.float32
        )  # (num_paths, cond_dim)

        # Ensure x0 has correct shape
        if x0.dim() == 1:
            x0_expanded = x0.unsqueeze(1)  # (num_paths, 1)
        else:
            x0_expanded = x0  # Already (num_paths, state_dim)

        # Simulate all paths at once
        with torch.no_grad():
            paths, _, _ = self.model.forward(x0_expanded, cond_tensor, horizon)

        # paths: (num_paths, n_steps+1, state_dim)
        # Extract first dimension and compute per-step jumps
        path_array = paths[:, :, 0]  # (num_paths, n_steps+1)

        # Convert to per-step jumps (differences along time axis)
        jumps = torch.diff(
            path_array, dim=1, prepend=torch.zeros(num_paths, 1, device=self.device)
        )  # (num_paths, n_steps+1)

        # Aggregate to daily jumps
        n_steps_per_day = int(1.0 / self.config.dt)
        daily_jumps_list = []

        for day in range(horizon):
            start_idx = day * n_steps_per_day
            end_idx = (day + 1) * n_steps_per_day
            day_jump = torch.sum(jumps[:, start_idx:end_idx], dim=1)  # (num_paths,)
            daily_jumps_list.append(day_jump)

        # Stack to (num_paths, horizon)
        daily_jumps = torch.stack(daily_jumps_list, dim=1)

        return daily_jumps

    def train_on_historical(
        self,
        paths: Sequence[Sequence[float]],
        conditioning_sequences: Sequence[Sequence[float]],
        jump_indicators: Optional[Sequence[Sequence[int]]] = None,
        epochs: int = 10,
        batch_size: int = 16,
    ) -> dict:
        """
        Train on historical paths.

        Args:
            paths: Historical price paths
            conditioning_sequences: Conditioning features per path
            jump_indicators: Binary jump indicators (optional)
            epochs: Training epochs
            batch_size: Batch size

        Returns:
            Training metrics
        """
        self.model.train()

        n_samples = len(paths)
        indices = list(range(n_samples))

        # Convert to tensors
        # Find max length
        max_len = max(len(p) for p in paths)

        paths_list = []
        cond_list = []
        jump_list = [] if jump_indicators else None

        for i, (path, cond) in enumerate(zip(paths, conditioning_sequences)):
            # Pad path
            padded = list(path) + [path[-1]] * (max_len - len(path))
            paths_list.append([[p] for p in padded])  # (n_steps, 1)

            # Pad conditioning
            cond_padded = list(cond) + [0.0] * (self.config.cond_dim - len(cond))
            cond_padded = cond_padded[: self.config.cond_dim]
            cond_list.append(cond_padded)

            # Pad jump indicators
            if jump_indicators:
                jumps = list(jump_indicators[i]) + [0] * (max_len - len(jump_indicators[i]))
                jump_list.append(jumps)

        paths_tensor = torch.tensor(
            paths_list, device=self.device, dtype=torch.float32
        )  # (n, steps, 1)
        cond_tensor = torch.tensor(cond_list, device=self.device, dtype=torch.float32)

        if jump_list:
            jump_tensor = torch.tensor(jump_list, device=self.device, dtype=torch.float32)
        else:
            jump_tensor = None

        # Training loop
        metrics_history = {"trajectory_loss": [], "total_loss": []}
        if jump_list:
            metrics_history["jump_loss"] = []

        for epoch in range(epochs):
            import random

            random.shuffle(indices)

            epoch_metrics = {k: [] for k in metrics_history.keys()}

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_indices = indices[start:end]

                batch_paths = paths_tensor[batch_indices]
                batch_cond = cond_tensor[batch_indices]
                batch_jumps = jump_tensor[batch_indices] if jump_tensor is not None else None

                metrics = self.model.train_step(batch_paths, batch_cond, batch_jumps)

                for k in metrics_history.keys():
                    if k in metrics:
                        epoch_metrics[k].append(metrics[k])

            # Average
            for k in metrics_history.keys():
                if epoch_metrics[k]:
                    metrics_history[k].append(np.mean(epoch_metrics[k]))

        return metrics_history

    def save(self, path: Path) -> None:
        """Save model."""
        self.model.save(path)

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "NeuralJumpSDEEngine":
        """Load model."""
        model = NeuralJumpSDE.load(path)
        engine = cls(config=model.config, device=device)
        engine.model = model
        return engine
