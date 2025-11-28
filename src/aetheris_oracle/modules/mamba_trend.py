"""
MambaTS Trend Backbone (Simplified).

Wrapper around Mamba state-space models for trend forecasting.
Uses mamba-ssm library for core Mamba blocks.

Based on:
- "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu & Dao, 2024)
- "MambaTS: Improved Selective State Space Models for Long-term Time Series Forecasting" (Cai et al., 2024)

Note: Full VAST (Variable-Aware Scan) implementation would require mamba-ssm>=1.2.0
and significant additional complexity. This provides a simplified but functional version.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# Note: mamba_ssm may not be installed; provide fallback
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba-ssm not installed. Using simplified implementation.")


@dataclass
class MambaTrendConfig:
    """Configuration for Mamba-based trend model."""

    d_model: int = 128
    n_layers: int = 4
    horizon: int = 90
    input_dim: int = 5  # returns + regime features
    learning_rate: float = 0.001
    artifact_path: str = "artifacts/mamba_trend_state.pt"


class SimplifiedMambaBlock(nn.Module):
    """
    Simplified Mamba block (fallback when mamba-ssm not available).

    Uses linear state-space model approximation.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        # State-space parameters (learned)
        self.A = nn.Parameter(torch.randn(d_model, d_model) * 0.01)
        self.B = nn.Parameter(torch.randn(d_model, d_model) * 0.01)
        self.C = nn.Parameter(torch.randn(d_model, d_model) * 0.01)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape

        # Initialize hidden state
        h = torch.zeros(batch, d_model, device=x.device)

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]

            # State update: h_{t+1} = Ah_t + Bx_t
            h = torch.tanh(h @ self.A.T + x_t @ self.B.T)

            # Output: y_t = Ch_t
            y_t = h @ self.C.T

            outputs.append(y_t)

        outputs = torch.stack(outputs, dim=1)  # (batch, seq_len, d_model)

        return self.norm(outputs + x)  # Residual connection


class MambaTrendBackbone(nn.Module):
    """
    Mamba-based trend forecasting backbone.

    Replaces AR + RNN + Transformer ensemble with unified SSM.
    """

    def __init__(self, config: MambaTrendConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Linear(config.input_dim, config.d_model)

        # Mamba blocks
        if MAMBA_AVAILABLE:
            self.mamba_blocks = nn.ModuleList([
                Mamba(d_model=config.d_model) for _ in range(config.n_layers)
            ])
        else:
            self.mamba_blocks = nn.ModuleList([
                SimplifiedMambaBlock(d_model=config.d_model) for _ in range(config.n_layers)
            ])

        # Multi-scale output heads
        self.output_heads = nn.ModuleDict({
            "short": nn.Linear(config.d_model, config.horizon),  # 1-3 day
            "medium": nn.Linear(config.d_model, config.horizon),  # 4-7 day
            "long": nn.Linear(config.d_model, config.horizon),  # 8-14 day
        })

        # Meta-weighting network
        self.meta_weight_net = nn.Sequential(
            nn.Linear(config.d_model, 64),
            nn.GELU(),
            nn.Linear(64, 3),  # Weights for short/medium/long
            nn.Softmax(dim=-1),
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=config.learning_rate)

    def forward(
        self, x: torch.Tensor, horizon: Optional[int] = None
    ) -> torch.Tensor:
        """
        Forecast trend.

        Args:
            x: (batch, seq_len, input_dim) input sequence
            horizon: Optional horizon override

        Returns:
            (batch, horizon) trend forecasts
        """
        if horizon is None:
            horizon = self.config.horizon

        # Project input
        x = self.input_proj(x)  # (batch, seq_len, d_model)

        # Pass through Mamba blocks
        for mamba_block in self.mamba_blocks:
            x = mamba_block(x)

        # Get final hidden state (last time step)
        h_final = x[:, -1, :]  # (batch, d_model)

        # Multi-scale predictions
        pred_short = self.output_heads["short"](h_final)[:, :horizon]
        pred_medium = self.output_heads["medium"](h_final)[:, :horizon]
        pred_long = self.output_heads["long"](h_final)[:, :horizon]

        # Meta-weighting
        weights = self.meta_weight_net(h_final)  # (batch, 3)

        # Weighted combination
        pred_combined = (
            weights[:, 0:1] * pred_short
            + weights[:, 1:2] * pred_medium
            + weights[:, 2:3] * pred_long
        )

        return pred_combined

    def training_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Training loss."""
        return F.mse_loss(predictions, targets)

    def train_step(
        self, x: torch.Tensor, targets: torch.Tensor
    ) -> dict:
        """Single training step."""
        self.optimizer.zero_grad()

        predictions = self.forward(x, horizon=targets.shape[1])
        loss = self.training_loss(predictions, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

        return {"loss": loss.item()}

    def save(self, path: Path) -> None:
        """Save model state."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": self.state_dict(), "config": self.config}, path)

    @classmethod
    def load(cls, path: Path) -> "MambaTrendBackbone":
        """Load model state."""
        checkpoint = torch.load(path, weights_only=False)
        model = cls(checkpoint["config"])
        model.load_state_dict(checkpoint["model_state"])
        return model


class MambaTrendWrapper:
    """
    Wrapper for Mamba trend model compatible with existing interface.
    """

    def __init__(self, config: Optional[MambaTrendConfig] = None, device: str = "cpu"):
        self.config = config or MambaTrendConfig()
        self.device = device
        self.model = MambaTrendBackbone(self.config).to(device)

    def predict_trend(
        self,
        normalized_closes: Sequence[float],
        horizon: int,
        regime_strength: float = 1.0,
    ) -> List[float]:
        """
        Predict trend (compatible with existing interface).

        Args:
            normalized_closes: Historical normalized closes
            horizon: Forecast horizon
            regime_strength: Regime strength scalar

        Returns:
            List of trend forecasts
        """
        self.model.eval()

        # Compute returns
        closes_list = list(normalized_closes)
        returns = [closes_list[i] - closes_list[i - 1] for i in range(1, len(closes_list))]

        # Use last 20 returns (or available)
        lookback = min(20, len(returns))
        recent_returns = returns[-lookback:]

        # Build input features: [return, regime, t, t^2, const]
        input_features = []
        for i, ret in enumerate(recent_returns):
            t_norm = i / lookback
            feat = [ret, regime_strength, t_norm, t_norm ** 2, 1.0]
            input_features.append(feat)

        # Convert to tensor
        x = torch.tensor([input_features], device=self.device, dtype=torch.float32)

        # Forecast
        with torch.no_grad():
            trend = self.model(x, horizon=horizon)

        # Convert incremental returns to cumulative trend from last close
        # Model outputs day-to-day returns, need to accumulate for price levels
        trend_list = trend[0].cpu().numpy().tolist()
        cumulative = 0.0
        cumulative_trend = []
        for i in range(horizon):
            cumulative += trend_list[i]
            cumulative_trend.append(closes_list[-1] + cumulative)

        return cumulative_trend

    def save(self, path: Path) -> None:
        """Save model."""
        self.model.save(path)

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "MambaTrendWrapper":
        """Load model."""
        model = MambaTrendBackbone.load(path)
        wrapper = cls(config=model.config, device=device)
        wrapper.model = model
        return wrapper
