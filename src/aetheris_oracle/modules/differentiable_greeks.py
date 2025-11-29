"""
Differentiable Greeks Market Maker Engine with Learned Attention.

Implements end-to-end learnable MM state computation using:
- Differentiable Black-Scholes for Greeks
- Multi-head attention over strike dimension
- Learned aggregation replacing hand-crafted indices

Based on "Deep Hedging" (Bühler et al., 2019) differentiable derivatives framework.
"""

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm


def _get_greeks_embed_dim() -> int:
    """Get Greeks embed_dim from tuned env var or use default."""
    return int(os.getenv("TUNING_GREEKS_EMBED_DIM", "64"))


def _get_greeks_n_heads() -> int:
    """Get Greeks n_heads from tuned env var or use default."""
    return int(os.getenv("TUNING_GREEKS_HEADS", "4"))


def _get_greeks_learning_rate() -> float:
    """Get Greeks learning_rate from tuned env var or use default."""
    return float(os.getenv("TUNING_GREEKS_LR", "0.001"))


def _get_greeks_epochs() -> int:
    """Get Greeks training epochs from tuned env var or use default."""
    return int(os.getenv("TUNING_GREEKS_EPOCHS", "30"))


@dataclass
class DifferentiableGreeksConfig:
    """Configuration for Differentiable Greeks MM Engine."""

    n_strikes: int = 20  # Number of strike points
    n_expiries: int = 4  # Number of expiries
    embed_dim: int = field(default_factory=_get_greeks_embed_dim)
    n_heads: int = field(default_factory=_get_greeks_n_heads)
    hidden_dim: int = 128
    mm_state_dim: int = 64  # Output MM embedding dimension
    learning_rate: float = field(default_factory=_get_greeks_learning_rate)
    default_epochs: int = field(default_factory=_get_greeks_epochs)
    artifact_path: str = "artifacts/diff_greeks_state.pt"


class DifferentiableBlackScholes(nn.Module):
    """
    Differentiable Black-Scholes Greeks computation.

    Allows gradients to flow through options pricing for end-to-end learning.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        spot: torch.Tensor,
        strike: torch.Tensor,
        tau: torch.Tensor,
        iv: torch.Tensor,
        r: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Black-Scholes Greeks differentiably.

        Args:
            spot: Spot price (batch,)
            strike: Strike prices (batch, n_strikes)
            tau: Time to expiry in years (batch, n_strikes)
            iv: Implied volatility (batch, n_strikes)
            r: Risk-free rate

        Returns:
            Dict with delta, gamma, vanna, charm
        """
        # Ensure positive values
        iv = torch.clamp(iv, min=1e-4)
        tau = torch.clamp(tau, min=1e-4)

        # Expand spot to match strike dimensions
        if spot.dim() == 1 and strike.dim() == 2:
            spot = spot.unsqueeze(1)

        # d1 and d2
        d1 = (torch.log(spot / strike) + (r + 0.5 * iv ** 2) * tau) / (
            iv * torch.sqrt(tau)
        )
        d2 = d1 - iv * torch.sqrt(tau)

        # Standard normal CDF and PDF
        # Using torch operations that support autograd
        sqrt_2pi = math.sqrt(2 * math.pi)
        pdf_d1 = torch.exp(-0.5 * d1 ** 2) / sqrt_2pi

        # Delta (call)
        delta = 0.5 * (1 + torch.erf(d1 / math.sqrt(2)))

        # Gamma
        gamma = pdf_d1 / (spot * iv * torch.sqrt(tau))

        # Vanna: ∂delta/∂σ
        vanna = -pdf_d1 * d2 / iv

        # Charm: ∂delta/∂t
        charm = -pdf_d1 * (
            r / (iv * torch.sqrt(tau))
            - (d2 * iv) / (2 * tau)
        )

        return {
            "delta": delta,
            "gamma": gamma,
            "vanna": vanna,
            "charm": charm,
            "d1": d1,
            "d2": d2,
        }


class StrikeAttention(nn.Module):
    """
    Multi-head attention over strike dimension.

    Learns which strikes are most important for MM positioning.
    """

    def __init__(self, embed_dim: int, n_heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, n_heads, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, n_strikes, embed_dim)

        Returns:
            attended: (batch, n_strikes, embed_dim)
            weights: (batch, n_heads, n_strikes, n_strikes) attention weights
        """
        attended, weights = self.attention(x, x, x)
        attended = self.layer_norm(attended + x)  # Residual connection
        return attended, weights


class DifferentiableMMEngine(nn.Module):
    """
    End-to-end learnable Market Maker Engine.

    Replaces hand-crafted indices with learned representations via:
    1. Differentiable Greeks computation
    2. Attention over strikes to learn importance
    3. Aggregation to MM state embedding
    """

    def __init__(self, config: DifferentiableGreeksConfig):
        super().__init__()
        self.config = config

        # Differentiable Black-Scholes
        self.bs_layer = DifferentiableBlackScholes()

        # Strike encoder: map Greeks + OI to embedding
        greeks_dim = 4  # delta, gamma, vanna, charm
        self.strike_encoder = nn.Sequential(
            nn.Linear(greeks_dim + 1, config.embed_dim),  # +1 for OI
            nn.GELU(),
            nn.LayerNorm(config.embed_dim),
        )

        # Attention over strikes
        self.strike_attention = StrikeAttention(config.embed_dim, config.n_heads)

        # Global context encoder (funding, basis, etc.)
        self.context_encoder = nn.Sequential(
            nn.Linear(4, config.hidden_dim),  # funding, basis, skew, imbalance
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.embed_dim),
        )

        # Aggregator: strikes + context → MM state
        self.aggregator = nn.Sequential(
            nn.Linear(config.embed_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.mm_state_dim),
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=config.learning_rate)

    def forward(
        self,
        spot: torch.Tensor,
        strikes: torch.Tensor,
        taus: torch.Tensor,
        ivs: torch.Tensor,
        ois: torch.Tensor,
        funding: torch.Tensor,
        basis: torch.Tensor,
        skew: torch.Tensor,
        imbalance: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute MM state embedding.

        Args:
            spot: (batch,) spot prices
            strikes: (batch, n_strikes) strike prices
            taus: (batch, n_strikes) times to expiry
            ivs: (batch, n_strikes) implied vols
            ois: (batch, n_strikes) open interests
            funding: (batch,) funding rates
            basis: (batch,) basis
            skew: (batch,) skew metric
            imbalance: (batch,) order imbalance

        Returns:
            mm_state: (batch, mm_state_dim) MM embedding
            attention_weights: (batch, n_heads, n_strikes, n_strikes)
        """
        batch_size = spot.shape[0]

        # Compute Greeks differentiably
        greeks = self.bs_layer(spot, strikes, taus, ivs)

        # Stack Greeks: (batch, n_strikes, 4)
        greeks_stacked = torch.stack(
            [greeks["delta"], greeks["gamma"], greeks["vanna"], greeks["charm"]],
            dim=-1,
        )

        # Weight by OI: (batch, n_strikes, 5)
        strike_features = torch.cat([greeks_stacked, ois.unsqueeze(-1)], dim=-1)

        # Encode strikes: (batch, n_strikes, embed_dim)
        strike_embeddings = self.strike_encoder(strike_features)

        # Attention over strikes
        attended_strikes, attn_weights = self.strike_attention(strike_embeddings)

        # Aggregate strikes (mean pooling)
        strike_agg = attended_strikes.mean(dim=1)  # (batch, embed_dim)

        # Encode global context
        context = torch.stack([funding, basis, skew, imbalance], dim=-1)  # (batch, 4)
        context_emb = self.context_encoder(context)  # (batch, embed_dim)

        # Concatenate and aggregate
        combined = torch.cat([strike_agg, context_emb], dim=-1)  # (batch, embed_dim*2)
        mm_state = self.aggregator(combined)  # (batch, mm_state_dim)

        return mm_state, attn_weights

    def training_loss(
        self,
        mm_state: torch.Tensor,
        target_returns: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Training loss: predict future returns from MM state.

        In practice, would use full forecast pipeline loss.

        Args:
            mm_state: (batch, mm_state_dim) MM embeddings
            target_returns: (batch,) future returns to predict

        Returns:
            loss, metrics dict
        """
        # Simple prediction head for demonstration
        # In production, mm_state feeds into full forecast model
        pred_returns = mm_state[:, 0]  # Use first dimension as prediction

        loss = F.mse_loss(pred_returns, target_returns)

        metrics = {"loss": loss.item()}

        return loss, metrics

    def train_step(
        self,
        spot: torch.Tensor,
        strikes: torch.Tensor,
        taus: torch.Tensor,
        ivs: torch.Tensor,
        ois: torch.Tensor,
        funding: torch.Tensor,
        basis: torch.Tensor,
        skew: torch.Tensor,
        imbalance: torch.Tensor,
        target_returns: torch.Tensor,
    ) -> Dict[str, float]:
        """Single training step."""
        self.optimizer.zero_grad()

        mm_state, _ = self.forward(
            spot, strikes, taus, ivs, ois, funding, basis, skew, imbalance
        )

        loss, metrics = self.training_loss(mm_state, target_returns)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

        return metrics

    def save(self, path: Path) -> None:
        """Save model state."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": self.state_dict(), "config": self.config}, path)

    @classmethod
    def load(cls, path: Path) -> "DifferentiableMMEngine":
        """Load model state."""
        checkpoint = torch.load(path, weights_only=False)
        model = cls(checkpoint["config"])
        model.load_state_dict(checkpoint["model_state"])
        return model


class DifferentiableMMEngineWrapper:
    """
    Wrapper for Differentiable MM Engine compatible with existing interface.
    """

    def __init__(self, config: Optional[DifferentiableGreeksConfig] = None, device: str = "cpu"):
        self.config = config or DifferentiableGreeksConfig()
        self.device = device
        self.model = DifferentiableMMEngine(self.config).to(device)

    def compute_indices(
        self,
        spot: float,
        iv_term_structure: Dict[str, float],
        funding_rate: float,
        basis: float,
        order_imbalance: float,
        skew: float,
        option_oi_dict: Optional[Dict[float, float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute MM state (compatible with existing interface).

        Args:
            spot: Spot price
            iv_term_structure: {'iv_7d_atm': ..., 'iv_30d_atm': ...}
            funding_rate: Funding rate
            basis: Basis
            order_imbalance: Order imbalance
            skew: Skew metric
            option_oi_dict: {strike: OI} optional

        Returns:
            mm_state: (mm_state_dim,) MM embedding tensor
            attention_weights: Attention weights for interpretability
        """
        self.model.eval()

        # Generate strike grid around spot
        strike_range = torch.linspace(spot * 0.8, spot * 1.2, self.config.n_strikes)

        # Map IV term structure to strikes (simplified)
        iv_7d = iv_term_structure.get("iv_7d_atm", 0.5)
        iv_30d = iv_term_structure.get("iv_30d_atm", iv_7d)

        # Linear interpolation for different expiries
        ivs_list = []
        taus_list = []
        for i in range(self.config.n_expiries):
            tau = 7 * (i + 1) / 365.0  # 7d, 14d, 21d, 28d
            iv = iv_7d + (iv_30d - iv_7d) * (tau * 365 / 30)
            ivs_list.append(torch.full((self.config.n_strikes,), iv))
            taus_list.append(torch.full((self.config.n_strikes,), tau))

        # Flatten to (n_strikes * n_expiries,)
        strikes_flat = strike_range.repeat(self.config.n_expiries)
        ivs_flat = torch.cat(ivs_list)
        taus_flat = torch.cat(taus_list)

        # Generate synthetic OI if not provided
        if option_oi_dict is None:
            # Peaked around ATM
            distances = torch.abs(strikes_flat - spot) / spot
            ois_flat = torch.exp(-10 * distances ** 2)  # Gaussian around spot
        else:
            # Map provided OI
            ois_flat = torch.tensor(
                [option_oi_dict.get(float(s), 0.0) for s in strikes_flat]
            )

        # Convert to tensors
        spot_t = torch.tensor([spot], device=self.device, dtype=torch.float32)
        strikes_t = strikes_flat.unsqueeze(0).to(self.device)
        taus_t = taus_flat.unsqueeze(0).to(self.device)
        ivs_t = ivs_flat.unsqueeze(0).to(self.device)
        ois_t = ois_flat.unsqueeze(0).to(self.device)

        # Sanitize inputs - replace NaN with defaults to avoid propagation
        safe_funding = 0.0 if (funding_rate is None or (isinstance(funding_rate, float) and math.isnan(funding_rate))) else funding_rate
        safe_basis = 0.0 if (basis is None or (isinstance(basis, float) and math.isnan(basis))) else basis
        safe_skew = 0.0 if (skew is None or (isinstance(skew, float) and math.isnan(skew))) else skew
        safe_imbalance = 0.0 if (order_imbalance is None or (isinstance(order_imbalance, float) and math.isnan(order_imbalance))) else order_imbalance

        funding_t = torch.tensor([safe_funding], device=self.device, dtype=torch.float32)
        basis_t = torch.tensor([safe_basis], device=self.device, dtype=torch.float32)
        skew_t = torch.tensor([safe_skew], device=self.device, dtype=torch.float32)
        imbalance_t = torch.tensor([safe_imbalance], device=self.device, dtype=torch.float32)

        with torch.no_grad():
            mm_state, attn_weights = self.model(
                spot_t,
                strikes_t,
                taus_t,
                ivs_t,
                ois_t,
                funding_t,
                basis_t,
                skew_t,
                imbalance_t,
            )

        return mm_state.squeeze(0), attn_weights.squeeze(0)

    def save(self, path: Path) -> None:
        """Save model."""
        self.model.save(path)

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "DifferentiableMMEngineWrapper":
        """Load model."""
        model = DifferentiableMMEngine.load(path)
        wrapper = cls(config=model.config, device=device)
        wrapper.model = model
        return wrapper
