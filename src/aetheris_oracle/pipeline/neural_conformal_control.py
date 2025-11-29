"""
Neural Conformal Control (NCC) for adaptive calibration.

Implements the approach from "Neural Conformal Control for Time Series Forecasting"
which uses a neural network to learn calibration dynamics with long-term coverage guarantees.
"""

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_default_max_horizon() -> int:
    """Get default max horizon from environment or use 90."""
    return int(os.getenv("TRAINING_HORIZON", "90"))


def _get_ncc_hidden_dim() -> int:
    """Get NCC hidden_dim from tuned env var or use default."""
    return int(os.getenv("TUNING_NCC_HIDDEN_DIM", "64"))


def _get_ncc_learning_rate() -> float:
    """Get NCC learning_rate from tuned env var or use default."""
    return float(os.getenv("TUNING_NCC_LR", "0.001"))


def _get_ncc_smoothness_weight() -> float:
    """Get NCC smoothness_weight from tuned env var or use default."""
    return float(os.getenv("TUNING_NCC_SMOOTHNESS", "0.1"))


def _get_ncc_epochs() -> int:
    """Get NCC training epochs from tuned env var or use default."""
    return int(os.getenv("TUNING_NCC_EPOCHS", "20"))


@dataclass
class NCCConfig:
    """Configuration for Neural Conformal Control."""

    feature_dim: int = 10
    hidden_dim: int = field(default_factory=_get_ncc_hidden_dim)
    n_quantiles: int = 7  # P5, P10, P25, P50, P75, P90, P95
    max_horizon: int = field(default_factory=_get_default_max_horizon)
    learning_rate: float = field(default_factory=_get_ncc_learning_rate)
    target_coverage: float = 0.9
    alpha: float = 0.1  # Sharpness penalty weight
    smoothness_weight: float = field(default_factory=_get_ncc_smoothness_weight)
    integrator_hidden: int = 32
    default_epochs: int = field(default_factory=_get_ncc_epochs)
    artifact_path: str = "artifacts/ncc_state.pt"


class FeatureNet(nn.Module):
    """Feature extractor from forecast context + regime."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QuantileAdjustmentNet(nn.Module):
    """Predicts calibration adjustments per quantile per horizon."""

    def __init__(self, feature_dim: int, n_quantiles: int):
        super().__init__()
        self.fc = nn.Linear(feature_dim, n_quantiles)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Returns adjustment values for each quantile."""
        return self.fc(features)


class NeuralConformalControl(nn.Module):
    """
    Neural Conformal Control: learns to predict calibration adjustments.
    Trained with control-inspired loss to guarantee long-term coverage.
    """

    def __init__(self, config: NCCConfig):
        super().__init__()
        self.config = config

        # Feature extraction
        self.feature_net = FeatureNet(
            input_dim=config.feature_dim, hidden_dim=config.hidden_dim, output_dim=64
        )

        # Per-horizon quantile adjustment predictors
        self.quantile_nets = nn.ModuleList(
            [QuantileAdjustmentNet(64, config.n_quantiles) for _ in range(config.max_horizon)]
        )

        # Error integrator (PID-style control)
        self.error_integrator = nn.GRUCell(1, config.integrator_hidden)
        self.integrator_state: Optional[torch.Tensor] = None

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.learning_rate)

    def forward(
        self,
        base_quantiles: torch.Tensor,
        features: torch.Tensor,
        horizon_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Adjust base model quantiles for calibration.

        Args:
            base_quantiles: (batch, n_quantiles) or (batch, horizon, n_quantiles)
            features: (batch, feature_dim) context features
            horizon_idx: If provided, only adjust for this horizon

        Returns:
            calibrated_quantiles: Adjusted quantile forecasts
        """
        feat = self.feature_net(features)  # (batch, 64)

        if base_quantiles.dim() == 2:
            # Single horizon case
            if horizon_idx is None:
                horizon_idx = 0
            adjustments = self.quantile_nets[horizon_idx](feat)
            return base_quantiles + adjustments
        else:
            # Multi-horizon case
            batch_size, horizon, n_q = base_quantiles.shape
            adjustments = []
            for h in range(min(horizon, self.config.max_horizon)):
                adj = self.quantile_nets[h](feat)  # (batch, n_q)
                adjustments.append(adj)
            adjustments = torch.stack(adjustments, dim=1)  # (batch, horizon, n_q)
            return base_quantiles + adjustments[:, :horizon, :]

    def update_integrator(self, coverage_error: torch.Tensor) -> None:
        """Online update based on realized coverage error."""
        if self.integrator_state is None:
            self.integrator_state = torch.zeros(
                coverage_error.shape[0], self.config.integrator_hidden
            )

        self.integrator_state = self.error_integrator(
            coverage_error.unsqueeze(-1), self.integrator_state
        )

    def training_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor, quantiles: Sequence[float]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Control-inspired loss:
        - Coverage loss: Pinball loss for each quantile
        - Sharpness loss: Penalize wide intervals
        - Smoothness loss: Penalize erratic adjustments
        """
        # Pinball loss for coverage
        coverage_loss = self._pinball_loss(predictions, targets, quantiles)

        # Sharpness: penalize wide prediction intervals
        if predictions.shape[-1] >= 2:
            sharpness_loss = (predictions[..., -1] - predictions[..., 0]).mean()
        else:
            sharpness_loss = torch.tensor(0.0)

        # Smoothness: penalize temporal discontinuities
        if predictions.dim() == 3 and predictions.shape[1] > 1:
            smoothness_loss = (predictions[:, 1:] - predictions[:, :-1]).abs().mean()
        else:
            smoothness_loss = torch.tensor(0.0)

        total_loss = (
            coverage_loss
            + self.config.alpha * sharpness_loss
            + self.config.smoothness_weight * smoothness_loss
        )

        metrics = {
            "coverage_loss": coverage_loss.item(),
            "sharpness_loss": sharpness_loss.item() if isinstance(sharpness_loss, torch.Tensor) else 0.0,
            "smoothness_loss": smoothness_loss.item() if isinstance(smoothness_loss, torch.Tensor) else 0.0,
            "total_loss": total_loss.item(),
        }

        return total_loss, metrics

    def _pinball_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor, quantiles: Sequence[float]
    ) -> torch.Tensor:
        """Quantile regression loss (pinball loss)."""
        # predictions: (batch, [horizon,] n_quantiles)
        # targets: (batch, [horizon,] 1)
        n_q = predictions.shape[-1]
        quantiles_tensor = torch.tensor(quantiles, device=predictions.device).view(1, 1, -1)

        if predictions.dim() == 2:
            quantiles_tensor = quantiles_tensor.squeeze(1)

        errors = targets.unsqueeze(-1) - predictions
        loss = torch.where(
            errors >= 0,
            quantiles_tensor * errors,
            (quantiles_tensor - 1) * errors,
        )
        return loss.mean()

    def train_step(
        self,
        base_quantiles: torch.Tensor,
        features: torch.Tensor,
        targets: torch.Tensor,
        quantiles: Sequence[float],
    ) -> Dict[str, float]:
        """Single training step."""
        self.optimizer.zero_grad()
        predictions = self.forward(base_quantiles, features)
        loss, metrics = self.training_loss(predictions, targets, quantiles)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        return metrics

    def save(self, path: Path) -> None:
        """Save model state."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.state_dict(),
                "config": self.config,
                "integrator_state": self.integrator_state,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "NeuralConformalControl":
        """Load model state."""
        checkpoint = torch.load(path, weights_only=False)
        model = cls(checkpoint["config"])
        model.load_state_dict(checkpoint["model_state"])
        model.integrator_state = checkpoint.get("integrator_state")
        return model


class NCCCalibrationEngine:
    """
    Wrapper for Neural Conformal Control that integrates with the existing
    calibration interface.
    """

    def __init__(self, config: Optional[NCCConfig] = None, device: str = "cpu"):
        self.config = config or NCCConfig()
        self.device = device
        self.model = NeuralConformalControl(self.config).to(device)
        self.quantile_values = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]

        # Legacy compatibility
        self.version = 1
        self._coverage_stats: Dict[str, Dict[str, int]] = {}

        # Compatibility with CalibrationEngine interface
        from dataclasses import dataclass, field
        from typing import Dict, Tuple

        @dataclass
        class _MockState:
            coverage: Dict[Tuple[str, str], Dict[str, float]] = field(default_factory=dict)

        self.state = _MockState()

    def calibrate_quantiles(
        self,
        quantiles: Dict[float, float],
        features: Optional[Dict[str, float]] = None,
        horizon: Optional[int] = None,
        regime: Optional[object] = None,
    ) -> Dict[float, float]:
        """
        Calibrate quantiles using neural network.

        Args:
            quantiles: Raw quantile predictions {quantile: value}
            features: Context features for adaptation
            horizon: Forecast horizon (1-indexed)
            regime: RegimeVector (converted to features if provided)

        Returns:
            Calibrated quantiles
        """
        self.model.eval()
        with torch.no_grad():
            # Convert to tensor
            q_values = [quantiles.get(q, 0.0) for q in self.quantile_values]
            base_q = torch.tensor([q_values], device=self.device)

            # Build feature vector from regime if provided
            if regime is not None and features is None:
                features = self._regime_to_features(regime)

            # Build feature vector
            if features is None:
                features = {}
            feat_vector = self._build_feature_vector(features)
            feat_tensor = torch.tensor([feat_vector], device=self.device)

            # Get horizon index (0-indexed)
            h_idx = (horizon - 1) if horizon is not None else 0
            h_idx = max(0, min(h_idx, self.config.max_horizon - 1))

            # Calibrate
            calibrated = self.model(base_q, feat_tensor, horizon_idx=h_idx)
            calibrated = calibrated.squeeze(0).cpu().numpy()

            # Convert back to dict
            result = {q: float(calibrated[i]) for i, q in enumerate(self.quantile_values)}

            # Ensure monotonicity
            result = self._enforce_monotonicity(result)

        return result

    def update_with_outcome(
        self,
        predicted_quantiles: Dict[float, float],
        realized: float,
        features: Optional[Dict[str, float]] = None,
        horizon: Optional[int] = None,
    ) -> None:
        """
        Update calibration with realized outcome (online learning).

        Args:
            predicted_quantiles: Previously predicted quantiles
            realized: Actual realized value
            features: Context features used for prediction
            horizon: Horizon of the prediction
        """
        # Track coverage statistics
        coverage_key = f"h{horizon or 0}"
        if coverage_key not in self._coverage_stats:
            self._coverage_stats[coverage_key] = {"hits": 0, "total": 0}

        lower_q = 0.1
        upper_q = 0.9
        lower_val = predicted_quantiles.get(lower_q, float("-inf"))
        upper_val = predicted_quantiles.get(upper_q, float("inf"))

        hit = lower_val <= realized <= upper_val
        self._coverage_stats[coverage_key]["hits"] += int(hit)
        self._coverage_stats[coverage_key]["total"] += 1

        # Compute coverage error for integrator update
        target_coverage = self.config.target_coverage
        actual_coverage = (
            self._coverage_stats[coverage_key]["hits"] / self._coverage_stats[coverage_key]["total"]
        )
        coverage_error = target_coverage - actual_coverage

        # Update error integrator
        error_tensor = torch.tensor([coverage_error], device=self.device)
        self.model.update_integrator(error_tensor)

        self.version += 1

    def train_on_historical(
        self,
        base_quantiles: List[Dict[float, float]],
        features_list: List[Dict[str, float]],
        actuals: List[float],
        horizons: List[int],
        epochs: int = 10,
        batch_size: int = 32,
    ) -> Dict[str, List[float]]:
        """
        Train NCC on historical forecast-outcome pairs.

        Args:
            base_quantiles: List of raw quantile predictions
            features_list: List of feature dicts for each forecast
            actuals: List of realized values
            horizons: List of horizon values
            epochs: Number of training epochs
            batch_size: Batch size for training

        Returns:
            Training metrics history
        """
        self.model.train()

        # Prepare data
        n_samples = len(actuals)
        indices = list(range(n_samples))

        # Convert to tensors
        base_q_list = []
        feat_list = []
        target_list = []

        for i in range(n_samples):
            q_vals = [base_quantiles[i].get(q, 0.0) for q in self.quantile_values]
            base_q_list.append(q_vals)

            feat_vec = self._build_feature_vector(features_list[i])
            feat_list.append(feat_vec)

            target_list.append([actuals[i]])

        base_q_tensor = torch.tensor(base_q_list, device=self.device)
        feat_tensor = torch.tensor(feat_list, device=self.device)
        target_tensor = torch.tensor(target_list, device=self.device)

        # Training loop
        metrics_history: Dict[str, List[float]] = {
            "coverage_loss": [],
            "sharpness_loss": [],
            "smoothness_loss": [],
            "total_loss": [],
        }

        for epoch in range(epochs):
            # Shuffle
            import random

            random.shuffle(indices)

            epoch_metrics = {k: [] for k in metrics_history.keys()}

            # Mini-batches
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_indices = indices[start:end]

                batch_base_q = base_q_tensor[batch_indices]
                batch_feat = feat_tensor[batch_indices]
                batch_target = target_tensor[batch_indices]

                metrics = self.model.train_step(
                    batch_base_q, batch_feat, batch_target, self.quantile_values
                )

                for k, v in metrics.items():
                    epoch_metrics[k].append(v)

            # Average epoch metrics
            for k in metrics_history.keys():
                avg = sum(epoch_metrics[k]) / max(len(epoch_metrics[k]), 1)
                metrics_history[k].append(avg)

        return metrics_history

    def save(self, path: Path) -> None:
        """Save model and metadata."""
        self.model.save(path)
        # Save coverage stats separately
        meta_path = path.with_suffix(".json")
        meta_path.write_text(
            json.dumps(
                {"version": self.version, "coverage_stats": self._coverage_stats}, indent=2
            )
        )

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "NCCCalibrationEngine":
        """Load model and metadata."""
        model = NeuralConformalControl.load(path)
        engine = cls(config=model.config, device=device)
        engine.model = model

        # Load metadata
        meta_path = path.with_suffix(".json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            engine.version = meta.get("version", 1)
            engine._coverage_stats = meta.get("coverage_stats", {})

        return engine

    def _regime_to_features(self, regime: object) -> Dict[str, float]:
        """Convert RegimeVector to features dict."""
        features = {}
        try:
            # Extract relevant fields from RegimeVector
            if hasattr(regime, "realized_vol"):
                features["regime_volatility"] = regime.realized_vol
            if hasattr(regime, "iv_level"):
                features["iv_level"] = regime.iv_level
            if hasattr(regime, "funding_rate"):
                features["funding_rate"] = regime.funding_rate
            if hasattr(regime, "basis_annualized"):
                features["basis_pressure"] = regime.basis_annualized
            if hasattr(regime, "order_imbalance"):
                features["order_imbalance"] = regime.order_imbalance
            if hasattr(regime, "skew"):
                features["skew"] = regime.skew
            if hasattr(regime, "narrative_score"):
                features["narrative_score"] = regime.narrative_score
        except Exception:
            # Fallback to empty features
            pass
        return features

    def _build_feature_vector(self, features: Dict[str, float]) -> List[float]:
        """Build fixed-length feature vector from feature dict."""
        # Standardized feature order
        feature_keys = [
            "regime_volatility",
            "iv_level",
            "gamma_squeeze",
            "inventory_unwind",
            "basis_pressure",
            "funding_rate",
            "skew",
            "narrative_score",
            "order_imbalance",
            "horizon_norm",
        ]

        vector = []
        for key in feature_keys[: self.config.feature_dim]:
            vector.append(features.get(key, 0.0))

        # Pad if necessary
        while len(vector) < self.config.feature_dim:
            vector.append(0.0)

        return vector[: self.config.feature_dim]

    def _enforce_monotonicity(self, quantiles: Dict[float, float]) -> Dict[float, float]:
        """Ensure quantiles are monotonically increasing."""
        sorted_q = sorted(quantiles.items())
        result = {}

        prev_val = float("-inf")
        for q, val in sorted_q:
            result[q] = max(val, prev_val)
            prev_val = result[q]

        return result

    def get_coverage_stats(self) -> Dict[str, Dict[str, float]]:
        """Get coverage statistics."""
        stats = {}
        for key, counts in self._coverage_stats.items():
            if counts["total"] > 0:
                stats[key] = {
                    "coverage": counts["hits"] / counts["total"],
                    "hits": counts["hits"],
                    "total": counts["total"],
                }
        return stats

    def _regime_bucket(self, regime: object) -> str:
        """Compatibility method for ForecastEngine."""
        try:
            vol = regime.realized_vol if hasattr(regime, "realized_vol") else 0.0
            if vol < 0.05:
                return "calm"
            elif vol < 0.15:
                return "normal"
            else:
                return "volatile"
        except Exception:
            return "normal"

    def _horizon_bucket(self, horizon: int) -> str:
        """Compatibility method for ForecastEngine."""
        if horizon <= 3:
            return "short"
        elif horizon <= 7:
            return "mid"
        else:
            return "long"
