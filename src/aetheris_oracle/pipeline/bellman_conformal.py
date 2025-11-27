"""
Bellman Conformal Inference for multi-horizon calibration.

Implements dynamic programming for horizon-consistent calibration that
jointly optimizes coverage across all forecast steps.

Based on "Bellman Conformal Inference" (Yang, Candès & Lei, 2024).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass
class BellmanConfig:
    """Configuration for Bellman Conformal Inference."""

    max_horizon: int = 14
    target_coverage: float = 0.9
    discount_factor: float = 0.95  # Temporal discounting
    cost_undercoverage: float = 10.0  # Penalty for missing true value
    cost_overcoverage: float = 0.1  # Penalty for too wide intervals
    n_threshold_candidates: int = 50  # Grid search resolution
    artifact_path: str = "artifacts/bellman_conformal_state.json"


@dataclass
class BellmanState:
    """Persisted state for Bellman Conformal Inference."""

    optimal_thresholds: Dict[int, Dict[str, float]]  # horizon -> {lower, upper} quantile adjustments
    value_function: Dict[int, float]  # V[t] = expected cost from t to horizon
    version: int = 1

    def to_dict(self) -> Dict:
        return {
            "optimal_thresholds": {
                str(k): v for k, v in self.optimal_thresholds.items()
            },
            "value_function": {str(k): v for k, v in self.value_function.items()},
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, payload: Dict) -> "BellmanState":
        return cls(
            optimal_thresholds={
                int(k): v for k, v in payload.get("optimal_thresholds", {}).items()
            },
            value_function={int(k): v for k, v in payload.get("value_function", {}).items()},
            version=payload.get("version", 1),
        )


class BellmanConformalOptimizer:
    """
    Dynamic programming for horizon-consistent calibration.
    Jointly optimizes coverage across all forecast steps using Bellman equation.
    """

    def __init__(self, config: Optional[BellmanConfig] = None):
        self.config = config or BellmanConfig()
        self.state = BellmanState(optimal_thresholds={}, value_function={})

    def optimize_thresholds(
        self,
        base_quantiles: Dict[int, Dict[float, float]],  # horizon -> {quantile -> value}
        historical_actuals: Optional[Sequence[float]] = None,
        historical_predictions: Optional[Sequence[Dict[int, Dict[float, float]]]] = None,
    ) -> Dict[int, Dict[float, float]]:
        """
        Optimize calibration thresholds across all horizons using dynamic programming.

        Args:
            base_quantiles: Base quantile predictions per horizon
            historical_actuals: Past realized values for training
            historical_predictions: Past forecast quantiles

        Returns:
            Calibrated quantiles per horizon
        """
        horizons = sorted(base_quantiles.keys())
        if not horizons:
            return base_quantiles

        max_horizon = max(horizons)

        # If we have historical data, optimize thresholds
        if historical_actuals is not None and historical_predictions is not None:
            self._fit_optimal_thresholds(
                horizons, historical_actuals, historical_predictions
            )

        # Apply learned thresholds
        calibrated_quantiles = {}
        for h in horizons:
            thresholds = self.state.optimal_thresholds.get(h, {"lower": 1.0, "upper": 1.0})
            calibrated_quantiles[h] = self._apply_thresholds(
                base_quantiles[h], thresholds
            )

        return calibrated_quantiles

    def _fit_optimal_thresholds(
        self,
        horizons: List[int],
        actuals: Sequence[float],
        predictions: Sequence[Dict[int, Dict[float, float]]],
    ) -> None:
        """
        Fit optimal thresholds using historical data via dynamic programming.

        This solves the Bellman equation:
        V[t] = min_{threshold_t} E[cost_t + γ * V[t+1]]
        """
        max_horizon = max(horizons)
        n_samples = len(actuals)

        # Build cost matrix: cost[sample, horizon, threshold_candidate]
        # This is the immediate cost of using each threshold candidate
        threshold_candidates = self._generate_threshold_candidates()
        n_candidates = len(threshold_candidates)

        # Initialize value function (backward pass)
        V = {}
        V[max_horizon + 1] = 0.0  # Terminal value

        # Backward dynamic programming
        for h in reversed(horizons):
            # Compute expected costs for each threshold candidate
            expected_costs = np.zeros(n_candidates)

            for i, (lower_adj, upper_adj) in enumerate(threshold_candidates):
                # Compute immediate cost for this threshold
                immediate_cost = 0.0

                for sample_idx in range(n_samples):
                    if h not in predictions[sample_idx]:
                        continue

                    pred = predictions[sample_idx][h]
                    actual = actuals[sample_idx]

                    # Apply threshold adjustment
                    adjusted_lower = pred.get(0.1, 0.0) * lower_adj
                    adjusted_upper = pred.get(0.9, 0.0) * upper_adj

                    # Compute cost
                    immediate_cost += self._compute_cost(
                        actual, adjusted_lower, adjusted_upper
                    )

                immediate_cost /= max(n_samples, 1)

                # Add discounted future value
                next_h = h + 1
                future_value = V.get(next_h, 0.0)
                expected_costs[i] = immediate_cost + self.config.discount_factor * future_value

            # Find best threshold
            best_idx = np.argmin(expected_costs)
            best_threshold = threshold_candidates[best_idx]
            V[h] = expected_costs[best_idx]

            # Store optimal threshold
            self.state.optimal_thresholds[h] = {
                "lower": best_threshold[0],
                "upper": best_threshold[1],
            }

        # Store value function
        self.state.value_function = {int(k): float(v) for k, v in V.items()}
        self.state.version += 1

    def _generate_threshold_candidates(self) -> List[Tuple[float, float]]:
        """Generate grid of (lower_adj, upper_adj) threshold candidates."""
        n_cand = self.config.n_threshold_candidates
        # Search around 1.0 with reasonable range
        lower_range = np.linspace(0.8, 1.2, n_cand // 2)
        upper_range = np.linspace(0.8, 1.2, n_cand // 2)

        candidates = []
        for lower in lower_range:
            for upper in upper_range:
                if upper >= lower:  # Upper must be >= lower
                    candidates.append((lower, upper))

        return candidates[:n_cand]  # Limit to n_candidates

    def _compute_cost(
        self, actual: float, lower: float, upper: float
    ) -> float:
        """
        Compute cost of prediction interval.

        Cost = undercoverage_penalty if actual outside [lower, upper]
               + overcoverage_penalty * (interval width)
        """
        # Coverage cost
        if actual < lower or actual > upper:
            coverage_cost = self.config.cost_undercoverage
        else:
            coverage_cost = 0.0

        # Width cost (sharpness)
        width = max(upper - lower, 0.0)
        width_cost = self.config.cost_overcoverage * width

        return coverage_cost + width_cost

    def _apply_thresholds(
        self, quantiles: Dict[float, float], thresholds: Dict[str, float]
    ) -> Dict[float, float]:
        """Apply learned threshold adjustments to base quantiles."""
        median = quantiles.get(0.5, 0.0)
        result = {}

        for q, val in quantiles.items():
            if q == 0.5:
                result[q] = val
                continue

            delta = val - median

            if q < 0.5:
                # Lower quantile: apply lower threshold
                adj = thresholds.get("lower", 1.0)
            else:
                # Upper quantile: apply upper threshold
                adj = thresholds.get("upper", 1.0)

            result[q] = median + delta * adj

        return result

    def save(self, path: Path) -> None:
        """Save state to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.state.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path, config: Optional[BellmanConfig] = None) -> "BellmanConformalOptimizer":
        """Load state from disk."""
        payload = json.loads(path.read_text())
        optimizer = cls(config)
        optimizer.state = BellmanState.from_dict(payload)
        return optimizer


class BellmanNCCHybrid:
    """
    Hybrid approach combining Bellman optimization with Neural Conformal Control.

    Bellman provides multi-horizon consistency, NCC provides adaptive learning.
    """

    def __init__(
        self,
        ncc_model: Optional["NCCCalibrationEngine"] = None,
        bellman_config: Optional[BellmanConfig] = None,
    ):
        from .neural_conformal_control import NCCCalibrationEngine

        self.ncc = ncc_model or NCCCalibrationEngine()
        self.bellman = BellmanConformalOptimizer(bellman_config)

    def calibrate_quantiles(
        self,
        base_quantiles: Dict[int, Dict[float, float]],
        features: Optional[Dict[str, float]] = None,
    ) -> Dict[int, Dict[float, float]]:
        """
        Two-stage calibration:
        1. NCC provides adaptive, regime-aware adjustments
        2. Bellman ensures multi-horizon consistency
        """
        # Stage 1: NCC calibration per horizon
        ncc_calibrated = {}
        for horizon, quantiles in base_quantiles.items():
            feat_with_horizon = (features or {}).copy()
            feat_with_horizon["horizon_norm"] = horizon / 14.0
            ncc_calibrated[horizon] = self.ncc.calibrate_quantiles(
                quantiles, features=feat_with_horizon, horizon=horizon
            )

        # Stage 2: Bellman multi-horizon optimization
        final_calibrated = self.bellman.optimize_thresholds(ncc_calibrated)

        return final_calibrated

    def train_on_historical(
        self,
        base_quantiles_history: List[Dict[int, Dict[float, float]]],
        features_history: List[Dict[str, float]],
        actuals_history: List[float],
        ncc_epochs: int = 10,
    ) -> Dict[str, any]:
        """
        Train both components on historical data.

        Args:
            base_quantiles_history: List of forecast quantiles per horizon
            features_history: List of feature dicts
            actuals_history: List of realized values
            ncc_epochs: Epochs for NCC training

        Returns:
            Training metrics
        """
        # Train NCC (per-horizon adaptive learning)
        # Flatten multi-horizon forecasts for NCC
        flat_base_quantiles = []
        flat_features = []
        flat_actuals = []
        flat_horizons = []

        for forecasts, features, actual in zip(
            base_quantiles_history, features_history, actuals_history
        ):
            for horizon, quantiles in forecasts.items():
                flat_base_quantiles.append(quantiles)
                feat_with_h = features.copy()
                feat_with_h["horizon_norm"] = horizon / 14.0
                flat_features.append(feat_with_h)
                flat_actuals.append(actual)
                flat_horizons.append(horizon)

        ncc_metrics = self.ncc.train_on_historical(
            flat_base_quantiles,
            flat_features,
            flat_actuals,
            flat_horizons,
            epochs=ncc_epochs,
        )

        # Generate NCC-calibrated forecasts for Bellman
        ncc_calibrated_history = []
        for forecasts, features in zip(base_quantiles_history, features_history):
            calibrated = {}
            for horizon, quantiles in forecasts.items():
                feat_with_h = features.copy()
                feat_with_h["horizon_norm"] = horizon / 14.0
                calibrated[horizon] = self.ncc.calibrate_quantiles(
                    quantiles, features=feat_with_h, horizon=horizon
                )
            ncc_calibrated_history.append(calibrated)

        # Train Bellman (multi-horizon consistency)
        horizons = list(base_quantiles_history[0].keys())
        self.bellman._fit_optimal_thresholds(
            horizons, actuals_history, ncc_calibrated_history
        )

        return {
            "ncc_metrics": ncc_metrics,
            "bellman_value_function": self.bellman.state.value_function,
            "bellman_thresholds": self.bellman.state.optimal_thresholds,
        }

    def save(self, ncc_path: Path, bellman_path: Path) -> None:
        """Save both components."""
        self.ncc.save(ncc_path)
        self.bellman.save(bellman_path)

    @classmethod
    def load(
        cls,
        ncc_path: Path,
        bellman_path: Path,
        device: str = "cpu",
    ) -> "BellmanNCCHybrid":
        """Load both components."""
        from .neural_conformal_control import NCCCalibrationEngine

        ncc = NCCCalibrationEngine.load(ncc_path, device=device)
        bellman = BellmanConformalOptimizer.load(bellman_path)
        return cls(ncc, bellman.config)
