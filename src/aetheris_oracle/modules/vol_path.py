import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from ..data.schemas import VolPath


@dataclass
class VolPathConfig:
    mean_reversion: float = 0.18
    floor: float = 0.05
    regime_vol_scale: float = 0.08
    garch_alpha: float = 0.12
    garch_beta: float = 0.75
    artifact_path: str = "artifacts/vol_path_state.json"
    hidden_size: int = 6
    learning_rate: float = 0.01


@dataclass
class VolPathState:
    mlp_input: List[List[float]]
    mlp_bias: List[float]
    mlp_out: List[float]
    out_bias: float
    version: str = "v1"

    def to_dict(self) -> Dict:
        return {
            "mlp_input": self.mlp_input,
            "mlp_bias": self.mlp_bias,
            "mlp_out": self.mlp_out,
            "out_bias": self.out_bias,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, payload: Dict) -> "VolPathState":
        return cls(
            mlp_input=payload.get("mlp_input", [[0.05 for _ in range(5)] for _ in range(6)]),
            mlp_bias=payload.get("mlp_bias", [0.0 for _ in range(6)]),
            mlp_out=payload.get("mlp_out", [0.1 for _ in range(6)]),
            out_bias=payload.get("out_bias", 0.0),
            version=payload.get("version", "v1"),
        )


class VolPathEngine:
    """Forecasts implied volatility and skew paths with a small learned MLP."""

    def __init__(self, config: VolPathConfig | None = None, state: VolPathState | None = None) -> None:
        self.config = config or VolPathConfig()
        self.state = state or self._load_state(Path(self.config.artifact_path))

    def _safe_get(self, d: Dict[str, float], key: str, default: float) -> float:
        """Get value from dict, returning default if missing or NaN."""
        value = d.get(key)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return value

    def forecast(
        self,
        iv_points: Dict[str, float],
        horizon: int,
        regime_strength: float = 1.0,
        mm_indices: Tuple[float, float, float] | None = None,
    ) -> VolPath:
        base = self._safe_get(iv_points, "iv_7d_atm", 0.45)
        medium = self._safe_get(iv_points, "iv_14d_atm", base)
        longer = self._safe_get(iv_points, "iv_30d_atm", medium)
        skew = self._safe_get(iv_points, "skew_25d", 0.0)
        gsi, inventory, basis_pressure = mm_indices or (0.0, 0.0, 0.0)
        input_vec = [base, medium, longer, skew, regime_strength, gsi, inventory, basis_pressure]
        hidden = self._forward_hidden(input_vec)
        horizon_factor = 1 + 0.05 * regime_strength
        path: List[float] = []
        current = base
        variance = base ** 2
        for step in range(horizon):
            hidden = self._forward_hidden([current, medium, longer, skew, regime_strength, gsi, inventory, basis_pressure])
            delta = sum(w * h for w, h in zip(self.state.mlp_out, hidden)) + self.state.out_bias
            drift = -self.config.mean_reversion * (current - base)
            drift += self.config.regime_vol_scale * (regime_strength - 1)
            variance = (
                self.config.garch_alpha * (current ** 2)
                + self.config.garch_beta * variance
                + 0.01
            )
            garch_term = (variance ** 0.5) * 0.04
            current = max(self.config.floor, current + drift + delta * horizon_factor + garch_term)
            path.append(current)
        return VolPath(path=path)

    def train(
        self,
        features: Sequence[Dict[str, float]],
        targets: Sequence[List[float]],
        regime_strengths: Sequence[float],
    ) -> VolPathState:
        """Fit the MLP on historical IV/skew sequences."""
        if not features or not targets:
            return self.state
        lr = self.config.learning_rate
        for _ in range(4):
            for feat, tgt, reg in zip(features, targets, regime_strengths):
                iv_vec = [
                    feat.get("iv_7d_atm", 0.45),
                    feat.get("iv_14d_atm", feat.get("iv_7d_atm", 0.45)),
                    feat.get("iv_30d_atm", feat.get("iv_14d_atm", 0.45)),
                    feat.get("skew_25d", 0.0),
                    reg,
                    feat.get("gamma_squeeze", 0.0),
                    feat.get("inventory_unwind", 0.0),
                    feat.get("basis_pressure", 0.0),
                ]
                hidden = self._forward_hidden(iv_vec)
                pred = sum(w * h for w, h in zip(self.state.mlp_out, hidden)) + self.state.out_bias
                true_val = sum(tgt) / max(len(tgt), 1)
                error = pred - true_val
                grad_out = [error * h for h in hidden]
                self.state.mlp_out = [w - lr * g for w, g in zip(self.state.mlp_out, grad_out)]
                self.state.out_bias -= lr * error
                # Backprop to hidden/input weights
                grad_hidden = [error * self.state.mlp_out[i] for i in range(len(hidden))]
                for i in range(len(hidden)):
                    grad_act = grad_hidden[i] * (1 - hidden[i] ** 2)
                    for j in range(len(iv_vec)):
                        self.state.mlp_input[i][j] -= lr * grad_act * iv_vec[j]
                    self.state.mlp_bias[i] -= lr * grad_act
        self.save(Path(self.config.artifact_path))
        return self.state

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.state.to_dict(), indent=2))

    @staticmethod
    def _load_state(path: Path) -> VolPathState:
        if path.exists():
            try:
                return VolPathState.from_dict(json.loads(path.read_text()))
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to load volatility model state from {path}: {e}. Using default state.")
        return VolPathState(
            mlp_input=[[0.05 for _ in range(8)] for _ in range(6)],
            mlp_bias=[0.0 for _ in range(6)],
            mlp_out=[0.08 for _ in range(6)],
            out_bias=0.0,
        )

    def _forward_hidden(self, x: Sequence[float]) -> List[float]:
        hidden: List[float] = []
        for weights, bias in zip(self.state.mlp_input, self.state.mlp_bias):
            raw = sum(w * xi for w, xi in zip(weights, x)) + bias
            hidden.append(math.tanh(raw))
        return hidden
