import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ..data.schemas import TrendPath


@dataclass
class TrendConfig:
    smoothing: float = 0.55  # base weight for long-horizon component
    drift_regime_scale: float = 0.05
    meta_learning_rate: float = 0.35
    weight_floor: float = 0.08
    artifact_path: str = "artifacts/trend_state.json"
    lag: int = 5
    hidden_size: int = 8


@dataclass
class TrendEnsembleState:
    """Persisted weights for the AR + neural trend ensemble."""

    ar_coeffs: List[float]
    ar_bias: float
    rnn_input: List[List[float]]
    rnn_hidden: List[List[float]]
    rnn_bias: List[float]
    rnn_out: List[float]
    rnn_out_bias: float
    meta_weights: Dict[str, float]
    version: str = "v1"

    def to_dict(self) -> Dict:
        return {
            "ar_coeffs": self.ar_coeffs,
            "ar_bias": self.ar_bias,
            "rnn_input": self.rnn_input,
            "rnn_hidden": self.rnn_hidden,
            "rnn_bias": self.rnn_bias,
            "rnn_out": self.rnn_out,
            "rnn_out_bias": self.rnn_out_bias,
            "meta_weights": self.meta_weights,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, payload: Dict) -> "TrendEnsembleState":
        return cls(
            ar_coeffs=payload.get("ar_coeffs", [0.1]),
            ar_bias=payload.get("ar_bias", 0.0),
            rnn_input=payload.get("rnn_input", [[0.05, 0.05]]),
            rnn_hidden=payload.get("rnn_hidden", [[0.05]]),
            rnn_bias=payload.get("rnn_bias", [0.0]),
            rnn_out=payload.get("rnn_out", [0.1]),
            rnn_out_bias=payload.get("rnn_out_bias", 0.0),
            meta_weights=payload.get("meta_weights", {"ar": 0.5, "rnn": 0.5}),
            version=payload.get("version", "v1"),
        )


class TrendEnsemble:
    """Data-driven ensemble with AR + gated neural component and adaptive weights."""

    def __init__(
        self,
        config: TrendConfig | None = None,
        seed: Optional[int] = None,
        state: TrendEnsembleState | None = None,
    ) -> None:
        self.config = config or TrendConfig()
        self._rng = random.Random(seed)
        self.state = state or self._load_state(Path(self.config.artifact_path))
        self._prev_losses = {"ar": 0.1, "rnn": 0.1}

    def predict_trend(
        self, normalized_closes: List[float], horizon: int, regime_strength: float = 1.0
    ) -> TrendPath:
        if not normalized_closes:
            raise ValueError("normalized_closes must not be empty")
        ar_forecast = self._ar_component(normalized_closes, horizon)
        rnn_forecast = self._neural_component(normalized_closes, horizon, regime_strength)
        weights = self._meta_weights()
        path = []
        for i in range(horizon):
            combined = weights["ar"] * ar_forecast[i] + weights["rnn"] * rnn_forecast[i]
            path.append(combined)
        return TrendPath(path=path)

    def update_online_weights(self, horizon_loss: float) -> None:
        """EWAF update with a single scalar loss for the blended forecast."""
        if horizon_loss <= 0:
            return
        for name, loss in self._prev_losses.items():
            adj = math.exp(-self.config.meta_learning_rate * horizon_loss * (1 + loss))
            self.state.meta_weights[name] = max(self.config.weight_floor, self.state.meta_weights[name] * adj)
        self._normalize_meta()

    def train_from_windows(
        self, windows: Sequence[List[float]], targets: Sequence[List[float]]
    ) -> TrendEnsembleState:
        """Lightweight fitting using past windows and horizon targets."""
        if not windows or not targets:
            return self.state
        self.state = self._fit_ar(windows, targets)
        self.state = self._fit_neural(windows, targets, self.state)
        self.state = self._fit_meta(windows, targets, self.state)
        self.save(Path(self.config.artifact_path))
        return self.state

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.state.to_dict(), indent=2))

    @staticmethod
    def _load_state(path: Path) -> TrendEnsembleState:
        if path.exists():
            try:
                return TrendEnsembleState.from_dict(json.loads(path.read_text()))
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to load trend model state from {path}: {e}. Using default state.")
        return TrendEnsembleState(
            ar_coeffs=[0.08, 0.05, 0.02],
            ar_bias=0.0,
            rnn_input=[[0.05, 0.02] for _ in range(8)],
            rnn_hidden=[[0.1 if i == j else 0.0 for j in range(8)] for i in range(8)],
            rnn_bias=[0.0 for _ in range(8)],
            rnn_out=[0.05 for _ in range(8)],
            rnn_out_bias=0.0,
            meta_weights={"ar": 0.55, "rnn": 0.45},
        )

    def _ar_component(self, series: List[float], horizon: int) -> List[float]:
        lags = min(self.config.lag, len(series) - 1)
        if lags <= 0:
            return [series[-1]] * horizon
        returns = [series[i] - series[i - 1] for i in range(1, len(series))]
        path: List[float] = []
        for step in range(horizon):
            feat = returns[-lags:]
            coeffs = self.state.ar_coeffs[:lags]
            prediction = sum(c * r for c, r in zip(coeffs, feat)) + self.state.ar_bias
            projected = series[-1] + (step + 1) * prediction
            path.append(projected)
        return path

    def _neural_component(self, series: List[float], horizon: int, regime_strength: float) -> List[float]:
        hidden = [0.0 for _ in range(len(self.state.rnn_bias))]
        recent = series[-(self.config.lag + 1) :]
        returns = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
        for r in returns:
            x = [r, regime_strength]
            hidden = self._gru_step(hidden, x)
        path: List[float] = []
        last = series[-1]
        for step in range(horizon):
            x_step = [returns[-1] if returns else 0.0, regime_strength * (1 + 0.1 * step)]
            hidden = self._gru_step(hidden, x_step)
            delta = sum(w * h for w, h in zip(self.state.rnn_out, hidden)) + self.state.rnn_out_bias
            path.append(last + delta * (step + 1))
        return path

    def _gru_step(self, hidden: List[float], x: List[float]) -> List[float]:
        new_h: List[float] = []
        for i in range(len(self.state.rnn_bias)):
            gate_input = sum(w * xi for w, xi in zip(self.state.rnn_input[i], x))
            gate_hidden = sum(w * h for w, h in zip(self.state.rnn_hidden[i], hidden))
            raw = gate_input + gate_hidden + self.state.rnn_bias[i]
            z = 1 / (1 + math.exp(-raw))
            proposal = math.tanh(raw)
            new_h.append((1 - z) * hidden[i] + z * proposal)
        return new_h

    def _fit_ar(
        self, windows: Sequence[List[float]], targets: Sequence[List[float]]
    ) -> TrendEnsembleState:
        # Ordinary least squares on lagged returns for horizon-mean return
        lag = min(self.config.lag, min(len(w) for w in windows) - 1)
        if lag <= 0:
            return self.state
        design: List[List[float]] = []
        y: List[float] = []
        for window, target in zip(windows, targets):
            returns = [window[i] - window[i - 1] for i in range(1, len(window))]
            feat = returns[-lag:]
            design.append(feat)
            y.append(sum(target) / max(len(target), 1) - window[-1])
        coeffs, bias = _solve_least_squares(design, y)
        self.state.ar_coeffs = coeffs or self.state.ar_coeffs
        self.state.ar_bias = bias if bias is not None else self.state.ar_bias
        return self.state

    def _fit_neural(
        self,
        windows: Sequence[List[float]],
        targets: Sequence[List[float]],
        state: TrendEnsembleState,
    ) -> TrendEnsembleState:
        lr = 0.01
        for _ in range(3):
            for window, target in zip(windows, targets):
                hidden = [0.0 for _ in range(len(state.rnn_bias))]
                recent = window[-(self.config.lag + 1) :]
                returns = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
                for r in returns:
                    hidden = self._gru_step(hidden, [r, 1.0])
                pred_seq = []
                for step, tgt in enumerate(target):
                    hidden = self._gru_step(hidden, [returns[-1] if returns else 0.0, 1.0])
                    out = sum(w * h for w, h in zip(state.rnn_out, hidden)) + state.rnn_out_bias
                    pred_seq.append(out * (step + 1) + window[-1])
                    error = (pred_seq[-1] - tgt) / max(len(target), 1)
                    grad_out = [error * h for h in hidden]
                    state.rnn_out = [w - lr * g for w, g in zip(state.rnn_out, grad_out)]
                    state.rnn_out_bias -= lr * error
        return state

    def _fit_meta(
        self,
        windows: Sequence[List[float]],
        targets: Sequence[List[float]],
        state: TrendEnsembleState,
    ) -> TrendEnsembleState:
        ar_err, rnn_err = 1e-6, 1e-6
        for window, target in zip(windows, targets):
            ar_pred = self._ar_component(window, len(target))
            rnn_pred = self._neural_component(window, len(target), 1.0)
            for a, r, tgt in zip(ar_pred, rnn_pred, target):
                ar_err += abs(a - tgt)
                rnn_err += abs(r - tgt)
        ar_weight = math.exp(-self.config.meta_learning_rate * ar_err)
        rnn_weight = math.exp(-self.config.meta_learning_rate * rnn_err)
        total = ar_weight + rnn_weight
        state.meta_weights = {
            "ar": max(self.config.weight_floor, ar_weight / total),
            "rnn": max(self.config.weight_floor, rnn_weight / total),
        }
        self._normalize_meta()
        self._prev_losses = {"ar": ar_err, "rnn": rnn_err}
        return state

    def _meta_weights(self) -> Dict[str, float]:
        self._normalize_meta()
        return self.state.meta_weights

    def _normalize_meta(self) -> None:
        total = sum(self.state.meta_weights.values())
        if total <= 0:
            self.state.meta_weights = {"ar": 0.5, "rnn": 0.5}
            return
        for k in list(self.state.meta_weights.keys()):
            self.state.meta_weights[k] = self.state.meta_weights[k] / total


def _solve_least_squares(design: Sequence[Sequence[float]], targets: Sequence[float]) -> tuple[List[float], float]:
    if not design:
        return [], 0.0
    # Simple normal equation solver for small matrices
    cols = len(design[0])
    xtx = [[0.0 for _ in range(cols)] for _ in range(cols)]
    xty = [0.0 for _ in range(cols)]
    for row, y in zip(design, targets):
        for i in range(cols):
            xty[i] += row[i] * y
            for j in range(cols):
                xtx[i][j] += row[i] * row[j]
    # Add small ridge for stability
    ridge = 1e-3
    for i in range(cols):
        xtx[i][i] += ridge
    coeffs = _gaussian_solve(xtx, xty)
    bias = sum(targets) / len(targets) if targets else 0.0
    return coeffs, bias


def _gaussian_solve(matrix: List[List[float]], vector: List[float]) -> List[float]:
    n = len(vector)
    aug = [row + [vector[i]] for i, row in enumerate(matrix)]
    for i in range(n):
        pivot = aug[i][i] or 1e-8
        for j in range(i, n + 1):
            aug[i][j] /= pivot
        for k in range(n):
            if k == i:
                continue
            factor = aug[k][i]
            for j in range(i, n + 1):
                aug[k][j] -= factor * aug[i][j]
    return [aug[i][-1] for i in range(n)]
