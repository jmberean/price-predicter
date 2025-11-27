import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from ..data.schemas import ResidualPaths


@dataclass
class ResidualConfig:
    base_vol: float = 0.02
    ar_coefficient: float = 0.25
    tail_weight: float = 0.18
    artifact_path: str = "artifacts/residual_state.json"
    learning_rate: float = 0.01
    hidden_size: int = 6


@dataclass
class ResidualState:
    rnn_input: List[List[float]]
    rnn_hidden: List[List[float]]
    rnn_bias: List[float]
    out_mean: List[float]
    out_log_std: List[float]
    out_bias: float
    version: str = "v1"

    def to_dict(self) -> dict:
        return {
            "rnn_input": self.rnn_input,
            "rnn_hidden": self.rnn_hidden,
            "rnn_bias": self.rnn_bias,
            "out_mean": self.out_mean,
            "out_log_std": self.out_log_std,
            "out_bias": self.out_bias,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ResidualState":
        return cls(
            rnn_input=payload.get("rnn_input", [[0.05 for _ in range(4)] for _ in range(6)]),
            rnn_hidden=payload.get("rnn_hidden", [[0.05 if i == j else 0.0 for j in range(6)] for i in range(6)]),
            rnn_bias=payload.get("rnn_bias", [0.0 for _ in range(6)]),
            out_mean=payload.get("out_mean", [0.1 for _ in range(6)]),
            out_log_std=payload.get("out_log_std", [0.05 for _ in range(6)]),
            out_bias=payload.get("out_bias", 0.0),
            version=payload.get("version", "v1"),
        )


class ResidualGenerator:
    """Learned residual generator conditioned on regime, vol path, and MM pressure."""

    def __init__(
        self,
        config: ResidualConfig | None = None,
        seed: Optional[int] = None,
        state: ResidualState | None = None,
    ) -> None:
        self.config = config or ResidualConfig()
        self._rng = random.Random(seed)
        self.state = state or self._load_state(Path(self.config.artifact_path))

    def sample_paths(
        self,
        horizon: int,
        num_paths: int,
        vol_path: List[float],
        regime_strength: float,
        mm_features: Sequence[float] | None = None,
    ) -> ResidualPaths:
        paths: List[List[float]] = []
        mm_gamma, mm_inventory, mm_basis = (mm_features or (0.0, 0.0, 0.0))
        for _ in range(num_paths):
            path = []
            hidden = [0.0 for _ in range(len(self.state.rnn_bias))]
            ar_state = 0.0
            for t in range(horizon):
                conditioning = [
                    vol_path[min(t, len(vol_path) - 1)],
                    regime_strength,
                    mm_gamma,
                    mm_inventory + mm_basis,
                ]
                hidden = self._rnn_step(hidden, conditioning)
                mean = sum(w * h for w, h in zip(self.state.out_mean, hidden)) + self.state.out_bias
                log_std = sum(w * h for w, h in zip(self.state.out_log_std, hidden)) - 2.0
                std = max(1e-4, math.exp(log_std)) + vol_path[min(t, len(vol_path) - 1)] * 0.05
                ar_state = self.config.ar_coefficient * ar_state + mean
                noise = self._rng.gauss(0, std)
                tail = self._tail_noise(std)
                path.append(ar_state + noise + tail)
            path = _zero_center(path)
            paths.append(path)
        return ResidualPaths(paths=paths)

    def train(
        self,
        conditioning: Sequence[Sequence[float]],
        residual_targets: Sequence[Sequence[float]],
    ) -> ResidualState:
        """Fit the residual model on observed residual sequences."""
        if not conditioning or not residual_targets:
            return self.state
        lr = self.config.learning_rate
        for _ in range(4):
            for cond_seq, target_seq in zip(conditioning, residual_targets):
                hidden = [0.0 for _ in range(len(self.state.rnn_bias))]
                for cond, target in zip(cond_seq, target_seq):
                    hidden = self._rnn_step(hidden, cond)
                    pred_mean = sum(w * h for w, h in zip(self.state.out_mean, hidden)) + self.state.out_bias
                    error = pred_mean - target
                    grad_out = [error * h for h in hidden]
                    self.state.out_mean = [w - lr * g for w, g in zip(self.state.out_mean, grad_out)]
                    self.state.out_bias -= lr * error
                    grad_hidden = [error * self.state.out_mean[i] for i in range(len(hidden))]
                    for i in range(len(hidden)):
                        grad_act = grad_hidden[i] * (1 - hidden[i] ** 2)
                        for j in range(len(cond)):
                            self.state.rnn_input[i][j] -= lr * grad_act * cond[j]
                        self.state.rnn_bias[i] -= lr * grad_act
        self.save(Path(self.config.artifact_path))
        return self.state

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.state.to_dict(), indent=2))

    @staticmethod
    def _load_state(path: Path) -> ResidualState:
        if path.exists():
            try:
                return ResidualState.from_dict(json.loads(path.read_text()))
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to load residual model state from {path}: {e}. Using default state.")
        return ResidualState(
            rnn_input=[[0.05 for _ in range(4)] for _ in range(6)],
            rnn_hidden=[[0.1 if i == j else 0.0 for j in range(6)] for i in range(6)],
            rnn_bias=[0.0 for _ in range(6)],
            out_mean=[0.05 for _ in range(6)],
            out_log_std=[0.03 for _ in range(6)],
            out_bias=0.0,
        )

    def _rnn_step(self, hidden: List[float], x: Sequence[float]) -> List[float]:
        new_h: List[float] = []
        for i in range(len(self.state.rnn_bias)):
            raw = sum(w * xi for w, xi in zip(self.state.rnn_input[i], x)) + sum(
                w * h for w, h in zip(self.state.rnn_hidden[i], hidden)
            )
            raw += self.state.rnn_bias[i]
            new_h.append(math.tanh(raw))
        return new_h

    def _tail_noise(self, sigma: float) -> float:
        if self._rng.random() > self.config.tail_weight:
            return 0.0
        sign = 1 if self._rng.random() > 0.5 else -1
        return sign * self._rng.expovariate(1 / max(sigma, 1e-6)) * 0.08


def _zero_center(seq: List[float]) -> List[float]:
    if not seq:
        return seq
    mean = sum(seq) / len(seq)
    return [v - mean for v in seq]
