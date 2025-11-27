import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ..data.schemas import JumpPath


@dataclass
class JumpConfig:
    base_lambda: float = 0.15
    max_jump: float = 0.12
    self_excitation: float = 0.1
    artifact_path: str = "artifacts/jump_state.json"
    learning_rate: float = 0.01


@dataclass
class JumpState:
    mlp_input: List[List[float]]
    mlp_bias: List[float]
    mlp_out: List[float]
    scale_out: List[float]
    out_bias: float
    version: str = "v1"

    def to_dict(self) -> Dict:
        return {
            "mlp_input": self.mlp_input,
            "mlp_bias": self.mlp_bias,
            "mlp_out": self.mlp_out,
            "scale_out": self.scale_out,
            "out_bias": self.out_bias,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, payload: Dict) -> "JumpState":
        return cls(
            mlp_input=payload.get("mlp_input", [[0.05 for _ in range(5)] for _ in range(4)]),
            mlp_bias=payload.get("mlp_bias", [0.0 for _ in range(4)]),
            mlp_out=payload.get("mlp_out", [0.08 for _ in range(4)]),
            scale_out=payload.get("scale_out", [0.02 for _ in range(4)]),
            out_bias=payload.get("out_bias", 0.0),
            version=payload.get("version", "v1"),
        )


class JumpModel:
    """Data-driven jump sampler conditioned on volatility, skew, and market-maker pressure."""

    def __init__(
        self,
        config: JumpConfig | None = None,
        seed: Optional[int] = None,
        state: JumpState | None = None,
    ) -> None:
        self.config = config or JumpConfig()
        self._rng = random.Random(seed)
        self.state = state or self._load_state(Path(self.config.artifact_path))

    def sample_path(
        self,
        horizon: int,
        vol_path: List[float],
        narrative_score: float,
        gamma_squeeze: float,
        regime_strength: float = 1.0,
        basis_pressure: float = 0.0,
    ) -> JumpPath:
        path: List[float] = []
        intensity_boost = 0.0
        skew_proxy = max(vol_path) - min(vol_path) if vol_path else 0.0
        for t in range(horizon):
            features = [
                vol_path[min(t, len(vol_path) - 1)],
                skew_proxy,
                gamma_squeeze,
                basis_pressure,
                narrative_score,
            ]
            hidden = self._forward_hidden(features)
            lam = max(0.0, sum(w * h for w, h in zip(self.state.mlp_out, hidden)) + self.state.out_bias)
            lam += self.config.base_lambda + regime_strength * 0.05 + intensity_boost
            scale = max(0.01, sum(w * h for w, h in zip(self.state.scale_out, hidden)))
            jump = 0.0
            if self._rng.random() < lam:
                direction = 1 if self._rng.random() > 0.5 else -1
                magnitude = min(self.config.max_jump, self._rng.gauss(scale, scale / 2)) * direction
                jump = magnitude
                intensity_boost = min(0.35, intensity_boost + self.config.self_excitation)
            else:
                intensity_boost = max(0.0, intensity_boost * 0.5)
            path.append(jump)
        return JumpPath(path=path)

    def train(
        self,
        feature_sequences: Sequence[Sequence[float]],
        jump_labels: Sequence[Sequence[float]],
    ) -> JumpState:
        if not feature_sequences or not jump_labels:
            return self.state
        lr = self.config.learning_rate
        for _ in range(3):
            for feat, label_seq in zip(feature_sequences, jump_labels):
                hidden = self._forward_hidden(feat)
                for target in label_seq:
                    lam_pred = sum(w * h for w, h in zip(self.state.mlp_out, hidden)) + self.state.out_bias
                    error = lam_pred - target
                    grad_out = [error * h for h in hidden]
                    self.state.mlp_out = [w - lr * g for w, g in zip(self.state.mlp_out, grad_out)]
                    self.state.out_bias -= lr * error
        self.save(Path(self.config.artifact_path))
        return self.state

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.state.to_dict(), indent=2))

    @staticmethod
    def _load_state(path: Path) -> JumpState:
        if path.exists():
            try:
                return JumpState.from_dict(json.loads(path.read_text()))
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to load jump model state from {path}: {e}. Using default state.")
        return JumpState(
            mlp_input=[[0.05 for _ in range(5)] for _ in range(4)],
            mlp_bias=[0.0 for _ in range(4)],
            mlp_out=[0.08 for _ in range(4)],
            scale_out=[0.02 for _ in range(4)],
            out_bias=0.0,
        )

    def _forward_hidden(self, x: Sequence[float]) -> List[float]:
        hidden: List[float] = []
        for weights, bias in zip(self.state.mlp_input, self.state.mlp_bias):
            raw = sum(w * xi for w, xi in zip(weights, x)) + bias
            hidden.append(math.tanh(raw))
        return hidden
