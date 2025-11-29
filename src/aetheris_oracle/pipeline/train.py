"""Offline training utilities for trend/vol/jump/residual modules."""

from __future__ import annotations

import logging

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Sequence

from ..data.free_connectors import FreeDataConnector
from ..data.interfaces import DataConnector
from ..features.regime import compute_regime_vector
from ..features.stationarity import StationarityNormalizer
from ..modules.jump import JumpConfig, JumpModel, JumpState
from ..modules.market_maker import MarketMakerEngine
from ..modules.residual import ResidualConfig, ResidualGenerator, ResidualState
from ..modules.trend import TrendConfig, TrendEnsemble, TrendEnsembleState
from ..modules.vol_path import VolPathConfig, VolPathEngine, VolPathState


@dataclass
class TrainConfig:
    asset_id: str = "BTC-USD"
    horizon_days: int = 7
    trailing_window_days: int = 90
    num_samples: int = 24
    artifact_root: str = "artifacts"
    seed: int | None = None
    validation_split: float = 0.2  # 80/20 train/val split
    early_stopping_patience: int = 5  # Stop if no improvement for N epochs
    log_validation: bool = True  # Log validation loss each epoch

    @property
    def window(self) -> timedelta:
        return timedelta(days=self.trailing_window_days + self.horizon_days)


@dataclass
class TrainArtifacts:
    trend: TrendEnsembleState
    vol: VolPathState
    residual: ResidualState
    jump: JumpState


def run_training(config: TrainConfig, connector: DataConnector | None = None) -> TrainArtifacts:
    connector = connector or FreeDataConnector()
    normalizer = StationarityNormalizer()
    mm_engine = MarketMakerEngine()

    trend_inputs: List[List[float]] = []
    trend_targets: List[List[float]] = []
    vol_features: List[Dict[str, float]] = []
    vol_targets: List[List[float]] = []
    vol_regimes: List[float] = []
    residual_conditioning: List[List[List[float]]] = []
    residual_targets: List[List[float]] = []
    jump_features: List[List[float]] = []
    jump_labels: List[List[float]] = []

    now = datetime.utcnow()
    for idx in range(config.num_samples):
        as_of = now - timedelta(days=idx)
        frame = connector.fetch_window(
            asset_id=config.asset_id,
            as_of=as_of,
            window=config.window,
        )
        if len(frame.closes) <= config.horizon_days + 5:
            continue
        past = frame.closes[: -config.horizon_days]
        future = frame.closes[-config.horizon_days :]
        normalized_past, stats = normalizer.normalize_and_stats(past)
        normalized_future = [(p - stats.mean) / stats.std for p in future]
        regime = compute_regime_vector(
            closes=past,
            iv_points=frame.iv_points,
            funding_rate=frame.funding_rate,
            basis=frame.basis,
            order_imbalance=frame.order_imbalance,
            narrative_scores=frame.narrative_scores,
            skew=frame.skew,
        )
        regime_strength = 1.0 + abs(regime.values[0]) if regime.values else 1.0
        mm = mm_engine.compute_indices(
            iv_term_structure=frame.iv_points,
            funding_rate=frame.funding_rate,
            basis=frame.basis,
            order_imbalance=frame.order_imbalance,
            skew=frame.skew,
        )

        trend_inputs.append(normalized_past)
        trend_targets.append(normalized_future)

        vol_features.append(
            {
                **frame.iv_points,
                "skew_25d": frame.skew,
                "gamma_squeeze": mm.gamma_squeeze,
                "inventory_unwind": mm.inventory_unwind,
                "basis_pressure": mm.basis_pressure,
            }
        )
        vol_targets.append(_simulate_vol_future(frame.iv_points, config.horizon_days, regime_strength))
        vol_regimes.append(regime_strength)

        residual_cond = []
        residual_seq = []
        base_trend = normalized_past[-1]
        for step, tgt in enumerate(normalized_future):
            vol_level = vol_targets[-1][step] if vol_targets else frame.iv_points.get("iv_7d_atm", 0.5)
            residual_seq.append(tgt - base_trend)
            residual_cond.append([vol_level, regime_strength, mm.gamma_squeeze, mm.inventory_unwind + mm.basis_pressure])
        residual_conditioning.append(residual_cond)
        residual_targets.append(residual_seq)

        jump_features.append(
            [
                vol_targets[-1][0] if vol_targets else frame.iv_points.get("iv_7d_atm", 0.5),
                max(vol_targets[-1]) - min(vol_targets[-1]) if vol_targets else 0.0,
                mm.gamma_squeeze,
                mm.basis_pressure,
                sum(frame.narrative_scores.values()) / max(len(frame.narrative_scores), 1),
            ]
        )
        jump_labels.append(_label_jumps(normalized_future))

    artifact_root = Path(config.artifact_root)
    trend_model = TrendEnsemble(
        seed=config.seed,
        config=_trend_config_with_path(artifact_root),
    )
    trend_state = trend_model.train_from_windows(trend_inputs, trend_targets)

    vol_engine = VolPathEngine(config=_vol_config_with_path(artifact_root))
    vol_state = vol_engine.train(vol_features, vol_targets, vol_regimes)

    residual_model = ResidualGenerator(config=_residual_config_with_path(artifact_root), seed=config.seed)
    residual_state = residual_model.train(residual_conditioning, residual_targets)

    jump_model = JumpModel(config=_jump_config_with_path(artifact_root), seed=config.seed)
    jump_state = jump_model.train(jump_features, jump_labels)

    return TrainArtifacts(trend=trend_state, vol=vol_state, residual=residual_state, jump=jump_state)


def split_temporal(data_list: List, val_ratio: float = 0.2):
    """
    Split data temporally for train/validation.
    
    Uses the most recent portion for validation to simulate real deployment.
    
    Args:
        data_list: List of data samples
        val_ratio: Fraction to use for validation (default 0.2 = 20%)
        
    Returns:
        (train_data, val_data) tuple
    """
    n = len(data_list)
    n_val = max(1, int(n * val_ratio))
    # Most recent data goes to validation (temporal split)
    return data_list[:-n_val], data_list[-n_val:]


def _simulate_vol_future(iv_points: Dict[str, float], horizon: int, regime_strength: float) -> List[float]:
    base = iv_points.get("iv_7d_atm", 0.45)
    drift = (iv_points.get("iv_30d_atm", base) - base) * 0.5
    path = []
    for step in range(horizon):
        decay = max(0.9, 1 - 0.02 * step)
        path.append(max(0.05, base * decay + drift * regime_strength * 0.1))
    return path


def _label_jumps(normalized_future: Sequence[float]) -> List[float]:
    if len(normalized_future) < 2:
        return [0.0 for _ in normalized_future]
    labels: List[float] = []
    for i in range(1, len(normalized_future)):
        move = abs(normalized_future[i] - normalized_future[i - 1])
        labels.append(1.0 if move > 0.08 else 0.0)
    return labels


def _trend_config_with_path(root: Path) -> TrendConfig:
    cfg = TrendConfig()
    cfg.artifact_path = str(root / "trend_state.json")
    return cfg


def _vol_config_with_path(root: Path) -> VolPathConfig:
    cfg = VolPathConfig()
    cfg.artifact_path = str(root / "vol_path_state.json")
    return cfg


def _residual_config_with_path(root: Path) -> ResidualConfig:
    cfg = ResidualConfig()
    cfg.artifact_path = str(root / "residual_state.json")
    return cfg


def _jump_config_with_path(root: Path) -> JumpConfig:
    cfg = JumpConfig()
    cfg.artifact_path = str(root / "jump_state.json")
    return cfg
