from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Sequence


@dataclass
class MarketFeatureFrame:
    """Container for aligned features pulled from data sources."""

    timestamps: List[datetime]
    closes: List[float]
    volumes: List[float]
    iv_points: Dict[str, float]
    funding_rate: float
    basis: float
    order_imbalance: float
    narrative_scores: Dict[str, float]
    skew: float

    def latest_price(self) -> float:
        return self.closes[-1]


@dataclass
class RegimeVector:
    values: Sequence[float]


@dataclass
class TrendPath:
    path: List[float]


@dataclass
class VolPath:
    path: List[float]


@dataclass
class JumpPath:
    path: List[float]


@dataclass
class ResidualPaths:
    paths: List[List[float]] = field(default_factory=list)
