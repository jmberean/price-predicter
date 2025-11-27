import math
import random
from datetime import datetime, timedelta
from typing import Optional

from .schemas import MarketFeatureFrame


class SyntheticDataConnector:
    """Generates synthetic yet plausible crypto market features for demos/tests."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    def fetch_window(
        self, asset_id: str, as_of: datetime, window: timedelta
    ) -> MarketFeatureFrame:
        steps = max(1, window.days)
        base_price = 30000.0
        timestamps = [as_of - timedelta(days=i) for i in range(steps)][::-1]
        closes, volumes = [], []
        price = base_price + self._rng.uniform(-500, 500)
        for i in range(steps):
            drift = self._rng.uniform(-0.02, 0.02)
            noise = self._rng.gauss(0, 0.01)
            price *= 1 + drift + noise
            price = max(price, 5000.0)
            closes.append(price)
            volumes.append(1000 + 50 * math.sin(i / 5) + self._rng.uniform(-50, 50))
        iv_points = {
            "iv_7d_atm": 0.55 + self._rng.uniform(-0.05, 0.05),
            "iv_14d_atm": 0.5 + self._rng.uniform(-0.05, 0.05),
            "iv_30d_atm": 0.45 + self._rng.uniform(-0.05, 0.05),
        }
        funding_rate = 0.01 + self._rng.uniform(-0.01, 0.01)
        basis = 0.02 + self._rng.uniform(-0.02, 0.02)
        order_imbalance = self._rng.uniform(-0.5, 0.5)
        skew = self._rng.uniform(-0.1, 0.1)
        narrative_scores = {
            "RegulationRisk": self._rng.uniform(0, 1),
            "ETF_Narrative": self._rng.uniform(0, 1),
            "TechUpgrade": self._rng.uniform(0, 1),
        }
        return MarketFeatureFrame(
            timestamps=timestamps,
            closes=closes,
            volumes=volumes,
            iv_points=iv_points,
            funding_rate=funding_rate,
            basis=basis,
            order_imbalance=order_imbalance,
            narrative_scores=narrative_scores,
            skew=skew,
        )
