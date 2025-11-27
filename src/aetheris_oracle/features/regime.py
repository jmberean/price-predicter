import statistics
from typing import Dict, List

from ..data.schemas import RegimeVector


def compute_regime_vector(
    closes: List[float],
    iv_points: Dict[str, float],
    funding_rate: float,
    basis: float,
    order_imbalance: float,
    narrative_scores: Dict[str, float],
    skew: float,
) -> RegimeVector:
    """Lightweight regime descriptor built from summary statistics."""

    window_vol = _realized_volatility(closes)
    iv_level = iv_points.get("iv_7d_atm", 0.5)
    iv_term_slope = iv_points.get("iv_30d_atm", iv_level) - iv_points.get(
        "iv_7d_atm", iv_level
    )
    narrative_intensity = sum(narrative_scores.values()) / max(
        len(narrative_scores), 1
    )
    values = [
        window_vol,
        iv_level,
        iv_term_slope,
        funding_rate,
        basis,
        order_imbalance,
        narrative_intensity,
        skew,
    ]
    return RegimeVector(values=values)


def _realized_volatility(closes: List[float]) -> float:
    if len(closes) < 2:
        return 0.0
    log_returns = []
    for i in range(1, len(closes)):
        if closes[i - 1] <= 0:
            continue
        log_returns.append((closes[i] - closes[i - 1]) / closes[i - 1])
    if not log_returns:
        return 0.0
    return statistics.pstdev(log_returns) * (len(log_returns) ** 0.5)
