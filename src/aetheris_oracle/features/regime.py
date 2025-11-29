import math
import statistics
from typing import Dict, List

from ..data.schemas import RegimeVector


def _safe_float(value: float, default: float = 0.0) -> float:
    """Return default if value is NaN or None."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    return value


def _safe_get(d: Dict[str, float], key: str, default: float) -> float:
    """Get value from dict, returning default if missing or NaN."""
    value = d.get(key)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    return value


def compute_regime_vector(
    closes: List[float],
    iv_points: Dict[str, float],
    funding_rate: float,
    basis: float,
    order_imbalance: float,
    narrative_scores: Dict[str, float],
    skew: float,
) -> RegimeVector:
    """
    Lightweight regime descriptor built from summary statistics.
    
    IMPORTANT: The 'closes' list must contain ONLY historical data relative to
    the forecast date. Including future data will cause look-ahead bias and
    inflate backtest performance unrealistically.
    
    Args:
        closes: Historical closing prices (strictly before forecast date)
        iv_points: Implied volatility term structure
        funding_rate: Current funding rate
        basis: Current basis (futures - spot)
        order_imbalance: Current order book imbalance
        narrative_scores: Sentiment/narrative scores
        skew: IV skew metric
        
    Returns:
        RegimeVector with computed features
    """
    if not closes:
        raise ValueError("closes list cannot be empty")
    
    window_vol = _realized_volatility(closes)
    iv_level = _safe_get(iv_points, "iv_7d_atm", 0.5)
    iv_30d = _safe_get(iv_points, "iv_30d_atm", iv_level)
    iv_term_slope = iv_30d - iv_level

    narrative_intensity = sum(narrative_scores.values()) / max(
        len(narrative_scores), 1
    )

    values = [
        window_vol,
        iv_level,
        iv_term_slope,
        _safe_float(funding_rate, 0.0),
        _safe_float(basis, 0.0),
        _safe_float(order_imbalance, 0.0),
        narrative_intensity,
        _safe_float(skew, 0.0),
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
