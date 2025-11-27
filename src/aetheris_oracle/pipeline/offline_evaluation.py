"""Offline evaluation for walk-forward calibration/coverage tracking."""

from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, Iterable, List, Tuple

from ..config import ForecastConfig
from .forecast import ForecastEngine, ForecastResult


@dataclass
class EvaluationResult:
    results: List[Tuple[ForecastConfig, ForecastResult]]
    crps: float
    coverage: Dict[str, float]

    def coverage_summary(self) -> dict:
        summary = {}
        for cfg, res in self.results:
            horizon = max(res.quantile_paths.keys())
            qs = res.quantile_paths[horizon]
            summary[f"{cfg.asset_id}:{horizon}"] = {
                "width": qs.get(0.9, 0.0) - qs.get(0.1, 0.0),
                "median": qs.get(0.5, 0.0),
            }
        summary["crps"] = self.crps
        summary["coverage_rate"] = self.coverage.get("rate", 0.0)
        return summary


def run_walk_forward(configs: Iterable[ForecastConfig], engine: ForecastEngine | None = None) -> EvaluationResult:
    eng = engine or ForecastEngine()
    outputs: List[Tuple[ForecastConfig, ForecastResult]] = []
    crps_accum = 0.0
    total = 0
    coverage_hits = 0
    for cfg in configs:
        res = eng.forecast(cfg)
        outputs.append((cfg, res))
        horizon = max(res.quantile_paths.keys())
        future_frame = eng.data_connector.fetch_window(
            asset_id=cfg.asset_id,
            as_of=cfg.as_of + timedelta(days=horizon),
            window=cfg.trailing_window + timedelta(days=horizon),
        )
        realized = future_frame.latest_price()
        crps_accum += _approx_crps(res.quantile_paths[horizon], realized)
        total += 1
        if res.quantile_paths[horizon].get(0.1, realized) <= realized <= res.quantile_paths[horizon].get(0.9, realized):
            coverage_hits += 1
        eng.update_calibration_with_realized(res, realized, horizon=horizon)
    coverage = {"hits": coverage_hits, "total": total, "rate": coverage_hits / total if total else 0.0}
    return EvaluationResult(results=outputs, crps=crps_accum / max(total, 1), coverage=coverage)


def _approx_crps(quantiles: Dict[float, float], realized: float) -> float:
    # Simple CRPS-like score using lower/median/upper quantiles
    median = quantiles.get(0.5, realized)
    lower = quantiles.get(0.1, median)
    upper = quantiles.get(0.9, median)
    width = upper - lower
    if width <= 0:
        return abs(median - realized)
    below = max(0.0, lower - realized)
    above = max(0.0, realized - upper)
    inside = abs(median - realized)
    return (below + above + inside) / 3
