from dataclasses import asdict
from typing import Any, Dict, List

from .api_schemas import ForecastRequest, ForecastResponse
from .config import ForecastConfig, ScenarioOverrides
from .pipeline.forecast import ForecastEngine, ForecastResult


def forecast_endpoint(
    payload: Dict[str, Any],
    engine: ForecastEngine | None = None,
) -> Dict[str, Any]:
    """Minimal forecast API handler. Validate payload and return serializable result."""

    request = ForecastRequest.model_validate(payload)
    cfg = _build_config(request)
    eng = engine or ForecastEngine()
    try:
        result = eng.forecast(cfg)
        return _serialize_result(result)
    except Exception as exc:
        if hasattr(eng, "metrics"):
            eng.metrics.emit_error("forecast_failed", {"detail": str(exc)})
        raise


def _build_config(request: ForecastRequest) -> ForecastConfig:
    scenario = None
    if request.scenario_overrides:
        scenario = ScenarioOverrides(
            iv_multiplier=request.scenario_overrides.iv_multiplier,
            funding_shift=request.scenario_overrides.funding_shift,
            basis_shift=request.scenario_overrides.basis_shift,
            narrative_overrides=request.scenario_overrides.narrative_overrides,
            description=request.scenario_overrides.description,
        )
    return ForecastConfig(
        asset_id=request.asset_id,
        horizon_days=request.horizon,
        num_paths=request.num_paths,
        thresholds=tuple(request.thresholds),
        scenario=scenario,
    )


def _serialize_result(result: ForecastResult) -> Dict[str, Any]:
    scenario_label = result.metadata.get("scenario_label") or result.metadata.get("scenario", "base")
    return {
        "quantile_paths": {
            step: {str(q): v for q, v in qs.items()} for step, qs in result.quantile_paths.items()
        },
        "threshold_probabilities": result.threshold_probabilities,
        "metadata": result.metadata,
        "drivers": _serialize_drivers(result.drivers),
        "scenario_label": scenario_label,
        "coverage": {
            "hits": int(result.metadata.get("coverage_hits", 0)),
            "total": int(result.metadata.get("coverage_total", 0)),
        },
        "regime_values": list(result.regime.values) if result.regime else [],
    }


def _serialize_drivers(drivers: List[tuple[str, float]]) -> List[Dict[str, float]]:
    return [{"feature": feature, "score": score} for feature, score in drivers]
