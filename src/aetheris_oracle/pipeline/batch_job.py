"""Batch forecast runner for multiple assets/horizons with calibration persistence."""

from pathlib import Path
from typing import Dict, Iterable, Tuple

from ..config import ForecastConfig
from .calibration import CalibrationEngine
from .forecast import ForecastEngine, ForecastResult


def run_batch(configs: Iterable[ForecastConfig], calibration_path: str | None = None) -> dict:
    calibration = None
    if calibration_path and Path(calibration_path).exists():
        calibration = CalibrationEngine.load(calibration_path)

    engine = ForecastEngine(calibration=calibration)
    results: Dict[str, ForecastResult] = {}
    errors: Dict[str, str] = {}
    for cfg in configs:
        key = f"{cfg.asset_id}:{cfg.as_of.isoformat()}:{cfg.horizon_days}"
        try:
            results[key] = engine.forecast(cfg)
        except Exception as exc:
            errors[key] = str(exc)

    if calibration_path:
        engine.calibration.save(calibration_path)

    return {"results": results, "errors": errors}
