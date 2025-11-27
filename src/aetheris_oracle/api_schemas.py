from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field


class ScenarioPayload(BaseModel):
    iv_multiplier: float = 1.0
    funding_shift: float = 0.0
    basis_shift: float = 0.0
    narrative_overrides: Dict[str, float] = Field(default_factory=dict)
    description: str = "api scenario"


class ForecastRequest(BaseModel):
    asset_id: str = "BTC-USD"
    horizon: int = Field(7, ge=1)
    num_paths: int = Field(500, ge=1)
    thresholds: Sequence[float] = (20000.0, 50000.0)
    scenario_overrides: Optional[ScenarioPayload] = None


class DriverResponse(BaseModel):
    feature: str
    score: float


class ForecastResponse(BaseModel):
    quantile_paths: Dict[int, Dict[str, float]]
    threshold_probabilities: Dict[float, Dict[str, float]]
    metadata: Dict[str, Any]
    drivers: List[DriverResponse]
    scenario_label: str
    coverage: Dict[str, int]
    regime_values: List[float]
