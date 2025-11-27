from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Iterable, Optional, Sequence


DEFAULT_QUANTILES: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


@dataclass
class ScenarioOverrides:
    """Simple what-if adjustments applied before inference."""

    iv_multiplier: float = 1.0
    funding_shift: float = 0.0
    basis_shift: float = 0.0
    narrative_overrides: Dict[str, float] = field(default_factory=dict)
    description: str = "synthetic scenario"


@dataclass
class ForecastConfig:
    asset_id: str = "BTC-USD"
    horizon_days: int = 7
    trailing_window_days: int = 90
    num_paths: int = 1000  # Balance between quality (tail estimates) and speed (inference time)
    quantiles: Sequence[float] = DEFAULT_QUANTILES
    thresholds: Sequence[float] = (20000.0, 50000.0)
    as_of: datetime = field(default_factory=datetime.utcnow)
    scenario: Optional[ScenarioOverrides] = None
    seed: Optional[int] = None
    use_importance_sampling: bool = True  # Enable importance sampling for tail quantiles

    def validate(self) -> None:
        """
        Validate configuration parameters.

        Raises:
            ValueError: If any parameter is invalid
        """
        if self.horizon_days <= 0:
            raise ValueError(f"horizon_days must be positive, got {self.horizon_days}")

        if self.trailing_window_days <= 0:
            raise ValueError(f"trailing_window_days must be positive, got {self.trailing_window_days}")

        if self.num_paths <= 0:
            raise ValueError(f"num_paths must be positive, got {self.num_paths}")

        if not self.asset_id or not self.asset_id.strip():
            raise ValueError("asset_id cannot be empty")

        for q in self.quantiles:
            if not 0 <= q <= 1:
                raise ValueError(f"Quantile must be in [0,1], got {q}")

        for t in self.thresholds:
            if t < 0:
                raise ValueError(f"Threshold must be non-negative, got {t}")

    def horizons(self) -> Iterable[int]:
        for step in range(1, self.horizon_days + 1):
            yield step

    @property
    def trailing_window(self) -> timedelta:
        return timedelta(days=self.trailing_window_days)
