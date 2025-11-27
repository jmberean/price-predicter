import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Tuple

from ..data.schemas import RegimeVector

logger = logging.getLogger(__name__)


RegimeBucket = str
HorizonBucket = str


@dataclass
class BucketStats:
    """Tracks scaling for a specific (regime, horizon) bucket."""

    width_scale: float = 1.0
    count: int = 0


@dataclass
class CalibrationState:
    """Holds global and bucket-specific calibration scaling factors."""

    global_width_scale: float = 1.05
    bucket_scales: Dict[Tuple[RegimeBucket, HorizonBucket], BucketStats] = field(
        default_factory=dict
    )
    version: int = 1
    coverage: Dict[Tuple[RegimeBucket, HorizonBucket], Dict[str, float]] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict:
        return {
            "global_width_scale": self.global_width_scale,
            "bucket_scales": {
                f"{regime}|{horizon}": asdict(stats) for (regime, horizon), stats in self.bucket_scales.items()
            },
            "version": self.version,
            "coverage": {
                f"{regime}|{horizon}": cov for (regime, horizon), cov in self.coverage.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: Dict) -> "CalibrationState":
        bucket_scales: Dict[Tuple[RegimeBucket, HorizonBucket], BucketStats] = {}
        for key, stats in payload.get("bucket_scales", {}).items():
            regime, horizon = key.split("|", maxsplit=1)
            bucket_scales[(regime, horizon)] = BucketStats(
                width_scale=stats.get("width_scale", 1.0), count=stats.get("count", 0)
            )
        coverage: Dict[Tuple[RegimeBucket, HorizonBucket], Dict[str, float]] = {}
        for key, cov in payload.get("coverage", {}).items():
            regime, horizon = key.split("|", maxsplit=1)
            coverage[(regime, horizon)] = cov
        return cls(
            global_width_scale=payload.get("global_width_scale", 1.05),
            bucket_scales=bucket_scales,
            version=payload.get("version", 1),
            coverage=coverage,
        )


class CalibrationEngine:
    """Applies regime- and horizon-aware conformal-like widening around median."""

    def __init__(self, state: CalibrationState | None = None) -> None:
        self.state = state or CalibrationState()

    def calibrate_quantiles(
        self,
        quantiles: Dict[float, float],
        regime: RegimeVector | None = None,
        horizon: int | None = None,
    ) -> Dict[float, float]:
        if 0.5 not in quantiles:
            return quantiles
        median = quantiles[0.5]
        width_scale = self.state.global_width_scale
        if regime is not None and horizon is not None:
            bucket_key = (self._regime_bucket(regime), self._horizon_bucket(horizon))
            bucket = self.state.bucket_scales.get(bucket_key)
            if bucket:
                width_scale *= bucket.width_scale

        calibrated: Dict[float, float] = {}
        for q, value in quantiles.items():
            if q == 0.5:
                calibrated[q] = value
                continue
            delta = value - median
            calibrated[q] = median + delta * width_scale
        return calibrated

    def update_with_outcome(
        self,
        predicted_quantiles: Dict[float, float],
        realized: float,
        regime: RegimeVector,
        horizon: int,
        step: float = 0.02,
    ) -> None:
        """Lightweight adaptive widening/narrowing based on coverage misses."""

        if not predicted_quantiles:
            return
        lower_q = min((q for q in predicted_quantiles if q < 0.5), default=None)
        upper_q = max((q for q in predicted_quantiles if q > 0.5), default=None)
        if lower_q is None or upper_q is None:
            return
        lower = predicted_quantiles[lower_q]
        upper = predicted_quantiles[upper_q]
        bucket_key = (self._regime_bucket(regime), self._horizon_bucket(horizon))
        bucket = self.state.bucket_scales.setdefault(bucket_key, BucketStats())

        if realized < lower or realized > upper:
            bucket.width_scale *= 1 + step
            self._update_coverage(bucket_key, hit=False)
        else:
            bucket.width_scale *= max(0.9, 1 - step / 2)
            self._update_coverage(bucket_key, hit=True)
        bucket.count += 1
        self.state.version += 1

    def save(self, path: str | Path) -> None:
        """
        Save calibration state to JSON file.

        Args:
            path: File path to save to

        Raises:
            ValueError: If path is invalid or contains path traversal attempts
        """
        path_obj = Path(path).resolve()

        # Validate path doesn't contain suspicious patterns
        path_str = str(path_obj)
        if ".." in path_str or path_str.startswith("/etc") or path_str.startswith("/sys"):
            raise ValueError(f"Invalid path: {path}. Path traversal not allowed.")

        # Create parent directories if needed
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Write calibration state
        path_obj.write_text(json.dumps(self.state.to_dict(), indent=2))
        logger.info(f"Saved calibration state to {path_obj}")

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationEngine":
        """
        Load calibration state from JSON file.

        Args:
            path: File path to load from

        Returns:
            CalibrationEngine with loaded state

        Raises:
            ValueError: If path is invalid
            FileNotFoundError: If file doesn't exist
        """
        path_obj = Path(path).resolve()

        # Validate path doesn't contain suspicious patterns
        path_str = str(path_obj)
        if ".." in path_str or path_str.startswith("/etc") or path_str.startswith("/sys"):
            raise ValueError(f"Invalid path: {path}. Path traversal not allowed.")

        if not path_obj.exists():
            raise FileNotFoundError(f"Calibration file not found: {path_obj}")

        payload = json.loads(path_obj.read_text())
        logger.info(f"Loaded calibration state from {path_obj}")
        return cls(state=CalibrationState.from_dict(payload))

    def _regime_bucket(self, regime: RegimeVector) -> RegimeBucket:
        vol = regime.values[0] if regime.values else 0.0
        if vol < 0.05:
            return "calm"
        if vol < 0.15:
            return "normal"
        return "volatile"

    def _horizon_bucket(self, horizon: int) -> HorizonBucket:
        if horizon <= 3:
            return "short"
        if horizon <= 7:
            return "mid"
        return "long"

    def _update_coverage(self, bucket_key: Tuple[RegimeBucket, HorizonBucket], hit: bool) -> None:
        stats = self.state.coverage.setdefault(bucket_key, {"hits": 0, "total": 0})
        stats["total"] += 1
        if hit:
            stats["hits"] += 1
