import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

from .interfaces import DataConnector
from .schemas import MarketFeatureFrame


class CsvDataConnector(DataConnector):
    """Loads aligned market features from a CSV file for offline demos."""

    def __init__(
        self,
        path: str | Path,
        timestamp_field: str = "timestamp",
        tz: Optional[str] = None,
    ) -> None:
        self.path = Path(path)
        self.timestamp_field = timestamp_field
        self.tz = tz

    def fetch_window(
        self, asset_id: str, as_of: datetime, window: timedelta
    ) -> MarketFeatureFrame:
        rows = self._load_rows()
        filtered = [
            row
            for row in rows
            if row.get("asset_id", asset_id) == asset_id
            and self._parse_ts(row[self.timestamp_field]) >= as_of - window
            and self._parse_ts(row[self.timestamp_field]) <= as_of
        ]
        if not filtered:
            raise ValueError(f"No data found for {asset_id} in window")
        filtered.sort(key=lambda r: self._parse_ts(r[self.timestamp_field]))
        timestamps = [self._parse_ts(r[self.timestamp_field]) for r in filtered]
        closes = [float(r.get("close", 0.0)) for r in filtered]
        volumes = [float(r.get("volume", 0.0)) for r in filtered]
        iv_points = {
            "iv_7d_atm": float(filtered[-1].get("iv_7d_atm", 0.5)),
            "iv_14d_atm": float(filtered[-1].get("iv_14d_atm", 0.5)),
            "iv_30d_atm": float(filtered[-1].get("iv_30d_atm", 0.5)),
        }
        funding_rate = float(filtered[-1].get("funding_rate", 0.0))
        basis = float(filtered[-1].get("basis", 0.0))
        order_imbalance = float(filtered[-1].get("order_imbalance", 0.0))
        skew = float(filtered[-1].get("skew", 0.0))
        narrative_scores = {
            "RegulationRisk": float(filtered[-1].get("RegulationRisk", 0.0)),
            "ETF_Narrative": float(filtered[-1].get("ETF_Narrative", 0.0)),
            "TechUpgrade": float(filtered[-1].get("TechUpgrade", 0.0)),
        }
        return MarketFeatureFrame(
            timestamps=timestamps,
            closes=closes,
            volumes=volumes,
            iv_points=iv_points,
            funding_rate= funding_rate,
            basis= basis,
            order_imbalance= order_imbalance,
            narrative_scores= narrative_scores,
            skew= skew,
        )

    def _load_rows(self) -> Iterable[dict]:
        if not self.path.exists():
            raise FileNotFoundError(f"CSV data file not found: {self.path}")
        with self.path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)

    def _parse_ts(self, value: str) -> datetime:
        return datetime.fromisoformat(value)
