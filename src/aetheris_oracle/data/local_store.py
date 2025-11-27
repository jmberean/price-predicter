import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, Optional

from .interfaces import DataConnector
from .schemas import MarketFeatureFrame


class LocalFeatureStore(DataConnector):
    """Reads market features from local CSV/Parquet files for offline indexing."""

    def __init__(self, root: str | Path, fmt: str = "csv") -> None:
        self.root = Path(root)
        self.fmt = fmt

    def fetch_window(
        self, asset_id: str, as_of: datetime, window: timedelta
    ) -> MarketFeatureFrame:
        path = self._file_path(asset_id)
        if not path.exists():
            raise FileNotFoundError(f"Local feature file not found: {path}")
        df = self._load_df(path)
        required = {"timestamp", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Local feature file missing required columns: {missing}")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        start = as_of - window
        mask = (df["timestamp"] >= start) & (df["timestamp"] <= as_of)
        window_df = df.loc[mask].sort_values("timestamp")
        if window_df.empty:
            raise ValueError(f"No data in window for {asset_id}")

        timestamps = [
            ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
            for ts in window_df["timestamp"].to_list()
        ]
        closes = window_df["close"].astype(float).tolist()
        volumes = window_df["volume"].astype(float).tolist()
        latest = window_df.iloc[-1]
        iv_points = {
            "iv_7d_atm": float(latest.get("iv_7d_atm", 0.5)),
            "iv_14d_atm": float(latest.get("iv_14d_atm", 0.5)),
            "iv_30d_atm": float(latest.get("iv_30d_atm", 0.5)),
        }
        funding_rate = float(latest.get("funding_rate", 0.0))
        basis = float(latest.get("basis", 0.0))
        order_imbalance = float(latest.get("order_imbalance", 0.0))
        skew = float(latest.get("skew", 0.0))
        narrative_scores = self._narratives_from_columns(latest)
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

    def _file_path(self, asset_id: str) -> Path:
        return self.root / f"{asset_id.replace('/','-')}.{self.fmt}"

    def _load_df(self, path: Path):
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)

    def _narratives_from_columns(self, row: pd.Series) -> Dict[str, float]:
        narratives: Dict[str, float] = {}
        for col, val in row.items():
            if col.lower().startswith("narrative_"):
                narratives[col.replace("narrative_", "")] = float(val)
        if not narratives:
            narratives = {
                "RegulationRisk": float(row.get("RegulationRisk", 0.0)),
                "ETF_Narrative": float(row.get("ETF_Narrative", 0.0)),
                "TechUpgrade": float(row.get("TechUpgrade", 0.0)),
            }
        return narratives
