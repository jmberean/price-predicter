"""
Historical data connector that reads from pre-collected parquet files.

This connector uses data saved by scripts/collect_historical_data.py
instead of fetching from APIs, avoiding rate limits and enabling
training on multi-year historical data.
"""

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

from .interfaces import DataConnector
from .schemas import MarketFeatureFrame


class HistoricalParquetConnector(DataConnector):
    """
    Reads historical market data from parquet files.

    Expects files created by scripts/collect_historical_data.py with columns:
    - open, high, low, close, volume (core OHLCV)
    - realized_vol_7d, realized_vol_30d (derived features)
    - macro_vix, macro_dxy, etc. (macro indicators)
    """

    def __init__(
        self,
        data_dir: str = "data/historical",
        asset_id: str = "BTC-USD",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.asset_id = asset_id
        self._df: Optional[pd.DataFrame] = None
        self._load_data()

    def _load_data(self) -> None:
        """Load parquet data into memory."""
        # Convert asset_id to filename format (BTC-USD -> btc_usd_historical.parquet)
        filename = f"{self.asset_id.replace('-', '_').lower()}_historical.parquet"
        filepath = self.data_dir / filename

        if not filepath.exists():
            # Try CSV fallback
            csv_path = filepath.with_suffix(".csv")
            if csv_path.exists():
                self._df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            else:
                raise FileNotFoundError(
                    f"Historical data not found: {filepath}\n"
                    f"Run: python scripts/collect_historical_data.py --asset {self.asset_id} --years 5"
                )
        else:
            self._df = pd.read_parquet(filepath)

        # Ensure index is datetime
        if not isinstance(self._df.index, pd.DatetimeIndex):
            self._df.index = pd.to_datetime(self._df.index)

        # Make index timezone-naive for easier comparison
        if self._df.index.tz is not None:
            self._df.index = self._df.index.tz_localize(None)

        print(f"Loaded {len(self._df)} days of {self.asset_id} data")
        print(f"  Period: {self._df.index[0].date()} to {self._df.index[-1].date()}")

    def fetch_window(
        self,
        asset_id: str,
        as_of: datetime,
        window: timedelta,
    ) -> MarketFeatureFrame:
        """
        Fetch historical data for a time window.

        Args:
            asset_id: Asset identifier (should match loaded data)
            as_of: End date of window
            window: Time window to fetch

        Returns:
            MarketFeatureFrame with historical data
        """
        if self._df is None:
            raise RuntimeError("Data not loaded")

        # Make as_of timezone-naive
        if as_of.tzinfo is not None:
            as_of = as_of.replace(tzinfo=None)

        start = as_of - window

        # Filter to window
        mask = (self._df.index >= start) & (self._df.index <= as_of)
        window_df = self._df.loc[mask].sort_index()

        if window_df.empty:
            raise ValueError(
                f"No data in window {start.date()} to {as_of.date()} for {asset_id}"
            )

        # Extract timestamps and prices
        timestamps = [ts.to_pydatetime() for ts in window_df.index]
        closes = window_df["close"].astype(float).tolist()
        volumes = window_df["volume"].astype(float).tolist()

        # Get latest row for point-in-time features
        latest = window_df.iloc[-1]

        # IV points - use realized vol as proxy if no IV data
        iv_7d = float(latest.get("realized_vol_7d", 0.5))
        iv_30d = float(latest.get("realized_vol_30d", 0.5))
        iv_points = {
            "iv_7d_atm": iv_7d,
            "iv_14d_atm": (iv_7d + iv_30d) / 2,
            "iv_30d_atm": iv_30d,
        }

        # Funding rate - approximate from momentum
        momentum = float(latest.get("momentum_7d", 0.0))
        funding_rate = momentum * 0.001  # Scale down

        # Basis - approximate from price deviation
        price_to_sma = float(latest.get("price_to_sma20", 0.0))
        basis = price_to_sma * closes[-1] * 0.01

        # Order imbalance - approximate from volume ratio
        volume_ratio = float(latest.get("volume_ratio", 1.0))
        order_imbalance = (volume_ratio - 1.0) * 0.5  # Center around 0

        # Skew - approximate from daily range
        daily_range = float(latest.get("daily_range", 0.0))
        skew = (daily_range - 0.02) * 10  # Normalize

        # Narrative scores - use macro data
        narrative_scores = {
            "RegulationRisk": 0.0,  # No direct proxy
            "ETF_Narrative": 0.0,
            "TechUpgrade": 0.0,
            "MacroRisk": float(latest.get("macro_vix", 20.0)) / 100.0,  # VIX as risk proxy
        }

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

    def get_date_range(self) -> tuple:
        """Get available date range."""
        if self._df is None:
            return (None, None)
        return (self._df.index[0], self._df.index[-1])

    def get_total_days(self) -> int:
        """Get total days of data available."""
        if self._df is None:
            return 0
        return len(self._df)
