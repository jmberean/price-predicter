from datetime import datetime, timedelta
from typing import Protocol

from .schemas import MarketFeatureFrame


class DataConnector(Protocol):
    """Protocol for data connectors fetching aligned feature windows."""

    def fetch_window(
        self, asset_id: str, as_of: datetime, window: timedelta
    ) -> MarketFeatureFrame:
        ...
