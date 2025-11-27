from datetime import datetime
from typing import Dict, Iterable, List

from .schemas import MarketFeatureFrame


class DummyOHLCVConnector:
    """Placeholder OHLCV connector returning empty frame; replace with real implementation."""

    def fetch(self, asset_id: str, start: datetime, end: datetime) -> MarketFeatureFrame:
        raise NotImplementedError("Implement real OHLCV connector")


class DummyOptionsConnector:
    """Placeholder options connector returning IV surface dict."""

    def fetch_iv_surface(self, asset_id: str, as_of: datetime) -> Dict[str, float]:
        raise NotImplementedError("Implement real options connector")


class DummyMacroConnector:
    """Placeholder macro/narrative connector."""

    def fetch_features(self, as_of: datetime) -> Dict[str, float]:
        raise NotImplementedError("Implement real macro connector")
