from datetime import datetime, timedelta
from typing import Protocol

from .schemas import MarketFeatureFrame


class OHLCVConnector(Protocol):
    def fetch(self, asset_id: str, start: datetime, end: datetime) -> MarketFeatureFrame:
        ...


class OptionsConnector(Protocol):
    def fetch_iv_surface(self, asset_id: str, as_of: datetime) -> dict:
        ...


class MacroConnector(Protocol):
    def fetch_features(self, as_of: datetime) -> dict:
        ...


class CompositeConnector:
    """Composite connector placeholder that can fuse multiple data sources."""

    def __init__(
        self,
        ohlcv: OHLCVConnector,
        options: OptionsConnector | None = None,
        macro: MacroConnector | None = None,
    ) -> None:
        self.ohlcv = ohlcv
        self.options = options
        self.macro = macro

    def fetch_window(
        self, asset_id: str, as_of: datetime, window: timedelta
    ) -> MarketFeatureFrame:
        start = as_of - window
        frame = self.ohlcv.fetch(asset_id=asset_id, start=start, end=as_of)
        if self.options:
            iv_surface = self.options.fetch_iv_surface(asset_id=asset_id, as_of=as_of)
            frame.iv_points.update(iv_surface)
        if self.macro:
            macro_feats = self.macro.fetch_features(as_of=as_of)
            frame.narrative_scores.update(macro_feats)
        return frame
