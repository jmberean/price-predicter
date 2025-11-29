from datetime import datetime, timedelta
from typing import Protocol, Optional
import logging

from .schemas import MarketFeatureFrame

logger = logging.getLogger(__name__)


class OHLCVConnector(Protocol):
    def fetch(self, asset_id: str, start: datetime, end: datetime) -> MarketFeatureFrame:
        ...


class OptionsConnector(Protocol):
    def fetch_iv_surface(self, asset_id: str, as_of: datetime) -> dict:
        ...


class PerpConnector(Protocol):
    """Fetches perpetual/futures data: funding rate, basis, order imbalance."""
    def fetch_perp_data(self, asset_id: str, as_of: datetime) -> dict:
        ...


class MacroConnector(Protocol):
    def fetch_features(self, as_of: datetime) -> dict:
        ...


class CompositeConnector:
    """Composite connector that fuses multiple data sources."""

    def __init__(
        self,
        ohlcv: OHLCVConnector,
        options: Optional[OptionsConnector] = None,
        perp: Optional[PerpConnector] = None,
        macro: Optional[MacroConnector] = None,
    ) -> None:
        self.ohlcv = ohlcv
        self.options = options
        self.perp = perp
        self.macro = macro

    def fetch_window(
        self, asset_id: str, as_of: datetime, window: timedelta
    ) -> MarketFeatureFrame:
        start = as_of - window
        frame = self.ohlcv.fetch(asset_id=asset_id, start=start, end=as_of)

        # Fetch IV surface from options connector
        if self.options:
            try:
                iv_surface = self.options.fetch_iv_surface(asset_id=asset_id, as_of=as_of)
                frame.iv_points.update(iv_surface)
            except Exception as e:
                logger.warning(f"Failed to fetch IV surface: {e}")

        # Fetch perp data (funding rate, basis, order imbalance)
        if self.perp:
            try:
                perp_data = self.perp.fetch_perp_data(asset_id=asset_id, as_of=as_of)
                if "funding_rate" in perp_data:
                    frame.funding_rate = perp_data["funding_rate"]
                if "basis" in perp_data:
                    frame.basis = perp_data["basis"]
                if "order_imbalance" in perp_data:
                    frame.order_imbalance = perp_data["order_imbalance"]
                if "mark_iv" in perp_data:
                    # Use mark IV from perp if available
                    frame.iv_points["mark_iv"] = perp_data["mark_iv"]
            except Exception as e:
                logger.warning(f"Failed to fetch perp data: {e}")

        # Fetch macro features
        if self.macro:
            try:
                macro_feats = self.macro.fetch_features(as_of=as_of)
                frame.narrative_scores.update(macro_feats)
            except Exception as e:
                logger.warning(f"Failed to fetch macro features: {e}")

        return frame
