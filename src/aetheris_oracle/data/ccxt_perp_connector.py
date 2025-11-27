import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import ccxt

from .interfaces import DataConnector
from .schemas import MarketFeatureFrame


class CCXTPerpConnector(DataConnector):
    """Fetches spot OHLCV plus perp funding/basis/order book via ccxt with simple caching."""

    def __init__(
        self,
        spot_exchange: str = "binance",
        perp_exchange: str = "binanceusdm",
        timeframe: str = "1d",
        cache_ttl_sec: int = 30,
        spot_client: Optional[ccxt.Exchange] = None,
        perp_client: Optional[ccxt.Exchange] = None,
    ) -> None:
        self.timeframe = timeframe
        self.cache_ttl = cache_ttl_sec
        self.spot_client = spot_client or getattr(ccxt, spot_exchange)()
        self.perp_client = perp_client or getattr(ccxt, perp_exchange)()
        self._cache: Dict[str, tuple[float, Any]] = {}

    def fetch_window(
        self, asset_id: str, as_of: datetime, window: timedelta
    ) -> MarketFeatureFrame:
        spot_symbol = self._to_symbol(asset_id, is_perp=False)
        perp_symbol = self._to_symbol(asset_id, is_perp=True)

        ohlcv = self._fetch_cached(
            f"ohlcv:{spot_symbol}",
            lambda: self.spot_client.fetch_ohlcv(
                spot_symbol, timeframe=self.timeframe, since=int((as_of - window).timestamp() * 1000)
            ),
        )
        timestamps, closes, volumes = self._parse_ohlcv(ohlcv, as_of, window)

        funding_rate = self._fetch_funding(perp_symbol)
        order_book = self._fetch_cached(
            f"orderbook:{perp_symbol}",
            lambda: self.perp_client.fetch_order_book(perp_symbol, limit=10),
        )
        basis, order_imbalance = self._compute_basis_and_imbalance(order_book, closes[-1])

        # No live options here; leave iv_points neutral
        iv_points = {
            "iv_7d_atm": 0.5,
            "iv_14d_atm": 0.5,
            "iv_30d_atm": 0.5,
        }
        narrative_scores = {"RegulationRisk": 0.0, "ETF_Narrative": 0.0, "TechUpgrade": 0.0}
        return MarketFeatureFrame(
            timestamps=timestamps,
            closes=closes,
            volumes=volumes,
            iv_points=iv_points,
            funding_rate=funding_rate,
            basis=basis,
            order_imbalance=order_imbalance,
            narrative_scores=narrative_scores,
            skew=0.0,
        )

    def _fetch_funding(self, perp_symbol: str) -> float:
        try:
            fr = self._fetch_cached(
                f"funding:{perp_symbol}",
                lambda: getattr(self.perp_client, "fetch_funding_rate")(perp_symbol),
            )
            return float(fr.get("fundingRate", fr.get("fundingRate", 0.0)))
        except Exception:
            return 0.0

    def _fetch_cached(self, key: str, fn):
        now = time.time()
        if key in self._cache:
            ts, val = self._cache[key]
            if now - ts < self.cache_ttl:
                return val
        val = fn()
        self._cache[key] = (now, val)
        return val

    def _parse_ohlcv(self, ohlcv: Any, as_of: datetime, window: timedelta):
        if not ohlcv:
            limit = max(1, window.days)
            timestamps = [as_of - timedelta(days=i) for i in range(limit)][::-1]
            closes = [30_000.0 for _ in range(limit)]
            volumes = [1_000.0 for _ in range(limit)]
            return timestamps, closes, volumes
        filtered = [
            c for c in ohlcv if c and (as_of - window).timestamp() * 1000 <= c[0] <= as_of.timestamp() * 1000
        ]
        if not filtered:
            filtered = ohlcv[-window.days :] if hasattr(window, "days") else ohlcv
        timestamps = [datetime.fromtimestamp(c[0] / 1000) for c in filtered]
        closes = [float(c[4]) for c in filtered]
        volumes = [float(c[5]) for c in filtered]
        return timestamps, closes, volumes

    def _compute_basis_and_imbalance(self, order_book: Any, spot_price: float) -> tuple[float, float]:
        try:
            bids = order_book.get("bids", [])
            asks = order_book.get("asks", [])
            bid_vol = sum(b[1] for b in bids)
            ask_vol = sum(a[1] for a in asks)
            order_imbalance = (bid_vol - ask_vol) / max(bid_vol + ask_vol, 1e-6)
            best_bid = bids[0][0] if bids else spot_price
            best_ask = asks[0][0] if asks else spot_price
            perp_mid = (best_bid + best_ask) / 2
            basis = (perp_mid - spot_price) / max(spot_price, 1e-6)
            return basis, order_imbalance
        except Exception:
            return 0.0, 0.0

    def _to_symbol(self, asset_id: str, is_perp: bool) -> str:
        if "/" in asset_id:
            return asset_id
        base, quote = asset_id.replace("-", "/").split("/")
        if quote.upper() == "USD":
            quote = "USDT"
        symbol = f"{base}/{quote}"
        if is_perp:
            return f"{symbol}:USDT"
        return symbol
