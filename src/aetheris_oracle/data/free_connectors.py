from datetime import datetime, timedelta
from typing import Dict, Optional
import logging

import httpx
import ccxt
import yfinance as yf

from .connectors_base import CompositeConnector, OHLCVConnector, OptionsConnector, MacroConnector
from .schemas import MarketFeatureFrame
from ..utils.cache import SimpleCache, make_cache_key, cached_fetch

logger = logging.getLogger(__name__)


class CCXTOHLCVConnector(OHLCVConnector):
    """Fetches OHLCV via ccxt (default Binance) with simple fallbacks."""

    def __init__(
        self,
        exchange: str = "binance",
        timeframe: str = "1d",
        client: Optional[ccxt.Exchange] = None,
    ) -> None:
        self.exchange = exchange
        self.timeframe = timeframe
        self.client = client or getattr(ccxt, exchange)()

    def fetch(self, asset_id: str, start: datetime, end: datetime) -> MarketFeatureFrame:
        symbol = self._to_symbol(asset_id)
        since = int(start.timestamp() * 1000)
        limit = min(500, max(1, (end - start).days + 1))

        # Try ccxt first
        try:
            candles = self.client.fetch_ohlcv(symbol, timeframe=self.timeframe, since=since, limit=limit)
        except Exception:
            candles = []

        # If ccxt fails (e.g., geo-blocked), try yfinance fallback
        if not candles and "-" in asset_id:
            try:
                ticker = yf.Ticker(asset_id)  # e.g., "BTC-USD"
                # Use start/end parameters to get historical data, not just recent
                hist = ticker.history(start=start, end=end)
                if not hist.empty:
                    timestamps = [dt.to_pydatetime() for dt in hist.index]
                    closes = hist['Close'].tolist()
                    volumes = hist['Volume'].tolist()
                    candles = "yfinance_fallback"  # Mark as successful
            except Exception:
                candles = []

        # Last resort: mock data
        if not candles:
            closes = [30_000.0 for _ in range(limit)]
            volumes = [1_000.0 for _ in range(limit)]
            timestamps = [start + timedelta(days=i) for i in range(limit)]
        elif candles != "yfinance_fallback":
            timestamps = [datetime.fromtimestamp(c[0] / 1000) for c in candles]
            closes = [float(c[4]) for c in candles]
            volumes = [float(c[5]) for c in candles]

        return MarketFeatureFrame(
            timestamps=timestamps,
            closes=closes,
            volumes=volumes,
            iv_points={"iv_7d_atm": 0.5, "iv_14d_atm": 0.5, "iv_30d_atm": 0.5},
            funding_rate=0.0,
            basis=0.0,
            order_imbalance=0.0,
            narrative_scores={"RegulationRisk": 0.0, "ETF_Narrative": 0.0, "TechUpgrade": 0.0},
            skew=0.0,
        )

    def _to_symbol(self, asset_id: str) -> str:
        if "/" in asset_id:
            return asset_id
        if "-" in asset_id:
            base, quote = asset_id.split("-", maxsplit=1)
            return f"{base}/{quote.replace('USD', 'USDT')}"
        return f"{asset_id}/USDT"


class DeribitIVConnector(OptionsConnector):
    """Fetches IV points from Deribit public API, with graceful fallback."""

    def __init__(self, currency: str = "BTC") -> None:
        self.currency = currency
        self._client = httpx.Client(timeout=5)

    def fetch_iv_surface(self, asset_id: str, as_of: datetime) -> Dict[str, float]:
        url = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
        try:
            resp = self._client.get(url, params={"currency": self._currency_from_asset(asset_id)})
            resp.raise_for_status()
            data = resp.json().get("result", {})
            vol = float(data.get("current_volatility", 0.5))
            return {
                "iv_7d_atm": vol,
                "iv_14d_atm": vol,
                "iv_30d_atm": vol,
            }
        except Exception:
            return {"iv_7d_atm": 0.5, "iv_14d_atm": 0.5, "iv_30d_atm": 0.5}

    def _currency_from_asset(self, asset_id: str) -> str:
        if "-" in asset_id:
            return asset_id.split("-", maxsplit=1)[0]
        if "/" in asset_id:
            return asset_id.split("/")[0]
        return self.currency


class YFinanceMacroConnector(MacroConnector):
    """Fetches macro/vol proxies using yfinance; accepts data_provider for tests."""

    def __init__(self, tickers: Optional[Dict[str, str]] = None, data_provider=None) -> None:
        self.tickers = tickers or {"VIX": "^VIX", "DXY": "DX-Y.NYB"}
        self._data_provider = data_provider

    def fetch_features(self, as_of: datetime) -> Dict[str, float]:
        if self._data_provider:
            return dict(self._data_provider())
        features: Dict[str, float] = {}
        for name, ticker in self.tickers.items():
            try:
                series = yf.Ticker(ticker).history(period="5d")["Close"]
                if not series.empty:
                    features[name] = float(series.iloc[-1])
            except Exception:
                continue
        return features


class FreeDataConnector:
    """Composes ccxt OHLCV with optional IV/macro for offline-friendly default use."""

    def __init__(
        self,
        ohlcv: Optional[OHLCVConnector] = None,
        options: Optional[OptionsConnector] = None,
        macro: Optional[MacroConnector] = None,
        enable_cache: bool = True,
        cache_ttl_seconds: int = 3600,  # 1 hour default
    ) -> None:
        self.composite = CompositeConnector(
            ohlcv=ohlcv or CCXTOHLCVConnector(),
            options=options or DeribitIVConnector(),
            macro=macro or YFinanceMacroConnector(),
        )

        # Initialize cache
        self.enable_cache = enable_cache
        if enable_cache:
            self._cache = SimpleCache(max_size=100, ttl_seconds=cache_ttl_seconds)
            logger.info(f"Data caching enabled (TTL: {cache_ttl_seconds}s)")
        else:
            self._cache = None

    def fetch_window(self, asset_id: str, as_of: datetime, window: timedelta) -> MarketFeatureFrame:
        """
        Fetch market data with optional caching.

        Args:
            asset_id: Asset identifier
            as_of: End datetime
            window: Time window to fetch

        Returns:
            MarketFeatureFrame with historical data
        """
        if not self.enable_cache or self._cache is None:
            # No caching - direct fetch
            return self.composite.fetch_window(asset_id=asset_id, as_of=as_of, window=window)

        # Create cache key
        start = as_of - window
        cache_key = make_cache_key(asset_id, start, as_of)

        # Try to fetch from cache, or fetch and cache
        def fetch_fn():
            logger.debug(f"Fetching data for {asset_id} from {start} to {as_of}")
            return self.composite.fetch_window(asset_id=asset_id, as_of=as_of, window=window)

        return cached_fetch(self._cache, cache_key, fetch_fn)

    def get_cache_stats(self) -> Optional[dict]:
        """
        Get cache statistics.

        Returns:
            Cache stats dictionary if caching is enabled, None otherwise
        """
        if self._cache is not None:
            return self._cache.stats()
        return None

    def clear_cache(self) -> None:
        """Clear cache if caching is enabled."""
        if self._cache is not None:
            self._cache.clear()
