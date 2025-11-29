"""
Free data connectors for fetching market data without API keys.

Uses:
- ccxt (Binance) for OHLCV, with yfinance fallback
- Deribit public API for IV, funding rate, basis
- yfinance for macro data (VIX, DXY)
"""

from datetime import datetime, timedelta
from typing import Dict, Optional
import logging

import httpx
import ccxt
import yfinance as yf

from .connectors_base import CompositeConnector, OHLCVConnector, OptionsConnector, PerpConnector, MacroConnector
from .schemas import MarketFeatureFrame
from ..utils.cache import SimpleCache, make_cache_key, cached_fetch

logger = logging.getLogger(__name__)


class CCXTOHLCVConnector(OHLCVConnector):
    """Fetches OHLCV via ccxt (default Binance) with yfinance fallback."""

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

        timestamps = None
        closes = None
        volumes = None
        data_source = None

        # Try ccxt first
        try:
            candles = self.client.fetch_ohlcv(symbol, timeframe=self.timeframe, since=since, limit=limit)
            if candles:
                timestamps = [datetime.fromtimestamp(c[0] / 1000) for c in candles]
                closes = [float(c[4]) for c in candles]
                volumes = [float(c[5]) for c in candles]
                data_source = "ccxt"
                logger.debug(f"Fetched {len(candles)} candles from ccxt/{self.exchange}")
        except Exception as e:
            logger.debug(f"ccxt fetch failed: {e}")

        # If ccxt fails, try yfinance fallback
        if closes is None and "-" in asset_id:
            try:
                ticker = yf.Ticker(asset_id)
                hist = ticker.history(start=start, end=end)
                if not hist.empty:
                    timestamps = [dt.to_pydatetime() for dt in hist.index]
                    closes = hist['Close'].tolist()
                    volumes = hist['Volume'].tolist()
                    data_source = "yfinance"
                    logger.debug(f"Fetched {len(closes)} candles from yfinance")
            except Exception as e:
                logger.debug(f"yfinance fetch failed: {e}")

        # Last resort: raise error or use mock data with warning
        if closes is None:
            logger.error(f"MOCK DATA for {asset_id} - all data sources failed!")
            closes = [30_000.0 for _ in range(limit)]
            volumes = [1_000.0 for _ in range(limit)]
            timestamps = [start + timedelta(days=i) for i in range(limit)]
            data_source = "mock"

        return MarketFeatureFrame(
            timestamps=timestamps,
            closes=closes,
            volumes=volumes,
            # Default values - will be overwritten by other connectors
            iv_points={"iv_7d_atm": float('nan'), "iv_14d_atm": float('nan'), "iv_30d_atm": float('nan')},
            funding_rate=float('nan'),
            basis=float('nan'),
            order_imbalance=float('nan'),
            narrative_scores={"RegulationRisk": 0.0, "ETF_Narrative": 0.0, "TechUpgrade": 0.0},
            skew=float('nan'),
        )

    def _to_symbol(self, asset_id: str) -> str:
        if "/" in asset_id:
            return asset_id
        if "-" in asset_id:
            base, quote = asset_id.split("-", maxsplit=1)
            return f"{base}/{quote.replace('USD', 'USDT')}"
        return f"{asset_id}/USDT"


class DeribitIVConnector(OptionsConnector):
    """Fetches IV from Deribit public API using DVOL index."""

    def __init__(self, currency: str = "BTC") -> None:
        self.currency = currency
        self._client = httpx.Client(timeout=10)

    def fetch_iv_surface(self, asset_id: str, as_of: datetime) -> Dict[str, float]:
        currency = self._currency_from_asset(asset_id)

        # Try to get DVOL (Deribit Volatility Index)
        # DVOL is the 30-day ATM implied volatility
        dvol = self._fetch_dvol(currency)

        if dvol is not None:
            # DVOL is annualized percentage, convert to decimal
            vol = dvol / 100.0
            logger.debug(f"Fetched DVOL for {currency}: {dvol}%")

            # Approximate term structure (typically slight contango in vol)
            # Short-term vol is usually slightly lower than 30-day
            return {
                "iv_7d_atm": vol * 0.95,   # 7-day slightly lower
                "iv_14d_atm": vol * 0.98,  # 14-day slightly lower
                "iv_30d_atm": vol,         # 30-day is DVOL
            }

        # Fallback: try to get IV from options ticker
        mark_iv = self._fetch_mark_iv_from_options(currency)
        if mark_iv is not None:
            vol = mark_iv
            logger.debug(f"Fetched mark IV from options: {vol:.2%}")
            return {
                "iv_7d_atm": vol,
                "iv_14d_atm": vol,
                "iv_30d_atm": vol,
            }

        logger.warning(f"Could not fetch IV for {currency}, using NaN")
        return {"iv_7d_atm": float('nan'), "iv_14d_atm": float('nan'), "iv_30d_atm": float('nan')}

    def _fetch_dvol(self, currency: str) -> Optional[float]:
        """Fetch DVOL index value."""
        try:
            # Get index price which includes volatility info
            url = "https://www.deribit.com/api/v2/public/get_index_price"
            resp = self._client.get(url, params={"index_name": f"{currency.lower()}_usd"})
            if resp.status_code == 200:
                data = resp.json().get("result", {})
                # Index price endpoint doesn't have DVOL directly
                pass

            # Try the volatility index endpoint with proper params
            url = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
            end_ts = int(datetime.now().timestamp() * 1000)
            start_ts = end_ts - (24 * 60 * 60 * 1000)  # Last 24 hours

            resp = self._client.get(url, params={
                "currency": currency,
                "start_timestamp": start_ts,
                "end_timestamp": end_ts,
                "resolution": "1D"
            })

            if resp.status_code == 200:
                data = resp.json().get("result", {})
                # Get the most recent data point
                if isinstance(data, dict) and "data" in data:
                    points = data["data"]
                    if points:
                        # Each point is [timestamp, open, high, low, close]
                        latest = points[-1]
                        return float(latest[4])  # Close value
                elif isinstance(data, list) and data:
                    return float(data[-1][4])

        except Exception as e:
            logger.debug(f"DVOL fetch failed: {e}")

        return None

    def _fetch_mark_iv_from_options(self, currency: str) -> Optional[float]:
        """Fetch mark IV from ATM options."""
        try:
            # Get instruments to find ATM options
            url = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
            resp = self._client.get(url, params={"currency": currency, "kind": "option"})

            if resp.status_code == 200:
                data = resp.json().get("result", [])
                if data:
                    # Find options with highest volume (likely ATM)
                    # Filter for reasonable mark_iv values
                    ivs = []
                    for opt in data:
                        mark_iv = opt.get("mark_iv")
                        volume = opt.get("volume", 0) or 0
                        if mark_iv and 0 < mark_iv < 500 and volume > 0:
                            ivs.append((volume, mark_iv / 100.0))

                    if ivs:
                        # Volume-weighted average of top options
                        ivs.sort(reverse=True)
                        top_ivs = ivs[:10]
                        total_vol = sum(v for v, _ in top_ivs)
                        if total_vol > 0:
                            weighted_iv = sum(v * iv for v, iv in top_ivs) / total_vol
                            return weighted_iv

        except Exception as e:
            logger.debug(f"Options IV fetch failed: {e}")

        return None

    def _currency_from_asset(self, asset_id: str) -> str:
        if "-" in asset_id:
            return asset_id.split("-", maxsplit=1)[0]
        if "/" in asset_id:
            return asset_id.split("/")[0]
        return self.currency


class DeribitPerpConnector(PerpConnector):
    """Fetches perpetual data from Deribit: funding rate, basis, order imbalance."""

    def __init__(self, currency: str = "BTC") -> None:
        self.currency = currency
        self._client = httpx.Client(timeout=10)

    def fetch_perp_data(self, asset_id: str, as_of: datetime) -> Dict[str, float]:
        currency = self._currency_from_asset(asset_id)
        instrument = f"{currency}-PERPETUAL"

        result = {}

        try:
            # Fetch ticker data for perpetual
            url = "https://www.deribit.com/api/v2/public/ticker"
            resp = self._client.get(url, params={"instrument_name": instrument})

            if resp.status_code == 200:
                data = resp.json().get("result", {})

                # Funding rate (current 8h funding rate)
                current_funding = data.get("current_funding")
                if current_funding is not None:
                    # Annualize: 8h rate * 3 * 365
                    result["funding_rate"] = float(current_funding) * 3 * 365
                    logger.debug(f"Funding rate (annualized): {result['funding_rate']:.4%}")

                # Basis = (mark_price - index_price) / index_price
                mark_price = data.get("mark_price")
                index_price = data.get("index_price")
                if mark_price and index_price and index_price > 0:
                    result["basis"] = (float(mark_price) - float(index_price)) / float(index_price)
                    logger.debug(f"Basis: {result['basis']:.4%}")

                # Order imbalance from bid/ask amounts
                best_bid_amount = data.get("best_bid_amount", 0) or 0
                best_ask_amount = data.get("best_ask_amount", 0) or 0
                total = best_bid_amount + best_ask_amount
                if total > 0:
                    # Normalized: positive = more bids, negative = more asks
                    result["order_imbalance"] = (best_bid_amount - best_ask_amount) / total
                    logger.debug(f"Order imbalance: {result['order_imbalance']:.4f}")

                # Mark IV if available (for perpetual it's usually not, but check)
                mark_iv = data.get("mark_iv")
                if mark_iv:
                    result["mark_iv"] = float(mark_iv) / 100.0

        except Exception as e:
            logger.warning(f"Failed to fetch perp data for {instrument}: {e}")

        return result

    def _currency_from_asset(self, asset_id: str) -> str:
        if "-" in asset_id:
            return asset_id.split("-", maxsplit=1)[0]
        if "/" in asset_id:
            return asset_id.split("/")[0]
        return self.currency


class YFinanceMacroConnector(MacroConnector):
    """Fetches macro/vol proxies using yfinance."""

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
                    logger.debug(f"Fetched {name}: {features[name]:.2f}")
            except Exception as e:
                logger.debug(f"Failed to fetch {name}: {e}")
                continue
        return features


class FreeDataConnector:
    """Composes all free data sources into a single connector."""

    def __init__(
        self,
        ohlcv: Optional[OHLCVConnector] = None,
        options: Optional[OptionsConnector] = None,
        perp: Optional[PerpConnector] = None,
        macro: Optional[MacroConnector] = None,
        enable_cache: bool = True,
        cache_ttl_seconds: int = 3600,
    ) -> None:
        self.composite = CompositeConnector(
            ohlcv=ohlcv or CCXTOHLCVConnector(),
            options=options or DeribitIVConnector(),
            perp=perp or DeribitPerpConnector(),
            macro=macro or YFinanceMacroConnector(),
        )

        self.enable_cache = enable_cache
        if enable_cache:
            self._cache = SimpleCache(max_size=100, ttl_seconds=cache_ttl_seconds)
            logger.info(f"Data caching enabled (TTL: {cache_ttl_seconds}s)")
        else:
            self._cache = None

    def fetch_window(self, asset_id: str, as_of: datetime, window: timedelta) -> MarketFeatureFrame:
        """Fetch market data with optional caching."""
        if not self.enable_cache or self._cache is None:
            return self.composite.fetch_window(asset_id=asset_id, as_of=as_of, window=window)

        start = as_of - window
        cache_key = make_cache_key(asset_id, start, as_of)

        def fetch_fn():
            logger.debug(f"Fetching data for {asset_id} from {start} to {as_of}")
            return self.composite.fetch_window(asset_id=asset_id, as_of=as_of, window=window)

        return cached_fetch(self._cache, cache_key, fetch_fn)

    def get_cache_stats(self) -> Optional[dict]:
        """Get cache statistics."""
        if self._cache is not None:
            return self._cache.stats()
        return None

    def clear_cache(self) -> None:
        """Clear cache if caching is enabled."""
        if self._cache is not None:
            self._cache.clear()
