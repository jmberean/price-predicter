from datetime import datetime, timedelta
import math

from aetheris_oracle.data.free_connectors import (
    CCXTOHLCVConnector,
    DeribitIVConnector,
    DeribitPerpConnector,
    FreeDataConnector,
    YFinanceMacroConnector,
)
from aetheris_oracle.data.ccxt_perp_connector import CCXTPerpConnector
from aetheris_oracle.data.schemas import MarketFeatureFrame


def test_ccxt_ohlcv_connector_uses_stub_client():
    class StubClient:
        def fetch_ohlcv(self, symbol, timeframe, since, limit):
            ts = int(datetime(2024, 1, 1).timestamp() * 1000)
            return [[ts, 1, 2, 0.5, 100.0, 1000.0]]

    connector = CCXTOHLCVConnector(client=StubClient())
    frame = connector.fetch("BTC-USD", datetime(2024, 1, 1), datetime(2024, 1, 2))
    assert frame.closes[-1] == 100.0
    assert frame.volumes[-1] == 1000.0
    # IV defaults to NaN when not fetched from options connector
    assert math.isnan(frame.iv_points["iv_7d_atm"])


def test_deribit_iv_connector_parses_response(monkeypatch):
    """Test DeribitIVConnector with DVOL API response."""
    class StubResp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            # DVOL endpoint returns data array with [timestamp, open, high, low, close]
            return {"result": {"data": [[1700000000000, 42.0, 43.0, 41.0, 42.0]]}}

    class StubClient:
        def get(self, url, params):
            return StubResp()

    conn = DeribitIVConnector()
    conn._client = StubClient()
    ivs = conn.fetch_iv_surface("BTC-USD", datetime.utcnow())
    # DVOL is 42%, converted to decimal = 0.42
    # 7d is 95% of 30d, 14d is 98% of 30d
    assert abs(ivs["iv_30d_atm"] - 0.42) < 0.001
    assert abs(ivs["iv_7d_atm"] - 0.42 * 0.95) < 0.001


def test_deribit_perp_connector_parses_response():
    """Test DeribitPerpConnector with ticker API response."""
    class StubResp:
        status_code = 200

        def json(self):
            return {
                "result": {
                    "current_funding": 0.0001,  # 0.01% per 8h
                    "mark_price": 50100.0,
                    "index_price": 50000.0,
                    "best_bid_amount": 100.0,
                    "best_ask_amount": 50.0,
                }
            }

    class StubClient:
        def get(self, url, params):
            return StubResp()

    conn = DeribitPerpConnector()
    conn._client = StubClient()
    data = conn.fetch_perp_data("BTC-USD", datetime.utcnow())

    # Funding rate annualized: 0.0001 * 3 * 365 = 0.1095 (10.95%)
    assert abs(data["funding_rate"] - 0.1095) < 0.001

    # Basis: (50100 - 50000) / 50000 = 0.002 (0.2%)
    assert abs(data["basis"] - 0.002) < 0.0001

    # Order imbalance: (100 - 50) / 150 = 0.333
    assert abs(data["order_imbalance"] - 0.333) < 0.01


def test_free_data_connector_merges_sources():
    class StaticOHLCV:
        def fetch(self, asset_id, start, end):
            return MarketFeatureFrame(
                timestamps=[start, end],
                closes=[100.0, 101.0],
                volumes=[1000.0, 1001.0],
                iv_points={"iv_7d_atm": 0.3, "iv_14d_atm": 0.31, "iv_30d_atm": 0.32},
                funding_rate=0.0,
                basis=0.0,
                order_imbalance=0.0,
                narrative_scores={"RegulationRisk": 0.0},
                skew=0.0,
            )

    class StaticIV:
        def fetch_iv_surface(self, asset_id, as_of):
            return {"iv_7d_atm": 0.4}

    macro = YFinanceMacroConnector(data_provider=lambda: {"DXY": 100.0})
    connector = FreeDataConnector(ohlcv=StaticOHLCV(), options=StaticIV(), macro=macro)
    now = datetime.utcnow()
    frame = connector.fetch_window("BTC-USD", as_of=now, window=timedelta(days=1))
    assert frame.iv_points["iv_7d_atm"] == 0.4
    assert frame.narrative_scores["DXY"] == 100.0


def test_ccxt_perp_connector_uses_stub_clients():
    class StubSpot:
        def fetch_ohlcv(self, symbol, timeframe, since):
            ts = int(datetime(2024, 1, 1).timestamp() * 1000)
            return [[ts, 1, 2, 0.5, 100.0, 1000.0]]

    class StubPerp:
        def fetch_funding_rate(self, symbol):
            return {"fundingRate": 0.01}

        def fetch_order_book(self, symbol, limit):
            return {"bids": [[101.0, 10]], "asks": [[103.0, 8]]}

    connector = CCXTPerpConnector(spot_client=StubSpot(), perp_client=StubPerp())
    frame = connector.fetch_window("BTC-USD", datetime(2024, 1, 2), timedelta(days=1))
    assert frame.funding_rate == 0.01
    assert frame.basis != 0.0
