from datetime import datetime, timedelta

from aetheris_oracle.data.free_connectors import (
    CCXTOHLCVConnector,
    DeribitIVConnector,
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
    assert frame.iv_points["iv_7d_atm"] == 0.5


def test_deribit_iv_connector_parses_response(monkeypatch):
    class StubResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"result": {"current_volatility": 0.42}}

    class StubClient:
        def get(self, url, params):
            return StubResp()

    conn = DeribitIVConnector()
    conn._client = StubClient()
    ivs = conn.fetch_iv_surface("BTC-USD", datetime.utcnow())
    assert ivs["iv_7d_atm"] == 0.42


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
