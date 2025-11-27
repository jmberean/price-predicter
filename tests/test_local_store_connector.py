from datetime import datetime, timedelta

import pandas as pd

from aetheris_oracle.data.local_store import LocalFeatureStore


def test_local_feature_store_reads_window(tmp_path):
    df = pd.DataFrame(
        {
            "timestamp": ["2024-01-01", "2024-01-02"],
            "close": [100.0, 105.0],
            "volume": [1000, 1100],
            "iv_7d_atm": [0.5, 0.55],
            "funding_rate": [0.01, 0.02],
            "basis": [0.02, 0.03],
            "order_imbalance": [0.0, 0.1],
            "skew": [0.0, 0.05],
            "narrative_RegulationRisk": [0.3, 0.4],
        }
    )
    path = tmp_path / "BTC-USD.csv"
    df.to_csv(path, index=False)

    store = LocalFeatureStore(tmp_path)
    frame = store.fetch_window("BTC-USD", as_of=datetime.fromisoformat("2024-01-02"), window=timedelta(days=2))

    assert frame.closes[-1] == 105.0
    assert frame.iv_points["iv_7d_atm"] == 0.55
    assert frame.narrative_scores["RegulationRisk"] == 0.4
