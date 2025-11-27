from datetime import datetime, timedelta

from aetheris_oracle.data.csv_connector import CsvDataConnector


def test_csv_connector_reads_window(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(
        "timestamp,asset_id,close,volume,iv_7d_atm,iv_14d_atm,iv_30d_atm,funding_rate,basis,order_imbalance,skew,RegulationRisk,ETF_Narrative,TechUpgrade\n"
        "2024-01-01T00:00:00,ASSET,100,1000,0.5,0.5,0.55,0.01,0.02,0.1,0.0,0.4,0.2,0.1\n"
        "2024-01-02T00:00:00,ASSET,105,1100,0.52,0.51,0.56,0.01,0.02,0.05,0.01,0.5,0.3,0.2\n"
    )
    connector = CsvDataConnector(csv_path)
    as_of = datetime.fromisoformat("2024-01-02T00:00:00")
    frame = connector.fetch_window("ASSET", as_of=as_of, window=timedelta(days=2))

    assert frame.closes[-1] == 105.0
    assert frame.iv_points["iv_30d_atm"] == 0.56
    assert frame.narrative_scores["RegulationRisk"] == 0.5
