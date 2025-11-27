# Connector scaffolding

- `SyntheticDataConnector` (default) generates aligned features for demos.
- `CsvDataConnector` loads aligned features from CSV for offline runs.
- `CompositeConnector` (in `data/connectors_base.py`) is a thin adapter to stitch real connectors:
  - `OHLCVConnector.fetch(asset_id, start, end) -> MarketFeatureFrame`
  - `OptionsConnector.fetch_iv_surface(asset_id, as_of) -> dict`
  - `MacroConnector.fetch_features(as_of) -> dict` (can include narratives/indices)

To implement a real connector, create classes matching those protocols and pass `connector=CompositeConnector(...)` into `ForecastEngine`. The composite will merge IV surface and macro/narrative features into the base frame returned by OHLCV. Only fields present in `MarketFeatureFrame` are required; missing fields are defaulted upstream.
