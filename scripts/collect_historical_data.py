"""
Historical data collection for 1+ year of crypto market data.

Collects and stores:
- OHLCV data from multiple exchanges via ccxt
- Volatility indices from Deribit
- Macro indicators from yfinance

Usage:
    python scripts/collect_historical_data.py --asset BTC-USD --years 2
    python scripts/collect_historical_data.py --asset ETH-USD --years 1 --output data/historical
    python scripts/collect_historical_data.py --all-assets --years 2
"""

import sys
sys.path.insert(0, "src")

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

# Try to import data sources
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("Warning: yfinance not installed")

try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    print("Warning: ccxt not installed")


def fetch_yfinance_data(
    ticker: str,
    start_date: datetime,
    end_date: datetime,
) -> Optional[pd.DataFrame]:
    """
    Fetch historical data from Yahoo Finance.

    Args:
        ticker: Yahoo Finance ticker (e.g., "BTC-USD")
        start_date: Start date
        end_date: End date

    Returns:
        DataFrame with OHLCV data or None if failed
    """
    if not YFINANCE_AVAILABLE:
        return None

    try:
        print(f"  Fetching {ticker} from yfinance ({start_date.date()} to {end_date.date()})...")
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(start=start_date, end=end_date)

        if df.empty:
            print(f"    No data returned for {ticker}")
            return None

        # Standardize column names
        df = df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })

        # Select only OHLCV columns
        df = df[["open", "high", "low", "close", "volume"]]

        print(f"    Retrieved {len(df)} days of data")
        return df

    except Exception as e:
        print(f"    Error fetching {ticker}: {e}")
        return None


def fetch_ccxt_data(
    symbol: str,
    exchange_id: str = "binance",
    start_date: datetime = None,
    end_date: datetime = None,
    timeframe: str = "1d",
) -> Optional[pd.DataFrame]:
    """
    Fetch historical data from ccxt exchange.

    Args:
        symbol: Trading pair (e.g., "BTC/USDT")
        exchange_id: Exchange identifier
        start_date: Start date
        end_date: End date
        timeframe: Candle timeframe

    Returns:
        DataFrame with OHLCV data or None if failed
    """
    if not CCXT_AVAILABLE:
        return None

    try:
        print(f"  Fetching {symbol} from {exchange_id} ({start_date.date()} to {end_date.date()})...")

        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class()

        # Convert dates to milliseconds
        since = int(start_date.timestamp() * 1000)
        end_ms = int(end_date.timestamp() * 1000)

        all_candles = []
        current_since = since

        # Fetch in batches (most exchanges limit to 500-1000 candles per request)
        while current_since < end_ms:
            try:
                candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=current_since, limit=500)
                if not candles:
                    break

                all_candles.extend(candles)

                # Move to next batch
                last_timestamp = candles[-1][0]
                if last_timestamp <= current_since:
                    break
                current_since = last_timestamp + 1

                # Rate limiting
                time.sleep(0.5)

            except Exception as e:
                print(f"    Batch error: {e}")
                break

        if not all_candles:
            print(f"    No data returned for {symbol}")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp")

        # Filter to date range
        df = df[(df.index >= start_date) & (df.index <= end_date)]

        print(f"    Retrieved {len(df)} days of data")
        return df

    except Exception as e:
        print(f"    Error fetching {symbol} from {exchange_id}: {e}")
        return None


def fetch_macro_data(
    tickers: Dict[str, str],
    start_date: datetime,
    end_date: datetime,
) -> Optional[pd.DataFrame]:
    """
    Fetch macro indicator data.

    Args:
        tickers: Dict of name -> Yahoo ticker
        start_date: Start date
        end_date: End date

    Returns:
        DataFrame with macro indicators
    """
    if not YFINANCE_AVAILABLE:
        return None

    macro_data = {}

    for name, ticker in tickers.items():
        try:
            print(f"  Fetching macro: {name} ({ticker})...")
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(start=start_date, end=end_date)

            if not df.empty:
                macro_data[name] = df["Close"]
                print(f"    Retrieved {len(df)} days")
            else:
                print(f"    No data for {name}")

        except Exception as e:
            print(f"    Error fetching {name}: {e}")

    if not macro_data:
        return None

    return pd.DataFrame(macro_data)


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived features from OHLCV data.

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        DataFrame with additional features
    """
    # Returns
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

    # Volatility (realized)
    df["realized_vol_7d"] = df["returns"].rolling(7).std() * np.sqrt(365)
    df["realized_vol_30d"] = df["returns"].rolling(30).std() * np.sqrt(365)

    # Volume metrics
    df["volume_ma_7d"] = df["volume"].rolling(7).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma_7d"]

    # Price range
    df["daily_range"] = (df["high"] - df["low"]) / df["close"]
    df["range_ma_7d"] = df["daily_range"].rolling(7).mean()

    # Momentum
    df["momentum_7d"] = df["close"] / df["close"].shift(7) - 1
    df["momentum_30d"] = df["close"] / df["close"].shift(30) - 1

    # Moving averages
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    df["sma_200"] = df["close"].rolling(200).mean()

    # Price relative to MAs
    df["price_to_sma20"] = df["close"] / df["sma_20"] - 1
    df["price_to_sma50"] = df["close"] / df["sma_50"] - 1

    return df


def save_dataset(
    df: pd.DataFrame,
    output_path: Path,
    asset_id: str,
    format: str = "parquet",
) -> Path:
    """Save dataset to file."""
    output_path.mkdir(parents=True, exist_ok=True)

    if format == "parquet":
        filepath = output_path / f"{asset_id.replace('-', '_').lower()}_historical.parquet"
        df.to_parquet(filepath)
    else:
        filepath = output_path / f"{asset_id.replace('-', '_').lower()}_historical.csv"
        df.to_csv(filepath)

    return filepath


def collect_asset_data(
    asset_id: str,
    years: float = 2,
    output_dir: str = "data/historical",
    include_macro: bool = True,
) -> Tuple[Optional[pd.DataFrame], Dict]:
    """
    Collect historical data for an asset.

    Args:
        asset_id: Asset identifier (e.g., "BTC-USD")
        years: Years of history to collect
        output_dir: Output directory
        include_macro: Include macro indicators

    Returns:
        Tuple of (DataFrame, metadata dict)
    """
    print(f"\n{'='*60}")
    print(f"Collecting data for {asset_id}")
    print(f"{'='*60}")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(years * 365))

    # Try yfinance first (most reliable for crypto)
    df = fetch_yfinance_data(asset_id, start_date, end_date)

    if df is None:
        # Try ccxt as fallback
        # Convert asset_id to ccxt format
        if "-" in asset_id:
            base, quote = asset_id.split("-")
            ccxt_symbol = f"{base}/{quote.replace('USD', 'USDT')}"
        else:
            ccxt_symbol = f"{asset_id}/USDT"

        df = fetch_ccxt_data(ccxt_symbol, "binance", start_date, end_date)

    if df is None:
        print(f"Failed to collect data for {asset_id}")
        return None, {}

    # Compute derived features
    print("  Computing derived features...")
    df = compute_derived_features(df)

    # Add macro data if requested
    if include_macro and YFINANCE_AVAILABLE:
        print("  Fetching macro indicators...")
        macro_tickers = {
            "VIX": "^VIX",
            "DXY": "DX-Y.NYB",
            "SPY": "SPY",
            "GOLD": "GC=F",
            "US10Y": "^TNX",
        }
        macro_df = fetch_macro_data(macro_tickers, start_date, end_date)

        if macro_df is not None:
            # Align macro data with asset data
            macro_df = macro_df.reindex(df.index, method="ffill")
            for col in macro_df.columns:
                df[f"macro_{col.lower()}"] = macro_df[col]

    # Drop rows with NaN only in core columns (from rolling calculations)
    # Keep rows even if macro data is missing
    core_columns = ['open', 'high', 'low', 'close', 'volume', 'returns', 'log_returns']
    initial_len = len(df)

    # First, forward-fill macro columns
    macro_cols = [c for c in df.columns if c.startswith('macro_')]
    if macro_cols:
        df[macro_cols] = df[macro_cols].ffill().bfill()

    # Drop rows only if core + derived features are NaN
    derived_cols = ['realized_vol_7d', 'realized_vol_30d', 'sma_20', 'sma_50']
    check_cols = [c for c in core_columns + derived_cols if c in df.columns]
    df = df.dropna(subset=check_cols)

    print(f"  Dropped {initial_len - len(df)} rows with NaN values")

    # Save dataset
    output_path = Path(output_dir)
    filepath = save_dataset(df, output_path, asset_id, format="parquet")
    print(f"  Saved to: {filepath}")

    # Also save as CSV for easier inspection
    csv_path = save_dataset(df, output_path, asset_id, format="csv")
    print(f"  CSV backup: {csv_path}")

    # Metadata
    metadata = {
        "asset_id": asset_id,
        "start_date": df.index.min().isoformat(),
        "end_date": df.index.max().isoformat(),
        "n_rows": len(df),
        "n_features": len(df.columns),
        "features": list(df.columns),
        "years": years,
        "collected_at": datetime.now().isoformat(),
    }

    # Save metadata
    metadata_path = output_path / f"{asset_id.replace('-', '_').lower()}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return df, metadata


def generate_data_summary(output_dir: str):
    """Generate summary of all collected datasets."""
    output_path = Path(output_dir)

    datasets = []
    for metadata_file in output_path.glob("*_metadata.json"):
        with open(metadata_file) as f:
            metadata = json.load(f)
            datasets.append(metadata)

    if not datasets:
        print("No datasets found")
        return

    print(f"\n{'='*60}")
    print("DATA COLLECTION SUMMARY")
    print(f"{'='*60}")

    total_rows = 0
    for ds in datasets:
        print(f"\n{ds['asset_id']}:")
        print(f"  Period: {ds['start_date'][:10]} to {ds['end_date'][:10]}")
        print(f"  Rows: {ds['n_rows']:,}")
        print(f"  Features: {ds['n_features']}")
        total_rows += ds['n_rows']

    print(f"\n{'='*60}")
    print(f"Total datasets: {len(datasets)}")
    print(f"Total rows: {total_rows:,}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Collect historical crypto data")
    parser.add_argument("--asset", type=str, default="BTC-USD",
                        help="Asset to collect (e.g., BTC-USD, ETH-USD)")
    parser.add_argument("--all-assets", action="store_true",
                        help="Collect data for all supported assets")
    parser.add_argument("--years", type=float, default=2,
                        help="Years of historical data to collect")
    parser.add_argument("--output", type=str, default="data/historical",
                        help="Output directory")
    parser.add_argument("--no-macro", action="store_true",
                        help="Skip macro indicators")
    parser.add_argument("--summary-only", action="store_true",
                        help="Only show summary of existing data")

    args = parser.parse_args()

    if args.summary_only:
        generate_data_summary(args.output)
        return

    if args.all_assets:
        assets = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"]
    else:
        assets = [args.asset]

    all_metadata = []

    for asset in assets:
        df, metadata = collect_asset_data(
            asset_id=asset,
            years=args.years,
            output_dir=args.output,
            include_macro=not args.no_macro,
        )
        if metadata:
            all_metadata.append(metadata)

    # Generate summary
    if all_metadata:
        generate_data_summary(args.output)

    print("\nData collection complete!")
    print(f"Use --summary-only to view existing datasets")


if __name__ == "__main__":
    main()
