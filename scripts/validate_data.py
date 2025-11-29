"""
Data Pipeline Validation Script

Tests all data sources and verifies data quality.
"""

import sys
import math
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import httpx
from aetheris_oracle.data.free_connectors import (
    FreeDataConnector,
    CCXTOHLCVConnector,
    DeribitIVConnector,
    DeribitPerpConnector,
    YFinanceMacroConnector,
)


def is_valid(value) -> bool:
    """Check if value is valid (not NaN and not None)."""
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return True


def test_deribit_api():
    """Test raw Deribit API connection."""
    print("\n" + "=" * 60)
    print("TEST 1: Raw Deribit API")
    print("=" * 60)

    # Test multiple endpoints to find working one
    endpoints = [
        ("get_volatility_index_data", {"currency": "BTC"}),
        ("get_index_price", {"index_name": "btc_usd"}),
        ("ticker", {"instrument_name": "BTC-PERPETUAL"}),
    ]

    client = httpx.Client(timeout=10)
    working = False

    for endpoint, params in endpoints:
        url = f"https://www.deribit.com/api/v2/public/{endpoint}"
        try:
            resp = client.get(url, params=params)
            print(f"  {endpoint}: HTTP {resp.status_code}")

            if resp.status_code == 200:
                data = resp.json()
                result = data.get("result", {})
                print(f"    Response keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
                working = True

                # Check for volatility data
                if "mark_iv" in str(result):
                    print(f"    Found IV: {result.get('mark_iv')}")
            else:
                print(f"    Error: {resp.text[:200]}")
        except Exception as e:
            print(f"    ERROR: {e}")

    if working:
        print("  PASS: At least one Deribit endpoint working")
        return True
    else:
        print("  FAIL: No Deribit endpoints working")
        return False


def test_deribit_connector():
    """Test DeribitIVConnector."""
    print("\n" + "=" * 60)
    print("TEST 2: DeribitIVConnector")
    print("=" * 60)

    connector = DeribitIVConnector()
    as_of = datetime.now()

    result = connector.fetch_iv_surface("BTC-USD", as_of)
    print(f"  Result: {result}")

    vals = list(result.values())
    if len(set(vals)) == 1 and vals[0] == 0.5:
        print("  FAIL: Got default 0.5 - Deribit fetch failed silently")
        return False
    else:
        print("  PASS: Got real IV data")
        return True


def test_ccxt_ohlcv():
    """Test CCXT OHLCV fetch."""
    print("\n" + "=" * 60)
    print("TEST 3: CCXT OHLCV (Binance)")
    print("=" * 60)

    connector = CCXTOHLCVConnector()
    end = datetime.now()
    start = end - timedelta(days=7)

    try:
        frame = connector.fetch("BTC-USD", start, end)
        print(f"  Timestamps: {len(frame.timestamps)}")
        print(f"  First close: ${frame.closes[0]:,.2f}")
        print(f"  Last close: ${frame.closes[-1]:,.2f}")

        # Check for mock data
        if all(c == 30000.0 for c in frame.closes):
            print("  FAIL: Got mock data ($30,000)")
            return False

        unique = len(set(frame.closes))
        if unique == 1:
            print("  WARN: All closes identical - suspicious")
            return False

        print(f"  Unique prices: {unique}")
        print("  PASS: Got real price data")
        return True
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        return False


def test_deribit_perp():
    """Test DeribitPerpConnector for funding/basis/order imbalance."""
    print("\n" + "=" * 60)
    print("TEST 4: DeribitPerpConnector (Funding, Basis, Order Imbalance)")
    print("=" * 60)

    connector = DeribitPerpConnector()
    as_of = datetime.now()

    try:
        result = connector.fetch_perp_data("BTC-USD", as_of)
        print(f"  Result: {result}")

        passed = True
        if "funding_rate" in result and is_valid(result["funding_rate"]):
            print(f"  Funding rate (annualized): {result['funding_rate']:.4%}")
        else:
            print("  WARN: No funding rate")
            passed = False

        if "basis" in result and is_valid(result["basis"]):
            print(f"  Basis: {result['basis']:.4%}")
        else:
            print("  WARN: No basis")
            passed = False

        if "order_imbalance" in result and is_valid(result["order_imbalance"]):
            print(f"  Order imbalance: {result['order_imbalance']:.4f}")
        else:
            print("  WARN: No order imbalance")
            passed = False

        if passed:
            print("  PASS: Got perp data")
        else:
            print("  PARTIAL: Some perp data missing")
        return passed

    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        return False


def test_yfinance_macro():
    """Test YFinance macro connector."""
    print("\n" + "=" * 60)
    print("TEST 5: YFinance Macro (VIX, DXY)")
    print("=" * 60)

    connector = YFinanceMacroConnector()
    as_of = datetime.now()

    try:
        result = connector.fetch_features(as_of)
        print(f"  Result: {result}")

        if not result:
            print("  WARN: No macro data returned")
            return False

        if "VIX" in result:
            print(f"  VIX: {result['VIX']:.2f}")
        if "DXY" in result:
            print(f"  DXY: {result['DXY']:.2f}")

        print("  PASS: Got macro data")
        return True
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        return False


def test_full_pipeline():
    """Test full FreeDataConnector pipeline."""
    print("\n" + "=" * 60)
    print("TEST 6: Full FreeDataConnector Pipeline")
    print("=" * 60)

    connector = FreeDataConnector(enable_cache=False)
    as_of = datetime.now()
    window = timedelta(days=30)

    try:
        frame = connector.fetch_window("BTC-USD", as_of=as_of, window=window)

        print(f"  Timestamps: {len(frame.timestamps)}")
        print(f"  Date range: {frame.timestamps[0]} to {frame.timestamps[-1]}")
        print(f"  Price range: ${min(frame.closes):,.2f} - ${max(frame.closes):,.2f}")

        # Check data quality
        issues = []
        successes = []

        # Price data check
        if all(c == 30000.0 for c in frame.closes):
            issues.append("MOCK PRICE DATA ($30,000)")
        else:
            successes.append(f"Price data: ${frame.closes[-1]:,.2f}")

        # IV data check
        iv_vals = [v for v in frame.iv_points.values() if is_valid(v)]
        if not iv_vals:
            issues.append("IV DATA: All NaN")
        elif len(set(iv_vals)) == 1 and iv_vals[0] == 0.5:
            issues.append("IV DATA: Default 0.5")
        else:
            successes.append(f"IV 30d: {frame.iv_points.get('iv_30d_atm', 'N/A'):.2%}" if is_valid(frame.iv_points.get('iv_30d_atm')) else "IV: partial")

        # Funding rate check
        if is_valid(frame.funding_rate):
            successes.append(f"Funding rate: {frame.funding_rate:.4%}")
        else:
            issues.append("FUNDING RATE: NaN/missing")

        # Basis check
        if is_valid(frame.basis):
            successes.append(f"Basis: {frame.basis:.4%}")
        else:
            issues.append("BASIS: NaN/missing")

        # Order imbalance check
        if is_valid(frame.order_imbalance):
            successes.append(f"Order imbalance: {frame.order_imbalance:.4f}")
        else:
            issues.append("ORDER IMBALANCE: NaN/missing")

        print("\n  Data Quality Check:")
        print(f"    IV points: {frame.iv_points}")
        print(f"    Funding rate: {frame.funding_rate}")
        print(f"    Basis: {frame.basis}")
        print(f"    Order imbalance: {frame.order_imbalance}")
        print(f"    Skew: {frame.skew}")

        if successes:
            print("\n  SUCCESSES:")
            for s in successes:
                print(f"    + {s}")

        if issues:
            print("\n  ISSUES:")
            for issue in issues:
                print(f"    - {issue}")

        # Check what's actually being used
        unique_closes = len(set(frame.closes))
        if unique_closes > 1:
            print("\n  PASS: Price data is real (not mock)")
            return True
        else:
            print("\n  FAIL: Suspicious data")
            return False

    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        return False


def test_data_usage_in_forecast():
    """Test that forecast engine receives correct data."""
    print("\n" + "=" * 60)
    print("TEST 7: Data Usage in Forecast Engine")
    print("=" * 60)

    try:
        from aetheris_oracle.pipeline.forecast import ForecastEngine
        from aetheris_oracle.config import ForecastConfig

        config = ForecastConfig(
            asset_id="BTC-USD",
            horizon_days=7,
            num_paths=100,  # Small for speed
        )

        engine = ForecastEngine(
            use_ncc_calibration=False,  # Skip SOTA for speed
            use_fm_gp_residuals=False,
            use_neural_jumps=False,
            use_diff_greeks=False,
            use_neural_rough_vol=False,
        )

        result = engine.forecast(config)

        print(f"  Forecast generated: {result is not None}")
        print(f"  Has quantile paths: {hasattr(result, 'quantile_paths')}")

        if result and hasattr(result, 'quantile_paths'):
            qp = result.quantile_paths
            print(f"  Keys: {list(qp.keys())}")

            # Check if keyed by day or by quantile
            if 0.5 in qp:
                # Keyed by quantile
                print(f"  P50 terminal: ${qp[0.5][-1]:,.2f}")
            elif 1 in qp or "1" in qp:
                # Keyed by day
                day_key = 1 if 1 in qp else "1"
                last_day = max(qp.keys())
                print(f"  Day {last_day} values: {qp[last_day]}")

            print("  PASS: Forecast engine working")
            return True
        else:
            print("  FAIL: No forecast result")
            return False

    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("  AETHERIS ORACLE - DATA VALIDATION")
    print("=" * 60)
    print(f"  Time: {datetime.now()}")

    results = {
        "Deribit API": test_deribit_api(),
        "Deribit IV Connector": test_deribit_connector(),
        "CCXT OHLCV": test_ccxt_ohlcv(),
        "Deribit Perp": test_deribit_perp(),
        "YFinance Macro": test_yfinance_macro(),
        "Full Pipeline": test_full_pipeline(),
        "Forecast Engine": test_data_usage_in_forecast(),
    }

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    passed = sum(results.values())
    total = len(results)
    print(f"\n  Total: {passed}/{total} tests passed")

    if passed < total:
        print("\n  RECOMMENDATIONS:")
        if not results["Deribit API"] or not results["Deribit IV Connector"]:
            print("  - IV data may be missing. Check Deribit API access.")
        if not results["Deribit Perp"]:
            print("  - Perp data (funding/basis) may be missing. Check Deribit API.")
        if not results["CCXT OHLCV"]:
            print("  - CCXT failed. Likely geo-blocked. Using yfinance fallback.")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
