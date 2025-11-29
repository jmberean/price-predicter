# Data Pipeline Audit Report

**Date:** 2025-11-28
**Scope:** Data fetching, storage, format, and usage validation

---

## Executive Summary

| Area | Status | Issues |
|------|--------|--------|
| **Data Fetching** | Partial | Silent mock fallback, no validation |
| **Data Schema** | Good | Clean, but missing validation |
| **Data Storage** | Good | Parquet + CSV, with metadata |
| **Caching** | Good | LRU + TTL, proper implementation |
| **Data Flow** | Moderate | Inconsistent preprocessing |

---

## 0. Live Validation Results

**Tested:** 2025-11-28
**Status:** ALL TESTS PASSING (7/7)

### What's Working

| Data Source | Status | Value |
|-------------|--------|-------|
| **Price Data (yfinance)** | PASS | $84,648 - $110,639 range |
| **IV Data (Deribit DVOL)** | PASS | 47.9% (30-day ATM) |
| **Funding Rate (Deribit)** | PASS | 0.0000% (annualized) |
| **Basis (Deribit)** | PASS | 0.0043% |
| **Order Imbalance (Deribit)** | PASS | -0.9425 |
| **Macro Data (VIX/DXY)** | PASS | VIX: 16.35, DXY: 99.46 |
| **Forecast Engine** | PASS | Generates quantile paths |

### Fixed Issues (2025-11-28)

1. **DeribitIVConnector**: Now uses `get_volatility_index_data` with proper `start_timestamp`/`end_timestamp` params
2. **DeribitPerpConnector**: New connector fetches funding rate, basis, order imbalance from `ticker` endpoint
3. **CompositeConnector**: Updated to merge perp data into MarketFeatureFrame
4. **Default values**: Changed from 0.0 to NaN to distinguish missing from zero

### Test Command

```bash
PYTHONPATH=src python scripts/validate_data.py
```

---

## 1. Data Fetching Issues

### 1.1 Silent Mock Data Fallback (CRITICAL)

**Location:** `free_connectors.py:54-58`
```python
# Last resort: mock data
if not candles:
    closes = [30_000.0 for _ in range(limit)]  # FAKE DATA!
    volumes = [1_000.0 for _ in range(limit)]
    timestamps = [start + timedelta(days=i) for i in range(limit)]
```

**Problem:** When both ccxt and yfinance fail, the connector returns **fake static data** ($30,000 BTC) without any warning. User gets a forecast based on fake data.

**Impact:** Completely invalid forecasts with no indication of failure.

**Fix:**
```python
if not candles:
    raise DataFetchError(f"Failed to fetch {asset_id} from all sources")
    # OR
    logger.warning(f"USING MOCK DATA for {asset_id} - forecasts unreliable!")
    # Set a flag on MarketFeatureFrame
```

---

### 1.2 IV Data is Single Point, Not Surface (MODERATE)

**Location:** `free_connectors.py:92-105`
```python
def fetch_iv_surface(self, asset_id: str, as_of: datetime) -> Dict[str, float]:
    # ...
    vol = float(data.get("current_volatility", 0.5))
    return {
        "iv_7d_atm": vol,   # Same value!
        "iv_14d_atm": vol,  # Same value!
        "iv_30d_atm": vol,  # Same value!
    }
```

**Problem:** All tenor IVs are the same value. There's no actual term structure.

**Impact:** Term structure slope is always 0. Models can't learn IV dynamics.

**Fix:** Fetch actual term structure from Deribit options chain API.

---

### 1.3 Default Values Mask Missing Data (MODERATE)

**Location:** `free_connectors.py:64-73`
```python
return MarketFeatureFrame(
    iv_points={"iv_7d_atm": 0.5, "iv_14d_atm": 0.5, "iv_30d_atm": 0.5},  # Default
    funding_rate=0.0,  # Default
    basis=0.0,  # Default
    order_imbalance=0.0,  # Default
    skew=0.0,  # Default
)
```

**Problem:** Missing data is filled with zeros/defaults, indistinguishable from actual zero values.

**Impact:** Model can't differentiate "missing" from "actually zero."

**Fix:** Use `Optional[float]` or `float('nan')` for missing values.

---

### 1.4 No Timestamp Validation

**Location:** Multiple connectors

**Problem:** No validation that:
- Timestamps are in order
- No gaps in daily data
- Timestamps match expected timezone
- Data isn't from the future

**Fix:**
```python
def validate_frame(frame: MarketFeatureFrame) -> None:
    assert len(frame.timestamps) == len(frame.closes)
    assert all(frame.timestamps[i] < frame.timestamps[i+1] for i in range(len(frame.timestamps)-1))
    assert frame.timestamps[-1] <= datetime.now()
```

---

## 2. Data Schema Issues

### 2.1 No Field Validation in MarketFeatureFrame

**Location:** `schemas.py:6-18`
```python
@dataclass
class MarketFeatureFrame:
    timestamps: List[datetime]
    closes: List[float]
    volumes: List[float]
    # ... no validation
```

**Problem:** Dataclass accepts any values. No constraints on:
- Positive prices
- Non-empty lists
- Matching lengths
- Valid ranges

**Fix:** Add validation with `__post_init__` or use Pydantic:
```python
def __post_init__(self):
    assert len(self.timestamps) == len(self.closes) == len(self.volumes)
    assert all(c > 0 for c in self.closes)
    assert all(v >= 0 for v in self.volumes)
```

---

### 2.2 IV Points Dict Has Inconsistent Keys

**Observed Usage:**
```python
# Some places use:
iv_points["iv_7d_atm"]

# Others use:
iv_points.get("iv_7d_atm", 0.5)

# Historical connector uses different format:
iv_points["7"] = 0.5  # Maybe?
```

**Problem:** No schema for IV dict keys. Different connectors may use different formats.

**Fix:** Define constants:
```python
IV_KEYS = ["iv_7d_atm", "iv_14d_atm", "iv_30d_atm"]
```

---

## 3. Data Storage Issues

### 3.1 Historical Data Format is Good

**Location:** `collect_historical_data.py`

The collection script properly:
- Uses parquet (efficient, typed)
- Saves CSV backup (human-readable)
- Stores metadata JSON
- Computes derived features

**No issues found.**

---

### 3.2 Derived Features Use Future Data (CRITICAL)

**Location:** `collect_historical_data.py:226-227`
```python
df["realized_vol_7d"] = df["returns"].rolling(7).std() * np.sqrt(365)
df["realized_vol_30d"] = df["returns"].rolling(30).std() * np.sqrt(365)
```

**Problem:** Rolling windows look backward from each row, which is correct. BUT...

**Location:** `historical_connector.py:112-113`
```python
iv_7d = float(latest.get("realized_vol_7d", 0.5))  # Using row as_of, not prior
```

When using historical connector, "latest" row IS the `as_of` date. If `as_of` is day T:
- `realized_vol_7d` uses returns from T-6 to T
- This is correct ONLY if we don't use day T's return for prediction

**Potential Leakage:** If day T's return is in the training target, and also in the vol feature, there's leakage.

**Fix:** Use `shift(1)` on derived features or explicitly use T-1 row:
```python
# Use prior day's volatility, not current day's
iv_7d = float(window_df.iloc[-2].get("realized_vol_7d", 0.5))
```

---

### 3.3 Proxy Features Are Weak (MODERATE)

**Location:** `historical_connector.py:120-134`
```python
# Funding rate - approximate from momentum
funding_rate = momentum * 0.001  # Arbitrary scaling

# Basis - approximate from price deviation
basis = price_to_sma * closes[-1] * 0.01  # Arbitrary

# Order imbalance - approximate from volume ratio
order_imbalance = (volume_ratio - 1.0) * 0.5  # Arbitrary
```

**Problem:** These are weak proxies. Momentum != funding rate. Volume ratio != order imbalance.

**Impact:** Models trained on proxies won't generalize to real features.

**Fix:** Either:
1. Collect actual funding/basis data from exchanges
2. Clearly label as proxies and don't expect same performance

---

## 4. Caching Issues

### 4.1 Cache Implementation is Good

**Location:** `cache.py`

The cache correctly implements:
- LRU eviction
- TTL expiration
- Thread-safe (basic)
- Statistics tracking

**No issues found.**

---

### 4.2 Cache Key Missing Window Parameter

**Location:** `free_connectors.py:179`
```python
cache_key = make_cache_key(asset_id, start, as_of)
```

**Location:** `cache.py:111-123`
```python
def make_cache_key(asset_id: str, start: datetime, end: datetime) -> str:
    return f"{asset_id}:{start.isoformat()}:{end.isoformat()}"
```

**Problem:** Cache key doesn't include `window` parameter. If you request:
- Request 1: `(BTC-USD, as_of=Dec1, window=30d)` → key includes start=Nov1
- Request 2: `(BTC-USD, as_of=Dec1, window=60d)` → different start, different key

Actually OK because `start = as_of - window`, so different windows produce different keys.

**Status:** No issue found after analysis.

---

## 5. Data Flow Issues

### 5.1 Normalization Applied Inconsistently

**Location:** `stationarity.py:34-36`
```python
def normalize_and_stats(self, series: List[float]) -> Tuple[List[float], NormalizationStats]:
    stats = self.fit(series)  # Uses FULL series for mean/std
    return self.normalize(series, stats), stats
```

**Problem:** Mean/std computed on full window. If window includes future data (relative to forecast date), there's look-ahead bias.

**Safe:** If series is strictly historical (before forecast date)
**Unsafe:** If series includes data up to forecast date

**Usage check needed** in each calling context.

---

### 5.2 Timezone Handling is Inconsistent

**Locations:**
```python
# historical_connector.py:62-63
if self._df.index.tz is not None:
    self._df.index = self._df.index.tz_localize(None)  # Remove TZ

# historical_connector.py:89-90
if as_of.tzinfo is not None:
    as_of = as_of.replace(tzinfo=None)  # Remove TZ
```

**Problem:** Timezones are stripped. If data source is in UTC and user passes local time, dates may mismatch.

**Fix:** Standardize on UTC everywhere:
```python
# Always convert to UTC
as_of = as_of.astimezone(timezone.utc).replace(tzinfo=None)
```

---

### 5.3 Length Mismatches Not Checked

**Example:**
```python
# training_data_prep.py:306
normalized, stats = normalizer.normalize_and_stats(closes[-horizon-1:])
if len(normalized) < horizon + 1:
    continue
```

Many places have manual length checks. Should be centralized.

**Fix:** Add validation in `MarketFeatureFrame`:
```python
@property
def length(self) -> int:
    lengths = [len(self.timestamps), len(self.closes), len(self.volumes)]
    if len(set(lengths)) != 1:
        raise ValueError(f"Mismatched lengths: {lengths}")
    return lengths[0]
```

---

## 6. Summary of Issues by Severity

### Critical (Fix Before Production)

| # | Issue | Location |
|---|-------|----------|
| 1 | Silent mock data fallback | `free_connectors.py:54-58` |
| 2 | Potential look-ahead in derived features | `historical_connector.py:112` |

### High (Fix Soon)

| # | Issue | Location |
|---|-------|----------|
| 3 | No validation on MarketFeatureFrame | `schemas.py` |
| 4 | Default values mask missing data | `free_connectors.py:64-73` |
| 5 | IV data is single point, not surface | `free_connectors.py:92-105` |

### Moderate (Improve)

| # | Issue | Location |
|---|-------|----------|
| 6 | Proxy features are weak | `historical_connector.py:120-134` |
| 7 | No timestamp validation | Multiple |
| 8 | Timezone handling inconsistent | Multiple |

---

## 7. Recommended Fixes

### Immediate

```python
# 1. Add mock data warning (free_connectors.py)
if not candles:
    logger.error(f"MOCK DATA USED for {asset_id} - all data sources failed!")
    # Consider raising exception instead

# 2. Add MarketFeatureFrame validation (schemas.py)
def __post_init__(self):
    if len(self.timestamps) != len(self.closes):
        raise ValueError("timestamps and closes length mismatch")
    if any(c <= 0 for c in self.closes):
        raise ValueError("closes must be positive")
```

### Short-term

```python
# 3. Fix look-ahead in historical connector
# Use prior day's features, not current day
prior_row = window_df.iloc[-2] if len(window_df) > 1 else window_df.iloc[-1]
iv_7d = float(prior_row.get("realized_vol_7d", 0.5))

# 4. Use NaN for missing data
iv_points={"iv_7d_atm": float('nan') if missing else vol}
```

---

## 8. Data Quality Checklist

Before training, verify:

- [ ] No mock data in training set
- [ ] All timestamps in correct order
- [ ] No future data leakage
- [ ] Sufficient data coverage (no large gaps)
- [ ] IV term structure has actual spread
- [ ] Features are from T-1, not T
- [ ] Normalization stats from historical data only

Run validation:
```bash
PYTHONPATH=src python scripts/validate_data.py
```

**Results (2025-11-28): 5/6 tests passed**
- Price data: PASS (real market data)
- Macro data: PASS (VIX, DXY)
- IV data: FAIL (Deribit API issue)
- Forecast engine: PASS
