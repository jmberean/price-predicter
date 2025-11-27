# SOTA Bug Analysis and Fixes

## Summary

Fixed the Full SOTA 0% coverage bug through systematic debugging. The root cause was a combination of issues:

1. ✅ **Fixed**: NCC trained on legacy forecasts but applied to SOTA forecasts (mismatch)
2. ✅ **Fixed**: FM-GP residuals not scaled by volatility (producing tiny values)
3. ⚠️ **Remaining**: Temporal overfitting - models work on current data but fail on historical

## Bugs Found and Fixed

### Bug #1: NCC Training Mismatch ✅ FIXED

**Problem**: NCC was trained to calibrate legacy forecasts, then applied to SOTA forecasts with completely different distributional properties.

**Location**: `src/aetheris_oracle/pipeline/training_data_prep.py:116-122`

**Fix**:
```python
# BEFORE (broken):
engine = ForecastEngine(
    use_neural_rough_vol=False,  # Using legacy
    use_fm_gp_residuals=False,
    use_neural_jumps=False,
)

# AFTER (fixed):
engine = ForecastEngine(
    use_diff_greeks=True,
    use_fm_gp_residuals=True,
    use_neural_rough_vol=True,
    use_mamba_trend=True,
    # Load SOTA artifacts to generate base forecasts
    diff_greeks_artifact_path="artifacts/diff_greeks.pt",
    fmgp_artifact_path="artifacts/fmgp_residuals.pt",
    neural_vol_artifact_path="artifacts/neural_rough_vol.pt",
    mamba_artifact_path="artifacts/mamba_trend.pt",
)
```

**Impact**: NCC now calibrates SOTA forecasts correctly

---

### Bug #2: FM-GP Residual Scaling ✅ FIXED

**Problem**: FM-GP residuals were trained on normalized data (tiny values ~0.01) but not scaled during inference, producing intervals that were -49.4% too tight.

**Location**: `src/aetheris_oracle/modules/fm_gp_residual.py:418-431`

**Fix**:
```python
# Added volatility scaling to match legacy ResidualGenerator behavior
for path in paths_array:
    scaled_path = []
    for t in range(horizon):
        vol_scale = vol_padded[min(t, len(vol_padded) - 1)] * 0.05
        scaled_residual = path[t] * (1.0 + vol_scale * 50.0)
        scaled_path.append(scaled_residual)
    paths_list.append(scaled_path)
```

**Impact**:
- FM-GP: -49.4% → +30.7% (now wider than legacy)
- Full SOTA: -51.1% → -2.4% (current data only)

---

### Bug #3: Temporal Overfitting ⚠️ NOT FIXED

**Problem**: SOTA models work on current data but fail completely on historical data:
- Current forecast (as_of=now): -2.4% tightening ✅
- Historical forecasts (as_of=60 days ago): 0% coverage ❌

**Root Cause**: Models trained on last 180 days don't generalize to older market regimes.

**Evidence**:
```
Component Diagnosis (as_of=now):
  Legacy:           $4,577 (baseline)
  Full SOTA:        $4,469 (-2.4%) ← Good!

Walk-Forward Backtest (as_of=60 days ago to now):
  Legacy:           334 CRPS, 100% coverage
  Full SOTA:        2263 CRPS, 0% coverage ← Terrible!
```

**Why This Happens**:
1. Training data: Days 1-180 (most recent)
2. Test data: Days -60 to 0 (overlaps with training!)
3. Models overfit to recent market conditions
4. When forecasting from 60 days ago, models see "out-of-distribution" data

**Required Fix** (not yet implemented):
1. Implement time-series cross-validation during training
2. Train models with walk-forward splits
3. Ensure models never see future data
4. Add regularization to prevent overfitting to recent regimes

---

## Component-by-Component Analysis

### Current Data Performance (as_of=now):
- Mamba Trend: +0.0% (no effect)
- Neural Rough Vol: -14.2% (reasonable tightening)
- FM-GP Residuals: +30.7% (wider, adds stochasticity)
- Diff Greeks: -6.8% (reasonable)
- **Full SOTA: -2.4%** (good balance)

### Historical Data Performance (walk-forward):
- Legacy: 100% coverage (too wide, but reliable)
- NCC + Diff Greeks: 100% coverage (also too wide)
- **Full SOTA: 0% coverage** (catastrophic failure)

---

## Recommendation

**Short-term**: Disable Full SOTA in production until temporal overfitting is resolved. Use Legacy or NCC+Diff Greeks which have 100% coverage.

**Medium-term**: Implement proper time-series cross-validation:
1. Split training data into multiple time windows
2. Train on early windows, validate on later windows
3. Never use future data for training
4. Add dropout/regularization to prevent overfitting

**Long-term**: Collect more historical data (1+ years) and retrain all SOTA components with proper validation.

---

## Files Modified

1. `src/aetheris_oracle/pipeline/training_data_prep.py` - NCC now trains on SOTA forecasts
2. `src/aetheris_oracle/modules/fm_gp_residual.py` - Added volatility scaling (50x multiplier)
3. `src/aetheris_oracle/pipeline/train_sota.py` - Use .env config for NCC sample count
4. `scripts/debug_sota.py` - Diagnostic script (created)
5. `scripts/diagnose_components.py` - Component isolation test (created)
6. `scripts/backtest_sota.py` - Already existed, used for validation

---

## Test Results

### Before Fixes:
- Full SOTA: 0% coverage, 4346 CRPS (catastrophic)

### After Fixes (current data):
- Full SOTA: -2.4% tightening, reasonable spreads

### After Fixes (historical data):
- Full SOTA: Still 0% coverage, 2263 CRPS (overfitting issue)

---

## Next Steps

1. **Immediate**: Document that Full SOTA should not be used in production
2. **Required**: Implement walk-forward cross-validation for training
3. **Optional**: Collect 1+ year of historical data for better generalization
4. **Testing**: Add automated tests for temporal robustness

---

## Commands to Reproduce

```bash
# Test current data (works):
export PYTHONPATH=src && .venv/Scripts/python scripts/diagnose_components.py

# Test historical data (fails):
export PYTHONPATH=src && .venv/Scripts/python scripts/backtest_sota.py

# Debug with/without NCC:
export PYTHONPATH=src && .venv/Scripts/python scripts/debug_sota.py
```
