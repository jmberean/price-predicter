# Aetheris Oracle - Full SOTA Component Interaction Bug Investigation

## Problem Statement

The "Full SOTA" configuration (all 6 neural components combined) has **0% coverage** on historical backtests, despite each component working perfectly when tested individually. This appears to be a component interaction effect causing forecast intervals to become catastrophically too tight.

## Background

Aetheris Oracle is a crypto price forecasting system with:
- **Legacy models**: Heuristic-based (work fine, 100% coverage)
- **SOTA models**: 6 neural components that can be enabled individually or combined

### SOTA Components:
1. **NCC (Neural Conformal Control)** - Adaptive calibration
2. **Differentiable Greeks** - Market maker state representation
3. **FM-GP Residuals** - Flow matching with GP priors for stochastic residuals
4. **Neural Rough Volatility** - Fractional Brownian motion (H≈0.1)
5. **Neural Jump SDE** - Learned jump-diffusion process
6. **MambaTS Trend** - State-space trend model

## What We've Done So Far

### Implemented:
1. **Holdout period (60 days)** - Prevents temporal overfitting
   - Location: `src/aetheris_oracle/pipeline/training_data_prep.py`
   - Config: `.env` file has `TRAINING_HOLDOUT_DAYS=60`
   - Verified working: Training period shows "2025-04-01 to 2025-09-28 (excluding last 60 days)"

2. **Retrained all models** with holdout period
   - All 6 components trained successfully
   - Artifacts saved to `artifacts/*.pt`

3. **Added Neural Jump SDE batch sampling**
   - Implemented `sample_sde_paths()` in `src/aetheris_oracle/modules/neural_jump_sde.py:416-482`

4. **Created walk-forward cross-validation infrastructure**
   - File: `src/aetheris_oracle/pipeline/walk_forward_cv.py`

### Issue Persists:
Full SOTA still has 0% coverage despite fixes.

## Key Findings

### Diagnostic Results (scripts/diagnose_component_isolation.py):

```
Component                           CRPS       Coverage     Status
----------------------------------------------------------------------
Baseline (NCC + Diff Greeks)        380        100%         PASS
+ FM-GP Residuals (alone)           129        100%         PASS
+ Neural Rough Vol (alone)          356        100%         PASS
+ Neural Jump SDE (alone)           387        100%         PASS
+ MambaTS Trend (alone)             721        100%         PASS
Full SOTA (ALL 6 combined)          1387       0%           FAIL
```

**Critical Finding**: Each component works fine individually (100% coverage), but when ALL are combined, coverage drops to 0% and CRPS degrades from ~380 to 1387.

## Reproduction Steps

### Setup:
```bash
cd C:\PC\Workspace\price-predicter
.venv\Scripts\activate
$env:PYTHONPATH="src"
```

### Run Diagnostic:
```bash
.venv\Scripts\python scripts/diagnose_component_isolation.py
```

### Run Full Backtest:
```bash
.venv\Scripts\python scripts/backtest_sota.py
```

Expected: Full SOTA shows 0% coverage, 1300+ CRPS

## Hypotheses for Root Cause

1. **Over-stacking tightening effects**: Multiple components each narrow intervals slightly, compounding to extreme over-confidence

2. **NCC calibration mismatch**: NCC was trained on SOTA forecasts, but may not properly account for the interaction of ALL components simultaneously

3. **Component interference**: Models assume independence but interfere when combined (e.g., Mamba Trend + FM-GP Residuals may conflict)

4. **Scaling/normalization interaction**: Components may apply conflicting normalizations that compound incorrectly

## Investigation Questions

1. **Which pair of components causes the issue?**
   - Test all pairwise combinations to identify problematic interactions
   - Modify `scripts/diagnose_component_isolation.py` to test: NCC+Diff+FMGP+Vol, NCC+Diff+FMGP+Jump, etc.

2. **How does NCC behave with Full SOTA?**
   - Retrain NCC specifically on Full SOTA forecasts (all 6 components as base)
   - Current: NCC trained on partial SOTA (line 124-137 in `training_data_prep.py`)
   - Fix: Enable ALL components when generating NCC training data

3. **Are forecast intervals actually too tight?**
   - Inspect actual forecast quantiles vs realized prices
   - Check if P10-P90 interval width degrades as components are added

4. **Is there a numerical instability?**
   - Check for NaN/Inf values in combined forecasts
   - Verify all components output reasonable ranges

## Key File Locations

### Config:
- `.env` - Training configuration (holdout period, sample counts)
- `src/aetheris_oracle/config.py` - ForecastConfig

### Training:
- `src/aetheris_oracle/pipeline/training_data_prep.py` - Data preparation with holdout
- `src/aetheris_oracle/pipeline/train_sota.py` - SOTA model training CLI

### Forecasting:
- `src/aetheris_oracle/pipeline/forecast.py:ForecastEngine` - Main forecast orchestration
  - Lines 614-650: Neural Jump SDE integration
  - Check how components combine in `generate_forecast()`

### Components:
- `src/aetheris_oracle/pipeline/neural_conformal_control.py` - NCC calibration
- `src/aetheris_oracle/modules/fm_gp_residual.py` - FM-GP (line 418-431 has volatility scaling)
- `src/aetheris_oracle/modules/neural_jump_sde.py` - Neural Jump SDE
- `src/aetheris_oracle/modules/neural_rough_vol.py` - Neural Rough Vol
- `src/aetheris_oracle/modules/mamba_trend.py` - MambaTS Trend
- `src/aetheris_oracle/modules/differentiable_greeks.py` - Diff Greeks

### Backtesting:
- `scripts/backtest_sota.py` - Walk-forward validation (18 forecasts)
- `scripts/diagnose_component_isolation.py` - Component isolation test (6 forecasts)
- `src/aetheris_oracle/pipeline/offline_evaluation.py` - Backtest infrastructure

## Suggested Next Steps

1. **Retrain NCC on Full SOTA base**:
   - Modify `prepare_ncc_training_data()` to enable all 6 components
   - Retrain NCC to learn proper calibration for combined system
   - Test if this fixes coverage

2. **Test pairwise combinations**:
   - Create systematic test of all 2-component, 3-component, 4-component combos
   - Identify minimum combination that breaks (e.g., "FMGP + Neural Vol = broken")

3. **Inspect forecast intervals**:
   - Add debug logging to print P10-P90 widths for each configuration
   - Compare interval widths: Baseline vs +Component1 vs +Component2 vs Full

4. **Check for numerical issues**:
   - Add assertions/logging for NaN/Inf in forecast paths
   - Verify residuals, jumps, vol paths are all in reasonable ranges

5. **Review component combination logic**:
   - In `forecast.py`, verify how trend + residuals + jumps combine
   - Check if there's double-counting or cancellation

## Environment

- **OS**: Windows (PowerShell)
- **Python**: .venv environment
- **Device**: CPU (no GPU required)
- **Data**: BTC-USD via FreeDataConnector (ccxt + Deribit + yfinance)

## Expected Outcome

After fixing, Full SOTA should achieve:
- **P10-P90 Coverage**: 75-85% (target: 80%)
- **CRPS**: < 400 (lower than Legacy's ~357)
- **Status**: PASS on backtest

## Related Documentation

- `README.md` - Project overview and setup
- `CLAUDE.md` - Development guide
- `docs/SOTA_BUG_ANALYSIS.md` - Documents earlier bugs (NCC mismatch, FM-GP scaling)

---

**TL;DR**: Full SOTA has 0% coverage due to component interaction effect (not individual component failure). Each of 6 neural components works fine alone (100% coverage), but when all combined, forecast intervals become catastrophically too tight. Need to identify which component pairs/combos cause the interaction and fix the combination logic or retrain NCC on Full SOTA forecasts.
