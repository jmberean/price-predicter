# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Aetheris Oracle v10.0** - A probabilistic N-day price forecasting engine for crypto assets (BTC-USD, ETH-USD). Produces forecast cones (quantile paths P5-P95) over 1-90 day horizons, designed to be robust across market regimes.

The system supports two model families:
- **Legacy**: Heuristic models (fast, production-ready, but overconfident ~5% spread)
- **5-SOTA (Recommended)**: 5-component neural stack trained on 5 years of historical data
  - Realistic ~33% spread for 7-day forecasts
  - Supports 7, 30, and 90-day horizons
  - Excludes MambaTS trend due to directional bias issues
  - Trained on 1,776 days of BTC-USD data (2021-2025)

For detailed documentation:
- [README.md](README.md) - Quick start, TODO checklist, known issues
- [docs/implementation/PROJECT_STATUS.md](docs/implementation/PROJECT_STATUS.md) - Implementation status
- [docs/design/plan.md](docs/design/plan.md) - System design specification

## Quick Start (Simplified)

All config is in `.env` - copy from `.env.example` and edit as needed.

### 1. Setup
```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env   # Windows
cp .env.example .env     # Linux/Mac
```

### 2. Train (optional - pretrained models exist)
```bash
.venv\Scripts\python train.py           # Full training
.venv\Scripts\python train.py --quick   # Quick training
```

### 3. Run
```bash
.venv\Scripts\python run.py             # Uses .env config
.venv\Scripts\python run.py --legacy    # Use legacy models
```

### Key .env Settings
```bash
# Forecast config
FORECAST_ASSET=BTC-USD      # Asset to forecast
FORECAST_HORIZON=7          # Days ahead (7, 30, 90)
FORECAST_PATHS=1000         # Monte Carlo paths

# Training config
TRAINING_HORIZON=90         # Horizon to train for

# Model selection (true = SOTA, false = legacy)
USE_NCC_CALIBRATION=true
USE_FM_GP_RESIDUALS=true
USE_NEURAL_JUMPS=true
USE_DIFF_GREEKS=true
USE_NEURAL_ROUGH_VOL=true
USE_MAMBA_TREND=false       # Keep false - has bias issues
```

## Development Commands

### Testing
```bash
.venv\Scripts\python -m pytest -q                                    # All tests
.venv\Scripts\python -m pytest tests/test_pipeline.py -v             # Single file
.venv\Scripts\python -m pytest tests/test_pipeline.py::test_name -v  # Single test
```

### Advanced CLI
```bash
# Set PYTHONPATH first
$env:PYTHONPATH="src"  # PowerShell
export PYTHONPATH=src  # Bash

# CLI with options
.venv\Scripts\python -m aetheris_oracle.cli --asset BTC-USD --horizon 7 --paths 500 --plot

# Alternate entry point
.venv\Scripts\python start.py --mode forecast --asset BTC-USD --horizon 30
```

### Training (Advanced)
```bash
# Retrain with historical parquet data
.venv\Scripts\python scripts/retrain_with_historical.py --skip-mamba

# Train single component
.venv\Scripts\python -m aetheris_oracle.pipeline.train_sota --component ncc --epochs 30
```

### Service
```bash
.venv\Scripts\python -m aetheris_oracle.server
```

### Validation
```bash
.venv\Scripts\python scripts/run_validation.py --recent
```

## Architecture

### Core Data Flow

```
DataConnector.fetch_window() → MarketFeatureFrame
       ↓
StationarityNormalizer.normalize_and_stats() → normalized closes + stats
       ↓
compute_regime_vector() → RegimeVector (conditioning for all models)
       ↓
┌─────────────────────────────────────────────────────────────┐
│ Model Pipeline (legacy or SOTA based on feature flags)      │
├─────────────────────────────────────────────────────────────┤
│ TrendEnsemble / MambaTrendWrapper → TrendPath               │
│ MarketMakerEngine / DifferentiableMMEngine → MM indices     │
│ VolPathEngine / NeuralRoughVolWrapper → VolPath             │
│ ResidualGenerator / FMGPResidualEngine → ResidualPaths      │
│ JumpModel / NeuralJumpSDEEngine → JumpPath                  │
└─────────────────────────────────────────────────────────────┘
       ↓
Path Assembly: trend + jumps + residuals → price paths (denormalized)
       ↓
CalibrationEngine / NCCCalibrationEngine → adjusted quantiles
       ↓
ForecastResult: quantile_paths, threshold_probs, drivers, metadata
```

### Key Source Locations

| Component | Legacy | SOTA |
|-----------|--------|------|
| Trend | `modules/trend.py` | `modules/mamba_trend.py` (not recommended) |
| Volatility | `modules/vol_path.py` | `modules/neural_rough_vol.py` |
| Jumps | `modules/jump.py` | `modules/neural_jump_sde.py` |
| Residuals | `modules/residual.py` | `modules/fm_gp_residual.py` |
| MM State | `modules/market_maker.py` | `modules/differentiable_greeks.py` |
| Calibration | `pipeline/calibration.py` | `pipeline/neural_conformal_control.py` |
| Orchestration | `pipeline/forecast.py` (ForecastEngine) |
| Config | `config.py` (ForecastConfig, ScenarioOverrides) |
| Schemas | `data/schemas.py` (MarketFeatureFrame, RegimeVector) |

### Data Connectors

- `SyntheticDataConnector` - Deterministic data for tests
- `FreeDataConnector` - ccxt + Deribit + yfinance (no API keys)
- `CsvDataConnector` - Local CSV files
- `CCXTPerpConnector` - Spot + perp with caching

Select via `--connector` CLI flag or `ForecastEngine(connector=...)`.

## Design Constraints

- **Zero-mean residuals**: Residual paths must have zero mean over horizon to avoid double-counting trend
- **Past-only normalization**: No look-ahead bias in `StationarityNormalizer`
- **Regime conditioning**: All models receive `RegimeVector` as input
- **Scenario labeling**: Scenario forecasts must be clearly labeled as conditional (not used for calibration)
- **Reproducibility**: All random seeds should be configurable

## Resolved Issues (2025-11-26)

The **Full SOTA Component Interaction Bug** has been resolved. Root causes:
1. **NCC training mismatch** - Fixed in `training_data_prep.py` (NCC now trained on SOTA forecasts)
2. **yfinance data bug** - Fixed in `free_connectors.py` (use `start/end` not `period`)
3. **Mamba cumulative sum** - Fixed in `mamba_trend.py` (proper return accumulation)
4. **Mamba directional bias** - UNFIXABLE fundamental limitation, removed from recommended config

**Recommendation**: Use 5-component SOTA (without Mamba) for best results.

## Feature Flags

### Recommended: 5-Component SOTA
```python
engine = ForecastEngine(
    use_ncc_calibration=True,
    use_diff_greeks=True,
    use_fm_gp_residuals=True,
    use_neural_rough_vol=True,
    use_neural_jumps=True,
    use_mamba_trend=False,  # Exclude - has directional bias issues
    device="cpu",  # or "cuda"
)
```

### Legacy (fast but overconfident)
```python
engine = ForecastEngine()  # All SOTA flags default to False
```

Models are loaded from `artifacts/` directory.
