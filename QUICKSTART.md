# Quick Start Guide

## Setup (One Time)

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env
```

## Option A: Just Run (Use Pretrained Models)

```bash
.venv\Scripts\python run.py
```

Done. Models are already trained in `artifacts/`.

---

## Option B: Train Fresh

```bash
# 1. Train
.venv\Scripts\python train.py

# 2. Run
.venv\Scripts\python run.py
```

---

## Option C: Full Optimization (Best Results)

```bash
# 1. Tune hyperparameters (auto-updates .env)
.venv\Scripts\python scripts/hyperparameter_tuning.py --component all --thorough

# 2. Train with tuned settings
.venv\Scripts\python train.py

# 3. Run
.venv\Scripts\python run.py
```

---

## Commands Reference

| Task | Command | Time |
|------|---------|------|
| **Run forecast** | `python run.py` | ~2 sec |
| **Run (legacy models)** | `python run.py --legacy` | ~1 sec |
| **Train (quick)** | `python train.py --quick` | ~5 min |
| **Train (full)** | `python train.py` | ~30 min |
| **Tune (quick)** | `python scripts/hyperparameter_tuning.py --component all --quick` | ~30 min |
| **Tune (thorough)** | `python scripts/hyperparameter_tuning.py --component all --thorough` | ~3-4 hrs |

### Tuning Modes

| Mode | Samples | Time Estimate |
|------|---------|---------------|
| `--validate` | 5 | ~1 min (just verifies it works) |
| Default (standard) | 80 | ~1-2 hours |
| `--quick` | 20 | ~15-20 min |
| `--thorough` | 150 | ~3-4 hours |

Tuned hyperparameters are saved to `.env` as `TUNING_*` variables and automatically used during training.

---

## Configuration (.env)

Edit `.env` to change settings:

```bash
# What to forecast
FORECAST_ASSET=BTC-USD
FORECAST_HORIZON=7          # 7, 30, or 90 days
FORECAST_PATHS=1000

# Model selection (all true = 5-SOTA, all false = legacy)
USE_NCC_CALIBRATION=true
USE_FM_GP_RESIDUALS=true
USE_NEURAL_JUMPS=true
USE_DIFF_GREEKS=true
USE_NEURAL_ROUGH_VOL=true
USE_MAMBA_TREND=false       # Keep false - has bias issues
```

---

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    FIRST TIME SETUP                         │
│  pip install → copy .env.example .env                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                 WANT BEST RESULTS?                          │
│                                                             │
│    YES ──────────────────────┐                              │
│                              ▼                              │
│              ┌───────────────────────────┐                  │
│              │  TUNE (--thorough)        │                  │
│              │  Auto-updates .env        │                  │
│              └───────────────────────────┘                  │
│                              │                              │
│    NO ───────────────────────┤                              │
│                              ▼                              │
│              ┌───────────────────────────┐                  │
│              │  TRAIN                    │                  │
│              │  Reads tuned params       │                  │
│              └───────────────────────────┘                  │
│                              │                              │
│                              ▼                              │
│              ┌───────────────────────────┐                  │
│              │  RUN                      │                  │
│              │  python run.py            │                  │
│              └───────────────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Output Example

```
======================================================================
  AETHERIS ORACLE - BTC-USD 7-Day Probabilistic Forecast
  5-SOTA Configuration
======================================================================

Forecast Cone:
----------------------------------------------------------------------
 Day          P05          P10          P50          P90          P95
----------------------------------------------------------------------
   1       91,234       92,456       95,123       97,890       98,765
   3       88,123       90,234       95,456       100,678      102,345
   7       82,345       86,789       96,123       105,456      109,876
----------------------------------------------------------------------

Terminal (Day 7) Summary:
  Median forecast:     $    96,123
  80% confidence:      $    86,789 - $   105,456
  P10-P90 spread:            19.4%
```

---

### Reset Trained Models

```bash
# Delete all trained model artifacts (forces retraining)
rm -rf artifacts/*.pt artifacts/*.json

# Windows PowerShell:
Remove-Item artifacts\*.pt, artifacts\*.json -ErrorAction SilentlyContinue

# Then retrain
.venv\Scripts\python train.py
```

### Reset Everything (Full Clean Slate)

```bash
# Windows PowerShell - Delete all generated files
Remove-Item artifacts\*.pt, artifacts\*.json -ErrorAction SilentlyContinue
Remove-Item artifacts\tuning\* -ErrorAction SilentlyContinue
Remove-Item outputs\* -ErrorAction SilentlyContinue

# Bash/Linux
rm -rf artifacts/*.pt artifacts/*.json artifacts/tuning/* outputs/*

# Then retrain from scratch
.venv\Scripts\python train.py
```

### Cache Behavior Summary

| Cache Type | Location | Lifespan | How to Clear |
|------------|----------|----------|--------------|
| Data cache | In-memory | 1 hour TTL, or until restart | Restart script or `enable_cache=False` |
| Model artifacts | `artifacts/*.pt` | Permanent until deleted | Delete files + retrain |
| Tuning results | `artifacts/tuning/` | Permanent | Delete files |
| Forecast outputs | `outputs/` | Permanent | Delete files |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `No historical data` | Run `python scripts/collect_historical_data.py --asset BTC-USD` |
| `Model not found` | Run `python train.py` |
| `CUDA not available` | Set `TORCH_DEVICE=cpu` in `.env` |
| `Slow forecast` | Reduce `FORECAST_PATHS` in `.env` |
| `Stale data` | Restart script (clears in-memory cache) |
| `Want fresh models` | Delete `artifacts/*.pt` and retrain |
