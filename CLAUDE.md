# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Documentation Structure

For complete project information, see:
- **[README.md](README.md)** - Project overview, quick start, TODO checklist
- **[docs/implementation/PROJECT_STATUS.md](docs/implementation/PROJECT_STATUS.md)** - Detailed implementation status
- **[docs/design/plan.md](docs/design/plan.md)** - Original system design specification
- **[docs/design/aetheris_oracle_critique_advanced.md](docs/design/aetheris_oracle_critique_advanced.md)** - SOTA critique and recommendations
- **[docs/testing/TESTING.md](docs/testing/TESTING.md)** - Testing guide and benchmarks

## Project Overview

**Aetheris Oracle v10.0 - State-of-the-Art Edition** - A probabilistic N-day price forecasting engine for crypto assets (primarily BTC-USD, ETH-USD). The system produces forecast cones (quantile paths) rather than point predictions, designed to be robust across different market regimes.

**NEW**: Now includes optional state-of-the-art (SOTA) neural components based on 2023-2024 research:
- Neural Conformal Control for adaptive calibration
- Flow Matching with Gaussian Process priors for residuals
- Neural Jump SDEs for event modeling
- Differentiable Greeks with attention for market maker state
- Neural Rough Volatility with fractional dynamics
- MambaTS state-space models for trend forecasting
- Integrated Gradients for faithful explainability
- Importance sampling for efficient tail quantile estimation

## Development Setup

### Environment Setup
```bash
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
```

Set `PYTHONPATH=src` before running CLI or tests:
```bash
$env:PYTHONPATH="src"  # PowerShell
export PYTHONPATH=src  # Bash
```

### Environment Configuration (Optional)

Create a `.env` file for configuration:
```bash
cp .env.example .env
```

**Key points**:
- The app uses **public/free APIs** (ccxt, Deribit, yfinance) - **no API keys required** for basic functionality
- Optional `AETHERIS_API_KEY` for securing the FastAPI service
- Optional SOTA feature flags and PyTorch device configuration
- See `API_KEYS.md` for detailed authentication guide
- See `.env.example` for all available options

**Quick start (no .env needed)**:
```bash
# Works out of the box without any API keys!
python -m aetheris_oracle.cli --asset BTC-USD --horizon 7
```

### Running Tests
```bash
$env:PYTHONPATH="src"
.venv\Scripts\python -m pytest -q
```

## Common Commands

### CLI Forecast
```bash
# Basic forecast
.venv\Scripts\python -m aetheris_oracle.cli --asset BTC-USD --horizon 7 --paths 500

# With scenarios
.venv\Scripts\python -m aetheris_oracle.cli --asset BTC-USD --horizon 7 \
  --iv-multiplier 1.2 --funding-shift 0.01 --narrative key=value

# With CSV connector
.venv\Scripts\python -m aetheris_oracle.cli --connector csv --csv-path data.csv

# With calibration persistence
.venv\Scripts\python -m aetheris_oracle.cli --calibration-path calib.json \
  --realized-price 45000
```

### Training

```bash
# Legacy offline training (synthetic demo)
.venv\Scripts\python -m aetheris_oracle.cli train --horizon 7 --samples 24 \
  --artifact-root artifacts

# SOTA Neural Components Training
# Train all SOTA components
.venv\Scripts\python -m aetheris_oracle.pipeline.train_sota --component all \
  --asset BTC-USD --device cpu --artifact-root artifacts

# Train individual components
.venv\Scripts\python -m aetheris_oracle.pipeline.train_sota --component ncc --epochs 20
.venv\Scripts\python -m aetheris_oracle.pipeline.train_sota --component fmgp --epochs 50
.venv\Scripts\python -m aetheris_oracle.pipeline.train_sota --component neural_jump --epochs 50
.venv\Scripts\python -m aetheris_oracle.pipeline.train_sota --component diff_greeks --epochs 30
.venv\Scripts\python -m aetheris_oracle.pipeline.train_sota --component neural_vol --epochs 30
.venv\Scripts\python -m aetheris_oracle.pipeline.train_sota --component mamba --epochs 40

# Train on GPU (if available)
.venv\Scripts\python -m aetheris_oracle.pipeline.train_sota --component all --device cuda
```

### Service
```bash
# Start FastAPI service
.venv\Scripts\python -m aetheris_oracle.server

# Or via start.py
.venv\Scripts\python start.py --mode service
```

### Evaluation
```bash
# Walk-forward evaluation
.venv\Scripts\python start.py --mode offline_eval
```

## Architecture

### Module Organization
```
src/aetheris_oracle/
├── config.py              # ForecastConfig, ScenarioOverrides (num_paths=10000)
├── data/                  # Data connectors and schemas
│   ├── connectors.py      # SyntheticDataConnector
│   ├── csv_connector.py   # CsvDataConnector
│   ├── free_connectors.py # FreeDataConnector (ccxt + Deribit + yfinance)
│   ├── ccxt_perp_connector.py  # CCXTPerpConnector
│   ├── local_store.py     # LocalFeatureStore (Parquet/CSV)
│   └── schemas.py         # MarketFeatureFrame, RegimeVector, etc.
├── features/              # Feature engineering
│   ├── stationarity.py    # StationarityNormalizer (RevIN-like)
│   └── regime.py          # compute_regime_vector
├── modules/               # Forecasting models (legacy + SOTA neural)
│   ├── trend.py           # TrendEnsemble (legacy)
│   ├── mamba_trend.py     # MambaTrendWrapper (SOTA)
│   ├── vol_path.py        # VolPathEngine (legacy)
│   ├── neural_rough_vol.py # NeuralRoughVolWrapper (SOTA)
│   ├── market_maker.py    # MarketMakerEngine (legacy)
│   ├── differentiable_greeks.py # DifferentiableMMEngine (SOTA)
│   ├── jump.py            # JumpModel (legacy)
│   ├── neural_jump_sde.py # NeuralJumpSDEEngine (SOTA)
│   ├── residual.py        # ResidualGenerator (legacy)
│   └── fm_gp_residual.py  # FMGPResidualEngine (SOTA)
├── pipeline/              # Orchestration and training
│   ├── forecast.py        # ForecastEngine (supports both legacy and SOTA)
│   ├── calibration.py     # CalibrationEngine (legacy)
│   ├── neural_conformal_control.py # NCCCalibrationEngine (SOTA)
│   ├── bellman_conformal.py # BellmanConformalOptimizer (SOTA)
│   ├── scenario.py        # apply_scenario
│   ├── explainability.py  # ExplainabilityEngine (legacy)
│   ├── integrated_gradients.py # IntegratedGradientsExplainer (SOTA)
│   ├── batch_job.py       # run_batch
│   ├── train.py           # run_training (legacy)
│   ├── train_sota.py      # SOTA components training
│   └── offline_evaluation.py  # run_walk_forward
├── monitoring/            # Metrics and evaluation
│   ├── sinks.py           # MetricsSink implementations
│   └── advanced_metrics.py # CRPS, QICE, Energy Score, Conditional FID (SOTA)
├── utils/                 # Utilities
│   └── importance_sampling.py # Importance sampling for tail quantiles (SOTA)
├── api_schemas.py         # Pydantic request/response models
├── api.py                 # API handler
├── server.py              # FastAPI app creation
└── cli.py                 # CLI entrypoint
```

### Forecast Flow

1. **Data Fetching**: `DataConnector.fetch_window()` → `MarketFeatureFrame`
   - Contains: closes, volumes, IV points, funding, basis, order imbalance, narratives, skew

2. **Normalization**: `StationarityNormalizer.normalize_and_stats()` → normalized closes + `NormalizationStats`
   - RevIN-like normalization (zero-mean, unit-variance using past-only windows)

3. **Regime Computation**: `compute_regime_vector()` → `RegimeVector`
   - Aggregates: realized vol, IV, funding, basis, order imbalance, narratives, skew
   - Produces fixed-length regime vector (conditioning for all models)

4. **Model Pipeline**:
   - `TrendEnsemble.predict_trend()` → `TrendPath` (deterministic centerline)
   - `MarketMakerEngine.compute_indices()` → `MarketMakerIndices` (gamma_squeeze, inventory_unwind, basis_pressure)
   - `VolPathEngine.forecast()` → `VolPath` (IV evolution forecast)
   - `ResidualGenerator.sample_paths()` → `ResidualPaths` (stochastic residuals)
   - `JumpModel.sample_jumps()` → `JumpPath` (discrete event modeling)

5. **Path Assembly**: Combine trend + jumps + residuals → price paths (denormalized)

6. **Quantile Extraction**: Compute P5, P10, P25, P50, P75, P90, P95 across paths

7. **Calibration**: `CalibrationEngine.calibrate()` → adjusted quantiles
   - Per regime/horizon buckets
   - Width scaling + coverage tracking

8. **Threshold Probabilities**: Calculate P(price < K), P(price > K) for user thresholds

9. **Explainability**: `ExplainabilityEngine.explain()` + `SurrogateExplainer` → top drivers

### Data Connectors

- **SyntheticDataConnector**: Deterministic synthetic data (for demos/tests)
- **CsvDataConnector**: Aligned CSV with all feature columns
- **LocalFeatureStore**: Parquet/CSV with validation
- **FreeDataConnector**: ccxt spot OHLCV + Deribit IV + yfinance macro (no API keys)
- **CCXTPerpConnector**: Spot + perp funding + order book + basis (with caching)

Select via `--connector` flag or pass to `ForecastEngine` constructor.

### Current Model Status

**Legacy Models** (heuristic stubs, used by default):
- **Trend**: Adaptive-weight blend of long/mid/short exponential smoothing
- **Vol path**: Mean reversion + regime scaling + GARCH-like variance
- **Jump**: Poisson-like with self-excitation, conditioned on vol/regime/MM
- **Residuals**: AR(1) medium-frequency + Gaussian high-frequency + heavy-tail shocks
- **Market Maker**: Simple index computations from IV/funding/basis/imbalance/skew

**SOTA Neural Models** (optional, enable via feature flags):
- **MambaTS Trend**: State-space model with Mamba blocks, multi-scale output heads, and meta-weighting
- **Neural Rough Volatility**: Fractional Brownian motion (H≈0.1) with neural parameterization
- **Differentiable Greeks MM**: Black-Scholes Greeks + multi-head attention over strikes + learned aggregation
- **Neural Jump SDE**: Learned intensity, jump size, and diffusion networks
- **FM-GP Residuals**: Flow matching with Gaussian Process priors and spectral mixture kernel
- **NCC Calibration**: Neural conformal control with PID-style error integration
- **Integrated Gradients**: Path integral of gradients for faithful attribution

**Enable SOTA Components**:
```python
from aetheris_oracle.pipeline.forecast import ForecastEngine

engine = ForecastEngine(
    use_ncc_calibration=True,
    use_fm_gp_residuals=True,
    use_neural_jumps=True,
    use_diff_greeks=True,
    use_neural_rough_vol=True,
    use_mamba_trend=True,
    use_integrated_gradients=True,
    use_importance_sampling=True,
    device="cpu",  # or "cuda"
    # Optional: load pretrained models
    ncc_artifact_path="artifacts/ncc_calibration.pt",
    fmgp_artifact_path="artifacts/fmgp_residuals.pt",
    # ... etc
)
```

Training infrastructure: `pipeline/train_sota.py` for neural models, `pipeline/train.py` for legacy models.

### Calibration

- **Bucketing**: By regime (calm/normal/volatile) and horizon (short/mid/long)
- **Persistence**: JSON serialization (`CalibrationEngine.save()/load()`)
- **Online Updates**: `update_calibration_with_realized()` for post-forecast refinement
- **Coverage Tracking**: Maintains counts of P10-P90 hits per bucket

### API Service

FastAPI service with two endpoints:
- `GET /health` - Health check
- `POST /forecast` - Generate forecast (see `api_schemas.py` for request/response schema)

Optional API key auth via `create_app(api_key="...")` (send `x-api-key` header).

## Key Design Principles

1. **Distributional Forecasts**: Always produce quantile paths, never single-point predictions
2. **Regime Awareness**: All models conditioned on regime vector
3. **Modular Architecture**: Each component (trend, vol, jump, residual) is independent and pluggable
4. **Calibration Layer**: Post-processing to ensure empirical coverage matches theoretical
5. **Scenario Support**: What-if mode via `ScenarioOverrides` (IV multipliers, funding shifts, etc.)
6. **Explainability**: Lightweight driver attribution via surrogate models

## Important Constraints

- Residual paths must have **zero mean** over horizon (to avoid double-counting trend)
- Normalization uses **past-only** windows (no look-ahead bias)
- Scenario forecasts must be clearly labeled as conditional (not for calibration)
- All random seeds should be configurable for reproducibility

## Testing Notes

- Tests use synthetic data via `SyntheticDataConnector`
- Free connectors (`FreeDataConnector`) are mocked in tests (no live network calls)
- Test suite covers: pipeline, calibration, scenarios, connectors, API/service, batch jobs, models
- SOTA component tests: `tests/test_sota_components.py`

## SOTA Components Deep Dive

### Neural Conformal Control (NCC)

**What**: Learned calibration dynamics replacing hand-crafted width scaling.

**Why**: Traditional calibration uses fixed rules. NCC learns optimal calibration adjustments from historical coverage errors.

**Architecture**:
- Feature encoder: maps (regime, vol, MM state) → embedding
- Per-quantile adjustment networks
- PID-style error integrator for online adaptation

**Training**: Optimize for coverage (e.g., P10-P90 should contain 80% of outcomes) + sharpness (tighter intervals preferred)

**Usage**:
```python
from aetheris_oracle.pipeline.neural_conformal_control import NCCCalibrationEngine

engine = NCCCalibrationEngine(device="cpu")
# Train on historical forecasts + outcomes
metrics = engine.train_online(base_quantiles_batch, actuals_batch, features_batch, ...)
```

### FM-GP Residual Generator

**What**: Flow matching with Gaussian Process base distribution.

**Why**: Standard flow matching uses N(0,I) as base. GP base provides principled temporal covariance structure.

**Architecture**:
- Spectral mixture kernel for flexible GP covariance
- Conditional flow matching network
- Classifier-free guidance for stronger conditioning

**Key Property**: Residuals have zero mean by construction (no double-counting trend).

### Neural Jump SDE

**What**: End-to-end learned jump-diffusion process.

**Why**: Legacy jump model uses hand-crafted Poisson intensity. Neural version learns jump dynamics from data.

**Architecture**:
- Intensity network: λ(x, c, t) predicting jump probability
- Jump size network: J(x, c) predicting jump magnitude
- Diffusion network: σ(x, c, t) predicting continuous volatility

**SDE**: dx = μ(x,c,t)dt + σ(x,c,t)dW + J(x,c)dN(λ(x,c,t))

### Differentiable Greeks MM Engine

**What**: Replaces hand-crafted MM indices with learned representation.

**Why**: Current gamma_squeeze/inventory_unwind are heuristics. Differentiable version learns optimal MM state from Greeks.

**Architecture**:
- Differentiable Black-Scholes for delta, gamma, vanna, charm
- Multi-head attention over strike dimension
- Learned aggregation to MM embedding

**Key Benefit**: End-to-end gradient flow allows joint optimization with forecast model.

### Neural Rough Volatility

**What**: Rough volatility dynamics with H ≈ 0.1 (fractional Brownian motion).

**Why**: Empirical finding: log-volatility is rougher (H~0.1) than Brownian motion (H=0.5). Explains vol clustering better.

**Architecture**:
- Fractional kernel: k(s,t) = H(2H-1)|t-s|^(2H-2)
- Neural parameterization of vol-of-vol, correlation, forward variance
- Conditioned on regime + MM state

**Math**: V(t) = V(0) * exp(ξ*fBm_H(t) - 0.5ξ²t^(2H))

### MambaTS Trend

**What**: State-space model based on Mamba architecture.

**Why**: Replaces AR + RNN + Transformer ensemble with unified selective state-space model.

**Architecture**:
- Mamba blocks (linear-time sequence modeling)
- Multi-scale output heads (short/medium/long)
- Meta-weighting network

**Note**: Falls back to SimplifiedMambaBlock if mamba-ssm not installed.

### Integrated Gradients

**What**: Attribution via path integral of gradients from baseline to input.

**Why**: More faithful than permutation importance or SHAP for neural models.

**Math**: IG_i = (x_i - x'_i) ∫₀¹ ∂f(x' + α(x - x'))/∂x_i dα

**Usage**:
```python
from aetheris_oracle.pipeline.integrated_gradients import IntegratedGradientsExplainer

explainer = IntegratedGradientsExplainer()
result = explainer.explain_forecast(features, model_fn)
# result.top_drivers, result.concept_explanations
```

### Importance Sampling

**What**: Reweights samples to improve tail quantile estimates (P5, P95).

**Why**: Standard empirical quantiles need many samples for accurate tails. Importance sampling achieves same accuracy with fewer samples.

**Math**: Use heavier-tailed proposal q(x), reweight by p(x)/q(x).

**Benefit**: Accurate P5/P95 with 10k paths instead of 50k+.

### Advanced Metrics

**CRPS** (Continuous Ranked Probability Score): Proper scoring rule for distributional forecasts.

**QICE** (Quantile Interval Coverage Error): Measures calibration quality across quantile intervals.

**Energy Score**: Multivariate generalization of CRPS.

**Conditional FID**: Evaluates path realism conditioned on regime (adapted from generative models).

**ProbCorr**: Probabilistic correlation for tail dependence.

## Implementation Status

### ✅ Completed (SOTA Components)

1. **Neural Trend (MambaTS)**: ✅ State-space model with multi-scale outputs
2. **Neural Volatility**: ✅ Rough volatility with H ≈ 0.1
3. **Neural Residuals (FM-GP)**: ✅ Flow matching with GP priors
4. **Neural Jumps (SDE)**: ✅ Learned intensity + jump size + diffusion
5. **Neural MM (Diff Greeks)**: ✅ Attention over Greeks with learned aggregation
6. **Neural Calibration (NCC)**: ✅ Conformal control with error integration
7. **Advanced Explainability**: ✅ Integrated Gradients for faithful attribution
8. **Efficient Sampling**: ✅ Importance sampling for tail quantiles
9. **Advanced Metrics**: ✅ CRPS, QICE, Energy Score, Conditional FID
10. **Training Infrastructure**: ✅ `train_sota.py` with CLI for all components
11. **Testing**: ✅ Comprehensive test suite in `test_sota_components.py`
12. **Integration**: ✅ ForecastEngine supports both legacy and SOTA via feature flags

### 🚧 Next Steps (Production Deployment)

1. **Real Data Training**: Train SOTA models on real historical data (currently using synthetic)
2. **Hyperparameter Tuning**: Grid search / Bayesian optimization for each component
3. **Walk-Forward Validation**: Systematic backtesting with CRPS/coverage metrics
4. **Model Persistence**: Version control for trained artifacts with metadata
5. **A/B Testing**: Compare legacy vs SOTA in production with gradual rollout
6. **Monitoring**: Real-time coverage tracking + drift detection
7. **Documentation**: User guide for practitioners (non-technical)

### 📊 Expected Performance (Post-Training on Real Data)

- **Calibration Coverage**: P10-P90 should achieve 78-82% hit rate (vs 70-75% legacy)
- **Sharpness**: 15-20% tighter intervals at same coverage (via NCC)
- **Tail Accuracy**: P5/P95 CRPS improvement of 25-30% (via importance sampling + FM-GP)
- **Event Capture**: 40-50% better jump detection (via Neural Jump SDE)
- **Inference Speed**: ~2x slower than legacy (acceptable tradeoff for quality)

Models are saved to `artifacts/` directory and auto-loaded by ForecastEngine.
