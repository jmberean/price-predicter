Implement this app and dont stop until all the tasks are complete 

# Aetheris Oracle v10.0

### System Design & Engineering Specification

**Purpose:** N-day probabilistic forecasting engine for financial time series (crypto + optionally other assets)
**Primary Horizon:** 1–14 days (configurable)

---

## 1. Scope & Objectives

### 1.1 Business Objective

Build a system that:

* Predicts the **distribution of future prices**, not a single point.
* Produces **forecast cones** (P10/P50/P90 and optionally more quantiles).
* Is **robust across market regimes** (calm, volatile, crisis).
* Can be integrated into:

  * internal trading & risk dashboards,
  * research pipelines,
  * potentially external APIs.

### 1.2 Out-of-Scope (for v10.0)

* Live order execution or trading logic (no OMS/EMS).
* High-frequency intraday trading (< 5 min horizons).
* Full-text ingestion at production scale (we’ll only support simple topic/narrative inputs or precomputed embeddings initially).
* Full altcoin hierarchy modeling (basic cross-asset features only in v10.0).

---

## 2. Functional Requirements

1. **Input**:

   * Asset identifier (e.g., `BTC-USD`)
   * Forecast horizon `N` (1–14 days)
   * As-of timestamp (T0)
   * Optional scenario overrides (for “what-if” mode only)

2. **Output**:

   * Time-indexed quantile paths:

     * P5, P10, P25, P50, P75, P90, P95 (configurable)
   * Probability of hitting user-defined thresholds (e.g., P(price < K), P(price > K)).
   * Optional “driver summary” for explainability:

     * “Top 3 drivers for elevated upside/downside risk.”

3. **Performance**:

   * Inference latency:

     * Target: < 1 sec per asset / forecast on GPU-backed deployment.
   * Batch inference for multiple assets must be supported (nightly jobs).

4. **Reliability**:

   * Empirical P10–P90 coverage close to 90% over rolling windows.
   * System degradation behavior defined when some data sources are missing.

---

## 3. Non-Functional Requirements

* **Availability**: 99% for inference API (internal SLA).
* **Resilience**: Graceful fallback when some features (e.g., options data) are missing.
* **Security**:

  * Read-only access to market data sources.
  * No PII.
* **Observability**:

  * Metrics: CRPS, coverage, latency, data freshness.
  * Logs: input snapshot, model version, prediction summary.

---

## 4. Data & Feature Specification

### 4.1 Assets Supported Initially

* BTC-USD
* ETH-USD
* (Optional) 3–5 additional large-cap assets once system is stable.

### 4.2 Data Sources

Engineers must provide **connectors** for:

1. **Historical & Live Market Data**

   * OHLCV (1h or 4h resolution).
   * Tick or trade data optional.

2. **Order Book & Microstructure**

   * L2 order book snapshots (depth levels, e.g., top 10).
   * Trade volume, trade direction stats (buy vs sell).

3. **Derivatives / Options**

   * Implied volatility surface points:

     * At least 3–5 expiries (e.g., 7d, 14d, 30d, 90d).
     * At least 2–3 deltas (e.g., ATM, 25Δ call, 25Δ put).
   * Option open interest by strike buckets (e.g., buckets around current spot).

4. **Futures / Perpetuals**

   * Funding rates (hourly/daily).
   * Perps vs spot basis.
   * OI and volume metrics.

5. **Macro & Vol Indices**

   * DXY, VIX (or crypto equivalent).
   * 2y/10y yields.

6. **On-Chain Data (Crypto)**

   * Net exchange inflows/outflows.
   * Stablecoin net issuance.

7. **Narrative / Topic Signals (simplified for v10.0)**

   * Precomputed topic dominance indices (e.g., from a separate NLP system):

     * `RegulationRisk`, `ETF_Narrative`, `TechUpgrade`, `MemecoinHype`, etc.
   * These are numeric time series, not raw text.

---

### 4.3 Data Storage & Access Layer

* Use a **time-series database** or well-partitioned columnar store:

  * Example: TimescaleDB, ClickHouse, or Parquet on object storage (S3/GCS) with a metadata index.
* All data referenced by:

  * `asset_id`
  * `timestamp`
  * `feature_group` (price, IV, micro, macro, narrative)

Minimal schema:

* Table: `market_features`

  * `asset_id`
  * `timestamp` (UTC)
  * `feature_name` (e.g., `ohlc.close`, `iv_7d_atm`, `funding_rate`)
  * `value`

Data ingestion & quality checks are a separate pipeline owned by data engineering.

---

## 5. Model Architecture Overview

### 5.1 Conceptual Decomposition

The model is composed of **five major modules**:

1. **Stationarity Layer**
2. **Trend Ensemble**
3. **Market-Maker Inventory Engine**
4. **Volatility Path Engine (IV / skew forecasting)**
5. **Residual Generative Engine**
6. **Calibration & Regime Layer**
7. **Explainability & Scenarios (separate output/UX modules)**

Each module has clear **inputs, outputs, and training scope**.

---

## 6. Stationarity & Regime Encoding

### 6.1 Stationarity (RevIN-like)

Per asset & per forecast time:

* Compute rolling statistics over trailing window (e.g., last 90 days of 1h candles):

  * mean, std for key price-related features.
* Normalize:

  * All main time-series inputs (price, returns, some IV features) are scaled to zero-mean, unit-variance using **past-only** windows.
* Store normalization parameters (`μ`, `σ`) per asset/time for later denormalization.

### 6.2 Regime Embedding

Compute regime descriptors over the same window:

* Realized volatility (short & long horizon).
* Implied volatility level and skew.
* Funding rate & basis percentiles.
* Order book imbalance summary.
* Narrative/topic indices.

These are aggregated into a **regime vector** (fixed-length, e.g., 32–64 dims).

Regime vector is passed to:

* Trend ensemble
* Market-maker engine
* Vol path engine
* Residual generative model
* Calibration module

---

## 7. Trend Ensemble Module

### 7.1 Purpose

Produce the **deterministic backbone** of the forecast: a centerline trend for 1–14 days.

### 7.2 Components

1. **Long-Horizon Model (SSM / Mamba-like):**

   * Input: normalized macro, IV, regime embedding.
   * Captures long-term behavior, structural drift.

2. **Local Statistical Model (e.g., AR/ARIMA / exponential smoothing):**

   * Input: recent log-returns.
   * Captures short-term mean reversion, local structure.

3. **Mid-Horizon Temporal Model (light Transformer / TCN / TFT-lite):**

   * Input: mixed features (macro + price).
   * Captures cyclic patterns, 3–7 day structures.

### 7.3 Aggregation: Meta-Weighting

* **Offline**: Meta-model (small network) learns initial weights per regime.
* **Online**: Adaptive reweighting (e.g., EWAF / exponential weighting) updates weights daily based on recent forecast errors of each component.

Output:

* A single **trend path** of length `N` (or possibly a small ensemble of trend paths).

---

## 8. Market-Maker Inventory Engine

### 8.1 Purpose

Quantify **dealer / market-maker flows & pressure**:

* Gamma squeezes
* Inventory unwind risk
* Basis pressure

### 8.2 Inputs

* Options OI & strikes (binned).
* IV surface relative to current price.
* Funding rates & changes.
* Basis (perps vs spot).
* Stablecoin flow metrics.

### 8.3 Outputs

Compute a **compact set of indices**, e.g.:

* `Gamma_Squeeze_Index`
* `Inventory_Unwind_Index`
* `Basis_Pressure_Index`

These scores are standardized and bounded (e.g., −3 to +3).

Usage:

* Condition the **Jump Model**:

  * High Gamma_Squeeze → higher probability & magnitude of up-jumps or rapid mean reversion after event.
* Condition the **Residual Generator**:

  * Inventory_Unwind → more mean-reverting noise.

---

## 9. Volatility Path Engine

### 9.1 Purpose

Forecast the **evolution of implied volatility and skew** over the next `N` days.

### 9.2 Inputs

* Past IV term structure time series.
* Past skew indices.
* Realized volatility & regime embedding.
* Market-maker indices.

### 9.3 Output

* `IV_path[t+1…t+N]` (for key expiries)
* `Skew_path[t+1…t+N]`

These are used to:

* Update **future Gamma_Squeeze** and **jump probabilities**.
* Condition residual generator (higher predicted vol leads to wider residuals).

---

## 10. Residual Generative Engine

### 10.1 Purpose

Model **residual volatility & noise** after trend and jumps:

* Realistic, jagged paths
* Tail heaviness
* Autocorrelation
* Regime-dependent volatility

### 10.2 Residual Definition

Residuals are modeled on normalized log-returns:

* Compute theoretical “trend+jump” path.
* Define residual returns as:

  * `resid_t = log(price_t) − log(trend_t + jump_t)` for training.

### 10.3 Multi-Resolution Design (Conceptual)

* Decompose residual series into:

  * Medium-frequency component: captured via simple SSM/AR.
  * High-frequency component: captured via generative model.

### 10.4 Generative Model Type

For v10.0, we **standardize on flow-like or rectified-flow model** (engineers can choose exact implementation, but conceptually):

* Input:

  * Noise samples
  * Conditioning: regime vector, trend latent state, IV future, MM indices.
* Output:

  * `num_paths` residual paths (`N` steps each).

Constraints:

* Residual paths must have **zero mean** over horizon (enforced via training or post-processing).
* Model must support **at least 1,000 paths** per inference with reasonable latency (configurable).

---

## 11. Jump Modeling

### 11.1 Purpose

Model discrete, rare events:

* ETF decisions
* Liquidation cascades
* Major hacks / regulatory announcements

### 11.2 Inputs

* Regime embedding.
* Volatility & IV paths.
* Market-maker indices.
* Event calendar proximity.

### 11.3 Behavior

For each simulated path:

* Sample number of jumps over horizon (Poisson or Hawkes-style, conditioned on volatility and events).
* Sample jump times (especially around event boundaries).
* Sample jump magnitudes (heavy-tailed distribution, direction influenced by:

  * skew,
  * narrative,
  * inventory / gamma pressure).

Jumps are **additive components** to trend path before residuals.

---

## 12. Path Reconstruction & Output

For each simulation `i`:

1. Compute trend path: `trend_i[t]`.
2. Sample jump path: `jump_i[t]`.
3. Sample residual path: `resid_i[t]`.
4. Combine in normalized space and denormalize via stored `μ`, `σ`:

   `price_i[t] = denorm(trend_i[t] + jump_i[t] + resid_i[t])`.

Across all `num_paths`:

* Compute quantiles per time-step (P5, P10, P25, P50, P75, P90, P95).
* Package as final forecast cone.

---

## 13. Calibration Engine

### 13.1 Goal

Ensure the predicted cones **match observed frequencies** in practice.

### 13.2 Regime- & Horizon-Aware Conformal Scheme

Maintain calibration buffers:

* Per regime (e.g., calm / normal / volatile).
* Per horizon bucket (short, mid, long).

For each forecast horizon & regime:

* Track historical error quantiles: how often actual values fall below predicted quantiles.
* Compute conformal adjustments: widen/narrow cones to match target coverage.

Calibration runs as a **post-processing layer**:

* It uses past forecast vs realized outcomes.
* It does **not retrain** the model; it adjusts the quantile bounds.

---

## 14. Explainability Layer

Separate from model core; runs after baseline forecast.

### 14.1 What We Explain

For P50 and maybe cone width:

* Which feature blocks contributed most to:

  * Upside risk (wider upper cone),
  * Downside risk,
  * Trend direction.

Feature blocks:

* Trend ensemble components
* IV / vol features
* MM inventory indices
* Funding & basis
* On-chain flows

### 14.2 Approach

Use **approximate attribution**, NOT heavy SHAP on the full model:

* Train a small surrogate model that approximates P50 / cone width from high-level feature aggregates.
* Run feature permutation / sensitivity analysis on the surrogate.
* Return “Top 3 drivers” as metadata along with forecast.

---

## 15. Scenario Engine (What-If Mode)

### 15.1 Purpose

Allow risk managers to ask:

> “What if X changes (e.g., IV spikes, funding flips) — how does the cone change?”

### 15.2 Constraints

* Only whitelisted features may be overridden:

  * IV level, skew,
  * correlation with BTC,
  * funding rates,
  * GSI / inventory index.
* Values must pass **sanity checks** (no impossible combinations).

### 15.3 Flow

1. User submits scenario overrides.
2. System constructs a **modified input feature set** (does not change historical data).
3. Rerun inference pipeline with new conditioning.
4. Output scenario forecast clearly labeled as **conditional**.

Scenario forecasts MUST NOT be used for calibration.

---

## 16. Pipelines

### 16.1 Offline Training Pipeline

1. **Data Extraction**:

   * Pull historical data for chosen horizon (e.g., last 3–5 years).
   * Align all modalities by timestamp.

2. **Feature Engineering**:

   * Compute regime descriptors, MM indices, etc.
   * Produce training samples:

     * input windows → future paths.

3. **Module Training**:

   * Train trend ensemble components.
   * Train IV path model.
   * Train jump model.
   * Train residual generative model.
   * Optionally pretrain modules separately, then fine-tune jointly.

4. **Initial Calibration Fit**:

   * Perform walk-forward validation.
   * Fit initial conformal adjustment per regime and horizon.

5. **Model Versioning**:

   * Each trained model is tagged with:

     * version ID,
     * training data ranges,
     * hyperparameters.

### 16.2 Online Inference Pipeline

For a given asset & T0:

1. Fetch latest features & normalization stats.
2. Construct regime embedding.
3. Run trend ensemble (with current weights).
4. Run vol path engine.
5. Run MM inventory engine.
6. Sample jumps.
7. Sample residual paths.
8. Rebuild price paths and compute quantiles.
9. Apply calibration.
10. (Optionally) run explainability surrogate.

### 16.3 Online Adaptive Components

* Daily or hourly:

  * Update ensemble weights based on yesterday’s forecast errors.
  * Update calibration buffers with realized outcomes.
  * Optionally fine-tune models on recent data (subject to safety checks).

---

## 17. Infrastructure & Deployment

### 17.1 Environments

* **Dev**: small subset of assets, reduced path count.
* **Staging**: full assets, full features, lower concurrency.
* **Prod**: full assets, full paths, with monitoring and alerts.

### 17.2 Services

* Data service (time-series feature retrieval).
* Model inference service:

  * Single API for forecast.
* Calibration service:

  * Maintains calibration state and metrics.
* Monitoring & metrics service.

### 17.3 Interfaces (High-Level)

**Forecast API (internal)**:

* `POST /forecast`

  * Body:

    * `asset_id`
    * `horizon`
    * `as_of_timestamp`
    * optional `scenario_overrides`
  * Response:

    * time-indexed quantiles
    * driver summary
    * model & calibration version IDs

---

## 18. Monitoring, Validation & Maintenance

### 18.1 Metrics

* CRPS per horizon
* Coverage per horizon & regime
* Average cone width
* Tail miss frequency (out-of-cone large moves)
* Latency & error rate

### 18.2 Dashboards

* Daily CRPS & coverage panel.
* Regime breakdown panel.
* Latency panel.

### 18.3 Alerts

* Undercoverage below threshold (e.g., P10–P90 < 85%).
* Latency above SLA.
* Missing data for critical features.

---

## 19. Implementation Priorities (MVP vs Full v10.0)

**MVP (Phase 1–2)**

* Single asset (BTC-USD).
* Stationarity layer + regime embedding.
* Trend ensemble (Mamba + AR).
* Simplified MM index (funding + basic OI + IV).
* Simplified IV path (1–2 key points).
* Residual generator (flow or simple stochastic process).
* Basic conformal calibration (one global buffer).

**Full v10.0**

* Multiple assets (BTC, ETH, etc.).
* Full MM inventory engine.
* Full IV path & skew forecasting.
* Regime/horizon-aware calibration.
* Scenario engine.
* Explainability layer.