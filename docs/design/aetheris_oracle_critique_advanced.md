# Critical Evaluation: Aetheris Oracle v10.0

## Probabilistic N-Day Forecasting Engine Design Review
### Advanced Implementation Focus

---

## 1. First Pass: Understanding & Restating the Plan

### 1.1 Core Business Goal and Use Cases

The Aetheris Oracle v10.0 aims to be a **probabilistic forecasting engine** for cryptocurrency (and potentially other assets) that:

- Produces **forecast cones** (P5 through P95 quantiles) over 1–14 day horizons
- Provides **distribution estimates** rather than single-point predictions
- Maintains **robustness across market regimes** (calm, volatile, crisis)
- Integrates into trading dashboards, research pipelines, and potentially external APIs
- Generates **driver summaries** for explainability ("Why is upside/downside risk elevated?")
- Supports **what-if scenario analysis** for risk managers

### 1.2 Main Architecture & Modules

| Module | Purpose |
|--------|---------|
| **Stationarity Layer (RevIN-like)** | Normalize inputs using rolling statistics; store μ, σ for denormalization |
| **Regime Embedding** | 32-64 dim vector capturing RV, IV, funding, basis, order book, narratives |
| **Trend Ensemble** | Backbone forecast combining SSM/Mamba (long-horizon), AR/ARIMA (local), Transformer/TCN/TFT (mid-horizon) with meta-weighting |
| **Market-Maker Inventory Engine** | Outputs indices: Gamma_Squeeze_Index, Inventory_Unwind_Index, Basis_Pressure_Index |
| **Volatility Path Engine** | Forecasts IV and skew paths over N days |
| **Residual Generative Engine** | Flow/rectified-flow model producing realistic "jitter" conditioned on regime |
| **Jump Model** | Poisson/Hawkes-style rare event sampling (ETF decisions, liquidations, hacks) |
| **Calibration Engine** | Regime- & horizon-aware conformal adjustment post-processing |
| **Explainability Layer** | Surrogate model + permutation importance → "Top 3 drivers" |
| **Scenario Engine** | What-if mode with constrained feature overrides |
| **Data Layer** | TimescaleDB/ClickHouse/Parquet with TSDB schema |
| **Pipelines** | Offline training, online inference, daily adaptive updates |

### 1.3 Intended Outputs

1. **Time-indexed quantile paths** (P5, P10, P25, P50, P75, P90, P95)
2. **Threshold probabilities**: P(price < K), P(price > K)
3. **Driver summary**: Top 3 factors contributing to upside/downside risk
4. **Scenario forecasts**: Conditional cones given user-specified overrides
5. **Metadata**: Model version, calibration version, input snapshot

### 1.4 Unstated Assumptions

The plan implicitly assumes but does not explicitly address:

1. **Data availability & quality**
   - Reliable, clean options data at required granularity (3-5 expiries, 2-3 deltas) — historically this is fragmented and noisy for crypto
   - Continuous and timely on-chain data feeds with consistent semantics
   - Accurate trade direction classification (buy vs. sell) from exchanges that may not provide it

2. **Compute budget**
   - GPU-backed deployment for <1s inference with 1,000+ path sampling
   - Sufficient storage for 3-5 years of multi-modal historical data
   - Budget for daily/hourly retraining of adaptive components

3. **Team skillset**
   - Expertise in flow-matching/generative models (cutting-edge ML research area)
   - Deep understanding of options market microstructure and dealer hedging mechanics
   - MLOps maturity for managing model versioning, calibration states, and drift detection

4. **Temporal stability**
   - Exchange microstructure remains stable (listing rules, tick sizes, fee structures)
   - Derivative market conventions don't change dramatically
   - On-chain metrics maintain consistent meaning (e.g., "exchange inflow" definition)

5. **Stationarity of regime definitions**
   - The regime buckets (calm/normal/volatile) remain meaningful over time
   - No emergence of entirely new market structures (novel derivative products, new DeFi primitives)

6. **Backtesting infrastructure**
   - Point-in-time data reconstruction capability for walk-forward validation
   - Feature versioning and reproducibility

---

## 2. Deep Critique by Dimension (Advanced Focus)

### 2.1 Scope, Requirements, and Success Criteria

**What the plan proposes:**
- Coverage target: P10-P90 ≈ 90%
- Latency: <1s per asset/forecast on GPU
- Internal SLA: 99% availability
- Metrics: CRPS, coverage, cone width, tail miss frequency

**Evaluation:**

✅ **Strengths:**
- CRPS is an appropriate proper scoring rule for probabilistic forecasts
- Coverage and tail miss frequency are directly tied to business value (risk management)
- Latency target is reasonable for 1-14 day horizons

⚠️ **Gaps for Advanced Implementation:**

1. **CRPS targets lack ambition benchmarking**: Should target specific improvement over state-of-the-art baselines (e.g., "15% better CRPS than Chronos zero-shot, 30% better than GARCH")

2. **Calibration metrics should be more sophisticated**: Beyond coverage, track:
   - **Quantile Interval Coverage Error (QICE)** — per-quantile calibration
   - **Conditional FID** — quality of generated path distributions
   - **ProbCorr** — correlation preservation in probabilistic forecasts

3. **Sharpness-calibration tradeoff unspecified**: Define explicit sharpness targets via Mean Prediction Interval Width (MPIW) bounds

4. **Advanced requirements to add:**
   - **Tail coherence**: P5 predictions should be more extreme than P10 (no quantile crossing)
   - **Temporal consistency**: Multi-step forecasts should be path-consistent
   - **Multimodality detection**: System should flag when predictive distribution is multimodal

**Concrete improvements:**

1. Add QICE, Conditional FID, ProbCorr to metrics suite
2. Define per-horizon CRPS targets relative to foundation model baselines
3. Specify MPIW upper bounds per horizon
4. Add path consistency requirements for multi-step forecasts

---

### 2.2 Data & Features

**What the plan proposes:**
- OHLCV (1h/4h), order book L2, trade direction stats
- IV surfaces (3-5 expiries, 2-3 deltas), options OI by strike
- Funding rates, basis, perpetual OI
- Macro indices (DXY, VIX, yields)
- On-chain flows (exchange in/outflows, stablecoin issuance)
- Narrative indices (precomputed topic dominance)
- Storage: TimescaleDB/ClickHouse/Parquet with simple schema

**Evaluation:**

✅ **Strengths:**
- Comprehensive feature set covering multiple information channels
- Narrative indices abstraction avoids NLP complexity in v10.0
- Standard TSDB choices are appropriate

⚠️ **Data Fragility Risks (Must Address Regardless of Approach):**

1. **Options data quality (CRITICAL)**
   - Crypto options IV surfaces are notoriously noisy and incomplete
   - Different exchanges quote differently; standardization is non-trivial
   - Strike buckets around spot require dynamic re-bucketing as price moves
   - Stale quotes can persist in low-liquidity strikes
   - **Required**: Arbitrage-free IV surface fitting (e.g., SVI parameterization with no-arbitrage constraints)

2. **Trade direction classification**
   - Most crypto exchanges don't provide native trade aggressor
   - Lee-Ready algorithm requires millisecond-level data; plan mentions 1h/4h OHLCV
   - **Required**: Either acquire tick data or use volume-clock based alternatives

3. **On-chain data semantics drift**
   - "Exchange inflow" definitions vary by provider and change over time
   - New wallet heuristics can retroactively change historical data
   - **Required**: Version all on-chain feature computations; maintain point-in-time snapshots

4. **Full OI surface needed for advanced MM engine**
   - Plan mentions "binned" OI — but differentiable Greeks approach needs full surface
   - **Required**: Store full strike-by-strike OI, not just buckets

**Schema upgrade for advanced use:**

The proposed EAV schema is insufficient. For differentiable operations on options data:

```
-- Options surface table (for differentiable Greeks)
CREATE TABLE options_surface (
    asset_id TEXT,
    timestamp TIMESTAMPTZ,
    expiry DATE,
    strike NUMERIC,
    option_type TEXT,  -- 'call' or 'put'
    iv NUMERIC,
    oi NUMERIC,
    delta NUMERIC,
    gamma NUMERIC,
    vega NUMERIC,
    PRIMARY KEY (asset_id, timestamp, expiry, strike, option_type)
);

-- Pre-computed features (wide table for fast retrieval)
CREATE TABLE feature_vectors (
    asset_id TEXT,
    timestamp TIMESTAMPTZ,
    feature_version TEXT,
    features FLOAT8[],  -- Dense vector
    PRIMARY KEY (asset_id, timestamp, feature_version)
);
```

**Concrete improvements:**

1. **Implement SVI surface fitting**: Arbitrage-free IV surface with quality scores
2. **Store full options surface**: Strike-by-strike for differentiable Greeks computation
3. **Add feature versioning**: Track computation code version with each feature
4. **Upgrade schema**: Wide tables or feature store for training efficiency
5. **Quality monitoring**: Per-source freshness, null rates, distribution drift alerts

---

### 2.3 Model Architecture & Decomposition

#### Overall Architecture Assessment

**What the plan proposes:**
A modular decomposition: Stationarity → Trend → MM Engine → Vol Path → Residuals → Jumps → Calibration → Explain

**Advanced Evaluation:**

The modular approach is reasonable for interpretability, but the current design has **integration gaps** that will compound errors. Two architectural paradigms are more advanced:

**Option A: Latent Diffusion with Structured Decoder (Recommended)**
```
Input Features → Encoder → Latent z
                              ↓
                    Diffusion Process in Latent Space
                              ↓
                    Structured Decoder → (trend, vol, jumps)
                              ↓
                    Path Reconstruction
```
- Single end-to-end training with auxiliary losses for each component
- Latent space captures regime implicitly
- Decoder outputs are interpretable but jointly optimized
- Avoids sequential error compounding

**Option B: Hierarchical State-Space Model**
```
Level 1 (Slow): Regime dynamics (discrete latent, HMM-like)
       ↓
Level 2 (Medium): Trend + Vol level (continuous latent, linear SSM)
       ↓
Level 3 (Fast): Residuals + Jumps (neural SDE)
```
- Structured prior encodes timescale separation
- All levels learned jointly
- Interpretable latent hierarchy

**If keeping modular approach**, fix these issues:

1. **Circular dependency**: IV path depends on MM indices, MM indices depend on IV
   - **Fix**: Compute simultaneously via shared hidden state, not sequentially

2. **Residual definition error** (CRITICAL BUG)
   - `resid_t = log(price_t) − log(trend_t + jump_t)` is mathematically wrong
   - Cannot add prices then log; should be:
   - **Fix**: `resid_t = log(price_t / trend_t) - jump_return_t` (multiplicative)
   - Or work entirely in return space

3. **Missing feedback loops**
   - Jumps affect vol, vol affects jump probability
   - **Fix**: Use neural SDE with coupled dynamics

---

#### Trend Ensemble — Upgrade to SOTA

**What the plan proposes:**
- SSM/Mamba for long-horizon (macro, IV, regime)
- AR/ARIMA for short-term mean reversion
- Transformer/TCN/TFT for mid-horizon cycles
- Meta-weighting with EWAF online adaptation

**Advanced Assessment:**

The ensemble of three separate models is **less sophisticated** than unified architectures. Current SOTA:

| Plan Component | SOTA Replacement | Why Better |
|----------------|------------------|------------|
| Generic "Mamba-like" | **Mamba-2** (Dao & Gu, 2024) | 2-8x faster, better hardware utilization via SSD (State Space Duality) |
| Separate AR component | **Integrated into MambaTS** | Temporal Mamba Block handles local structure |
| Separate Transformer | **Mamba + sparse attention hybrid** | Best of both: linear complexity + explicit attention where needed |
| Three-model ensemble | **MambaTS with VAST** | Single model learns optimal variable ordering, handles all horizons |

**Recommended Architecture:**

```python
class AdvancedTrendBackbone(nn.Module):
    def __init__(self):
        # MambaTS with Variable-Aware Scan
        self.mamba = MambaTS(
            d_model=256,
            n_layers=6,
            use_vast=True,  # Variable-aware scanning
            bidirectional=True
        )
        # Sparse attention for explicit cross-asset dependencies
        self.cross_attention = SparseAttention(
            n_heads=8,
            sparsity_pattern='learned'
        )
        # Multi-scale output heads
        self.heads = nn.ModuleDict({
            'short': nn.Linear(256, horizon),   # 1-3 day
            'medium': nn.Linear(256, horizon),  # 4-7 day
            'long': nn.Linear(256, horizon)     # 8-14 day
        })
```

**Key papers to implement from:**
- "MambaTS: Improved Selective State Space Models for Long-term Time Series Forecasting" (Cai et al., 2024)
- "Mamba-2: Structured State Space Models with Improved Training" (Dao & Gu, 2024)
- "SOR-Mamba: Sequential Order-Robust Mamba" (ICLR 2025 submission)

---

#### Market-Maker Inventory Engine — Go Differentiable

**What the plan proposes:**
- Inputs: Options OI, IV surface, funding rates, basis, stablecoin flows
- Outputs: Gamma_Squeeze_Index, Inventory_Unwind_Index, Basis_Pressure_Index (standardized, bounded)

**Advanced Assessment:**

Hand-crafted indices are **less expressive** than learned representations. The cutting-edge approach:

**Differentiable Greeks Layer:**

```python
class DifferentiableMMEngine(nn.Module):
    def __init__(self):
        # Differentiable Black-Scholes for Greeks
        self.bs_layer = DifferentiableBS()
        
        # Attention over strikes to learn importance
        self.strike_attention = nn.MultiheadAttention(
            embed_dim=64, num_heads=4
        )
        
        # Aggregation to MM state
        self.aggregator = nn.Sequential(
            nn.Linear(64 * n_expiries, 128),
            nn.GELU(),
            nn.Linear(128, 64)  # MM embedding
        )
    
    def forward(self, spot, iv_surface, oi_surface, funding, basis):
        # Compute Greeks differentiably
        greeks = self.bs_layer(spot, iv_surface)  # (delta, gamma, vanna, charm)
        
        # Weight by OI
        weighted_greeks = greeks * oi_surface
        
        # Attention over strikes
        strike_embeddings = self.strike_encoder(weighted_greeks)
        attended, weights = self.strike_attention(
            strike_embeddings, strike_embeddings, strike_embeddings
        )
        
        # Aggregate with funding/basis
        mm_state = self.aggregator(
            torch.cat([attended.flatten(), funding, basis], dim=-1)
        )
        
        return mm_state, weights  # weights provide interpretability
```

**Advantages over index approach:**
1. **End-to-end learning**: Model learns what matters for forecasting
2. **Differentiable**: Gradients flow through Greeks computation
3. **Interpretable via attention**: Which strikes matter? Attention weights tell you
4. **No manual tuning**: No need to define index formulas

**Implementation note**: Use JAX or PyTorch with custom autograd for Black-Scholes Greeks. Libraries like `pytorch-option-pricing` or similar provide differentiable implementations.

**Key papers:**
- "Deep Hedging" (Bühler et al., 2019) — differentiable derivatives framework
- "Gamma positioning and market quality" (ScienceDirect, 2024) — academic grounding

---

#### Volatility Path Engine — Neural Rough Volatility

**What the plan proposes:**
- Forecast IV and skew paths for N days
- Inputs: Past IV term structure, skew, RV, regime, MM indices
- Outputs condition jump probabilities and residual generator

**Advanced Assessment:**

The plan is vague. For true SOTA, implement **Neural Rough Volatility**:

**Why Rough Volatility?**
- Empirically, log-volatility behaves like fractional Brownian motion with H ≈ 0.1
- This explains: volatility clustering, term structure shapes, skew dynamics
- Classical models (Heston, SABR) assume H = 0.5 and miss these effects

**Neural Rough Vol Architecture:**

```python
class NeuralRoughVolEngine(nn.Module):
    def __init__(self, hurst=0.1):
        self.hurst = hurst
        
        # Fractional kernel (rough Bergomi style)
        self.frac_kernel = FractionalKernel(H=hurst)
        
        # Neural network for vol-of-vol and mean reversion
        self.vol_dynamics = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 3)  # (xi, rho, forward_var)
        )
        
        # Conditional on regime and MM state
        self.conditioning = FiLM(regime_dim + mm_dim, 128)
    
    def forward(self, past_vol, regime_embed, mm_embed, horizon):
        # Get rough vol parameters conditioned on state
        params = self.vol_dynamics(
            self.conditioning(past_vol, torch.cat([regime_embed, mm_embed]))
        )
        xi, rho, fwd_var = params.unbind(-1)
        
        # Simulate rough paths
        vol_paths = self.simulate_rough_bergomi(
            xi, rho, fwd_var, self.hurst, horizon
        )
        
        return vol_paths
    
    def simulate_rough_bergomi(self, xi, rho, fwd_var, H, horizon):
        # Hybrid scheme for rough Bergomi simulation
        # Uses Cholesky on fractional covariance
        ...
```

**Key papers:**
- "Volatility is Rough" (Gatheral, Jaisson & Rosenbaum, 2018)
- "Deep Learning Volatility" (Horvath et al., 2021)
- "Neural SDEs as Infinite-Dimensional GANs" (Kidger et al., 2021)

---

#### Residual Generative Engine — Flow Matching with GP Priors

**What the plan proposes:**
- Flow/rectified-flow model conditioned on regime, trend, IV, MM indices
- Zero-mean constraint; 1,000+ paths at inference

**Advanced Assessment:**

Flow matching is the right direction, but vanilla implementation lacks proper uncertainty quantification. **Upgrade to FM-GP:**

**Flow Matching with Gaussian Process Priors (ICLR 2025):**

```python
class FMGPResidualGenerator(nn.Module):
    """
    Flow Matching with GP prior for time-series residuals.
    Key insight: Use GP as the base distribution instead of standard Gaussian.
    This encodes temporal correlation structure into the prior.
    """
    def __init__(self, horizon, cond_dim):
        # GP kernel for prior (learns temporal correlation)
        self.gp_kernel = SpectralMixtureKernel(
            num_mixtures=4,
            input_dim=1  # time dimension
        )
        
        # Conditional flow network
        self.flow_net = ConditionalVectorField(
            data_dim=horizon,
            cond_dim=cond_dim,
            hidden_dims=[256, 256, 256],
            time_embed_dim=64
        )
        
        # Classifier-free guidance for stronger conditioning
        self.cfg_dropout = 0.1
    
    def sample(self, conditioning, n_paths=1000):
        # Sample from GP prior (not standard Gaussian!)
        gp_samples = self.sample_gp_prior(n_paths, self.gp_kernel)
        
        # Flow from GP prior to data distribution
        paths = self.ode_solve(
            gp_samples, 
            conditioning,
            t_span=[0, 1]
        )
        
        return paths
    
    def training_loss(self, x, conditioning):
        # Conditional flow matching loss with GP prior
        t = torch.rand(x.shape[0])
        
        # Interpolate between GP sample and data
        x0 = self.sample_gp_prior(x.shape[0], self.gp_kernel)
        xt = t * x + (1 - t) * x0
        
        # Target velocity
        v_target = x - x0
        
        # Predicted velocity (with CFG dropout)
        if torch.rand(1) < self.cfg_dropout:
            conditioning = None  # Unconditional
        v_pred = self.flow_net(xt, t, conditioning)
        
        return F.mse_loss(v_pred, v_target)
```

**Advantages over vanilla flow:**
1. **Structured prior**: GP encodes expected temporal correlation
2. **Better uncertainty**: Inherits GP's principled uncertainty
3. **Faster convergence**: Prior is closer to target distribution
4. **Classifier-free guidance**: Stronger regime/condition adherence

**Path count considerations:**
- 1,000 paths insufficient for P5/P95 (only ~50 samples in each tail)
- **Recommendation**: 10,000 paths minimum, or use importance sampling for tails
- With FM-GP, generation is fast (single ODE solve, not iterative diffusion)

**Key paper:**
- "Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting" (Kollovieh et al., ICLR 2025)

---

#### Jump Model — Neural Jump SDEs

**What the plan proposes:**
- Poisson/Hawkes-style sampling conditioned on volatility and events
- Heavy-tailed magnitude distribution
- Additive component to trend path

**Advanced Assessment:**

Parametric Hawkes is outdated. **Neural Jump SDEs** learn both timing and magnitude end-to-end:

```python
class NeuralJumpSDE(nn.Module):
    """
    Neural Jump Stochastic Differential Equation.
    Learns: intensity function λ(t), jump size distribution, diffusion.
    """
    def __init__(self, state_dim, cond_dim):
        # Intensity network (when do jumps occur?)
        self.intensity_net = nn.Sequential(
            nn.Linear(state_dim + cond_dim, 128),
            nn.Softplus(),
            nn.Linear(128, 64),
            nn.Softplus(),
            nn.Linear(64, 1),
            nn.Softplus()  # Intensity must be positive
        )
        
        # Jump size network (how big are jumps?)
        self.jump_size_net = nn.Sequential(
            nn.Linear(state_dim + cond_dim, 128),
            nn.GELU(),
            nn.Linear(128, 2)  # (mean, log_std) of jump size
        )
        
        # Diffusion network (continuous dynamics)
        self.diffusion_net = nn.Sequential(
            nn.Linear(state_dim + cond_dim, 128),
            nn.GELU(),
            nn.Linear(128, state_dim * 2)  # (drift, diffusion)
        )
        
    def forward(self, x0, conditioning, horizon, dt=1/24):
        """Simulate jump-diffusion paths."""
        batch_size = x0.shape[0]
        n_steps = int(horizon / dt)
        
        paths = [x0]
        jump_times = []
        jump_sizes = []
        
        x = x0
        for t in range(n_steps):
            cond = torch.cat([x, conditioning], dim=-1)
            
            # Intensity for this step
            intensity = self.intensity_net(cond)
            
            # Sample jump occurrence (thinning algorithm)
            jump_occurred = torch.rand(batch_size) < intensity * dt
            
            # Sample jump size where jumps occurred
            jump_params = self.jump_size_net(cond)
            jump_mean, jump_log_std = jump_params.chunk(2, dim=-1)
            jump = torch.randn_like(jump_mean) * jump_log_std.exp() + jump_mean
            jump = jump * jump_occurred.float().unsqueeze(-1)
            
            # Diffusion dynamics
            drift_diff = self.diffusion_net(cond)
            drift, diff_coef = drift_diff.chunk(2, dim=-1)
            diffusion = torch.randn_like(x) * diff_coef * np.sqrt(dt)
            
            # Update state
            x = x + drift * dt + diffusion + jump
            paths.append(x)
            
            if jump_occurred.any():
                jump_times.append((t, jump_occurred))
                jump_sizes.append(jump[jump_occurred])
        
        return torch.stack(paths, dim=1), jump_times, jump_sizes
```

**Advantages:**
1. **Learned intensity**: No need to specify Hawkes kernel; learns from data
2. **State-dependent jumps**: Jump probability depends on current state + conditioning
3. **Coupled dynamics**: Diffusion and jumps interact naturally
4. **End-to-end training**: Optimized for forecasting loss, not just likelihood

**Training**: Use continuous-time likelihood or simulation-based inference (adversarial/score matching).

**Key papers:**
- "Neural Jump SDEs" (Jia & Benson, NeurIPS 2019)
- "Latent ODEs for Irregularly-Sampled Time Series" (Rubanova et al., NeurIPS 2019)
- "Neural SDEs as Infinite-Dimensional GANs" (Kidger et al., NeurIPS 2021)

---

### 2.4 Calibration & Probabilistic Validity

**What the plan proposes:**
- Regime- and horizon-aware conformal calibration
- Maintain calibration buffers per regime bucket
- Post-processing layer that adjusts quantile bounds
- Does not retrain models

**Advanced Assessment:**

The proposed approach is **significantly behind SOTA**. Current best methods:

| Plan | SOTA Method | Key Advantage |
|------|-------------|---------------|
| Manual regime buffers | **Adaptive Conformal Inference (ACI)** | Learns adaptation rate; handles distribution shift |
| Independent horizons | **Bellman Conformal Inference** | Joint optimization across horizon via dynamic programming |
| Simple quantile adjustment | **Neural Conformal Control (NCC)** | Neural network learns calibration dynamics |
| Single prediction region | **Conformalized Normalizing Flows** | Disjoint regions for multimodal distributions |

**Recommended: Neural Conformal Control (NCC)**

```python
class NeuralConformalControl(nn.Module):
    """
    NCC: Learns to predict calibration adjustments using a neural network.
    Trained with control-inspired loss to guarantee long-term coverage.
    """
    def __init__(self, feature_dim, horizon):
        self.horizon = horizon
        
        # Feature extractor from forecast + context
        self.feature_net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64)
        )
        
        # Quantile predictor per horizon
        self.quantile_nets = nn.ModuleList([
            nn.Linear(64, n_quantiles) for _ in range(horizon)
        ])
        
        # Error integrator (PID-style)
        self.error_integrator = nn.GRUCell(1, 32)
        self.integrator_state = None
        
    def forward(self, base_quantiles, features, target_coverage=0.9):
        """
        Adjust base model quantiles for calibration.
        
        Args:
            base_quantiles: (batch, horizon, n_quantiles) from main model
            features: Context features for adaptation
            target_coverage: Desired coverage level
        
        Returns:
            calibrated_quantiles: Adjusted quantile forecasts
        """
        feat = self.feature_net(features)
        
        adjustments = []
        for h in range(self.horizon):
            adj = self.quantile_nets[h](feat)
            adjustments.append(adj)
        
        adjustments = torch.stack(adjustments, dim=1)
        calibrated = base_quantiles + adjustments
        
        return calibrated
    
    def update(self, coverage_error):
        """Online update based on realized coverage error."""
        # GRU-based error integration
        self.integrator_state = self.error_integrator(
            coverage_error, self.integrator_state
        )
    
    def training_loss(self, predictions, targets, alpha=0.1):
        """
        Control-inspired loss:
        - Coverage loss: Pinball loss for each quantile
        - Sharpness loss: Penalize wide intervals
        - Smoothness loss: Penalize erratic adjustments
        """
        coverage_loss = self.pinball_loss(predictions, targets)
        sharpness_loss = (predictions[..., -1] - predictions[..., 0]).mean()
        smoothness_loss = (predictions[:, 1:] - predictions[:, :-1]).abs().mean()
        
        return coverage_loss + alpha * sharpness_loss + 0.1 * smoothness_loss
```

**For multi-step coherence, add Bellman optimization:**

```python
class BellmanConformalOptimizer:
    """
    Dynamic programming for horizon-consistent calibration.
    Jointly optimizes coverage across all forecast steps.
    """
    def optimize_thresholds(self, base_quantiles, cost_matrix, horizon):
        # Value function V[t] = min expected cost from t to horizon
        V = [None] * (horizon + 1)
        V[horizon] = 0
        
        # Backward pass
        for t in range(horizon - 1, -1, -1):
            V[t] = self.bellman_update(V[t+1], cost_matrix[t])
        
        # Forward pass: extract optimal thresholds
        thresholds = self.extract_policy(V, cost_matrix)
        
        return thresholds
```

**For multimodal distributions, use Conformalized Flows:**

When the predictive distribution is multimodal (e.g., binary event outcome), standard intervals fail. Conformalized Normalizing Flows can produce disjoint prediction regions.

**Key papers:**
- "Adaptive Conformal Inference under Distribution Shift" (Gibbs & Candès, NeurIPS 2021)
- "Conformal PID Control for Time Series Prediction" (Angelopoulos et al., 2024)
- "Neural Conformal Control for Time Series Forecasting" (Rodriguez et al., 2024)
- "Bellman Conformal Inference" (Yang, Candès & Lei, 2024)
- "Conformalized Conditional Normalising Flows" (arXiv, 2024)

---

### 2.5 Explainability & Scenario Engine

#### Explainability — Upgrade to Integrated Attribution

**What the plan proposes:**
- Small surrogate model approximating P50/cone width from feature aggregates
- Permutation importance on surrogate → "Top 3 drivers"

**Advanced Assessment:**

Surrogate + permutation is low-fidelity. Better approaches:

**1. Attention-based Attribution (if using attention anywhere):**
```python
# Extract attention weights from MM engine, trend model
# Aggregate to feature-group level
attribution = aggregate_attention_weights(
    mm_attention_weights,  # Which strikes matter?
    trend_attention_weights,  # Which time lags matter?
    feature_groups=['iv', 'funding', 'onchain', 'macro']
)
```

**2. Integrated Gradients (model-agnostic, faithful):**
```python
def integrated_gradients(model, inputs, baseline, steps=50):
    """
    Integrated gradients: path integral of gradients from baseline to input.
    More faithful than permutation importance.
    """
    scaled_inputs = [baseline + (float(i) / steps) * (inputs - baseline) 
                     for i in range(steps + 1)]
    
    grads = []
    for scaled in scaled_inputs:
        scaled.requires_grad_(True)
        output = model(scaled)
        grad = torch.autograd.grad(output.sum(), scaled)[0]
        grads.append(grad)
    
    avg_grads = torch.stack(grads).mean(dim=0)
    integrated = (inputs - baseline) * avg_grads
    
    return integrated
```

**3. Concept-level explanations:**
Instead of raw features, explain via interpretable concepts:
- "Dealer gamma is net short → increased downside risk"
- "Funding rate spike → mean reversion expected"

**Recommendation**: Use integrated gradients on the actual model, aggregated to interpretable concept groups. Add attention visualization for MM engine.

#### Scenario Engine — Add Consistency Constraints

**What the plan proposes:**
- Whitelist of overridable features (IV, funding, correlations, indices)
- Sanity checks on inputs
- Clearly labeled conditional forecasts

**Advanced Improvement — Learned Consistency:**

```python
class ConsistentScenarioEngine:
    def __init__(self, consistency_model):
        # VAE or flow that models joint distribution of overridable features
        self.consistency_model = consistency_model
        
    def apply_scenario(self, base_features, overrides):
        """
        Apply scenario with learned consistency constraints.
        If user overrides IV, infer consistent skew, funding, etc.
        """
        # Encode base features
        z = self.consistency_model.encode(base_features)
        
        # Apply overrides in latent space
        for feature, value in overrides.items():
            z = self.project_to_constraint(z, feature, value)
        
        # Decode to consistent feature set
        consistent_features = self.consistency_model.decode(z)
        
        # Plausibility score
        plausibility = self.consistency_model.log_prob(consistent_features)
        
        return consistent_features, plausibility
```

**Benefits:**
- Impossible combinations are projected to nearest feasible point
- Plausibility score warns users of extreme scenarios
- Learned from data, not hand-coded rules

---

### 2.6 Infrastructure, MLOps & Monitoring

**What the plan proposes:**
- Dev/staging/prod environments
- Services: Data, Model Inference, Calibration, Monitoring
- Metrics: CRPS, coverage, width, tail miss, latency
- Dashboards and alerts

**Advanced MLOps Requirements:**

| Capability | Plan Status | Required for Advanced System |
|------------|-------------|------------------------------|
| Feature versioning | Missing | **Feature store** (Feast/Tecton) with lineage |
| Model registry | Basic tags | **MLflow/W&B** with full lineage, comparison |
| Experiment tracking | Not mentioned | **Essential** for hyperparameter optimization |
| A/B testing | Missing | **Shadow deployment** for new models |
| Drift detection | Basic | **Multi-dimensional**: input, output, concept drift |
| Retraining automation | Undefined | **Trigger-based**: scheduled + drift-based |

**Advanced Monitoring Stack:**

```yaml
# monitoring_config.yaml
metrics:
  # Forecast quality (existing)
  - crps_per_horizon
  - coverage_per_regime
  - tail_miss_frequency
  
  # Distribution quality (add these)
  - quantile_crossing_rate  # Should be 0
  - path_autocorrelation    # Should match historical
  - conditional_fid         # Generated vs realized paths
  
  # Model health
  - gradient_norm           # Training stability
  - latent_space_coverage   # Generative model mode collapse
  - attention_entropy       # Is model attending to everything or collapsing?
  
  # Data health
  - feature_null_rates
  - distribution_psi        # Population stability index per feature
  - options_surface_quality # IV surface arbitrage score
  
alerts:
  - coverage < 0.85 for 3 consecutive days
  - CRPS degrades > 20% vs 30-day baseline
  - quantile_crossing_rate > 0.01
  - any feature PSI > 0.25
  - options_surface_quality < 0.8
```

**Experiment Tracking Integration:**

```python
# Every training run should log:
wandb.init(project="aetheris-oracle")
wandb.config.update({
    "model_architecture": "fm_gp_residual",
    "mamba_layers": 6,
    "flow_steps": 100,
    "gp_kernel": "spectral_mixture",
    ...
})

# Log metrics during training
wandb.log({
    "train_loss": loss,
    "val_crps": crps,
    "val_coverage": coverage,
    ...
})

# Log model artifacts
wandb.save("checkpoints/best_model.pt")
```

---

### 2.7 Security, Safety, and Governance

**Gaps and Recommendations:**

| Gap | Risk | Solution |
|-----|------|----------|
| Access control undefined | Unauthorized changes | RBAC with roles: viewer, operator, developer, admin |
| No audit logging | Cannot trace issues | Log all: model deployments, calibration changes, scenario queries |
| Model governance absent | Unvalidated models in prod | Change request → peer review → staging test → approval → deploy |
| No model cards | Undocumented limitations | Mandatory model card with: intended use, limitations, failure modes |

**Model Card Template:**

```markdown
# Aetheris Oracle v10.0 - Model Card

## Model Details
- Version: 10.0.3
- Training data: 2020-01-01 to 2024-06-30
- Architecture: FM-GP residual + MambaTS trend + Neural Jump SDE

## Intended Use
- Probabilistic forecasting for BTC, ETH (1-14 day horizon)
- Input to risk management dashboards
- NOT for: live trading execution, regulatory capital calculations

## Limitations
- Performance degrades during regime transitions
- Tail estimates (P5, P95) have higher variance
- Options-derived features unavailable outside market hours

## Known Failure Modes
- Flash crashes: Jump model may underestimate magnitude
- New derivative products: MM engine may not capture novel dynamics
- Black swan events: Calibration may be stale
```

---

## 3. Comparison to State of the Art

### 3.1 Summary Table

| Component | Plan Approach | SOTA Approach | Recommendation |
|-----------|---------------|---------------|----------------|
| Trend backbone | Generic Mamba + AR + Transformer ensemble | MambaTS with VAST, Mamba-2 | Upgrade to MambaTS with variable-aware scanning |
| MM engine | Hand-crafted indices | Differentiable Greeks + learned attention | Implement differentiable options layer |
| Vol engine | Unspecified | Neural rough volatility | Use rough Bergomi with neural parameterization |
| Residual generator | Generic flow matching | FM-GP (flow + GP prior) | Implement FM-GP for principled uncertainty |
| Jump model | Parametric Hawkes | Neural Jump SDEs | End-to-end learned intensity + size |
| Calibration | Regime buffers | NCC / Bellman conformal | Replace entirely with NCC |
| Explainability | Surrogate + permutation | Integrated gradients + attention | Use faithful attribution methods |

### 3.2 Papers to Implement From (Prioritized)

**Tier 1: Core Architecture**

1. **"Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting"** (Kollovieh et al., ICLR 2025)
   - Your residual engine
   - Combines flow matching speed with GP uncertainty

2. **"Neural Conformal Control for Time Series Forecasting"** (Rodriguez et al., 2024)
   - Your calibration engine
   - Neural network learns calibration dynamics with coverage guarantees

3. **"MambaTS: Improved Selective State Space Models for Long-term Time Series Forecasting"** (Cai et al., 2024)
   - Your trend backbone
   - VAST for variable ordering; Temporal Mamba Block

**Tier 2: Specialized Components**

4. **"Neural Jump Stochastic Differential Equations"** (Jia & Benson, NeurIPS 2019)
   - Your jump model
   - End-to-end learned intensity and jump distribution

5. **"Volatility is Rough"** (Gatheral et al., 2018) + **"Deep Learning Volatility"** (Horvath et al., 2021)
   - Your vol engine
   - Rough volatility with neural calibration

6. **"Deep Hedging"** (Bühler et al., 2019)
   - Your MM engine
   - Differentiable Greeks framework

**Tier 3: Robustness & Monitoring**

7. **"Adaptive Conformal Inference under Distribution Shift"** (Gibbs & Candès, 2021)
   - Alternative/complement to NCC
   - Theoretical grounding for non-exchangeable settings

8. **"A Gentle Introduction to Conformal Time Series Forecasting"** (arXiv, Nov 2024)
   - Comprehensive review of calibration methods
   - Helps choose right approach for your setting

---

## 4. Prioritized Recommendations

### 4.1 Must-Upgrade for True SOTA

| # | Issue | Section | Why It Matters | Action |
|---|-------|---------|----------------|--------|
| 1 | **Calibration is 2020-era** | 2.4 | Coverage guarantees will fail during regime changes; current approach ignores temporal dependence | Replace with Neural Conformal Control (NCC) or ACI |
| 2 | **Generic Mamba insufficient** | 2.3 Trend | Ensemble of 3 models is less sophisticated than unified MambaTS | Implement MambaTS with VAST; single model handles all horizons |
| 3 | **Flow matching lacks proper UQ** | 2.3 Residual | Vanilla flow has no principled uncertainty; GP prior fixes this | Upgrade to FM-GP (flow matching with GP priors) |
| 4 | **Jump model is parametric** | 2.3 Jump | Hawkes can't learn complex intensity patterns; misses state-dependent dynamics | Replace with Neural Jump SDEs |
| 5 | **Residual formula bug** | 2.3 | `log(trend + jump)` is mathematically invalid | Fix to multiplicative decomposition or returns |
| 6 | **MM engine hand-crafted** | 2.3 MM | Indices may not capture what matters; not end-to-end | Implement differentiable Greeks with learned attention |

### 4.2 Should-Upgrade for Advanced Implementation

| # | Issue | Section | Why It Matters | Action |
|---|-------|---------|----------------|--------|
| 7 | **Vol engine unspecified** | 2.3 Vol | Missing rough volatility dynamics that are empirically validated | Implement neural rough volatility |
| 8 | **Path count insufficient** | 2.3 Residual | 1,000 paths = noisy tail estimates | Increase to 10,000+ or importance sampling |
| 9 | **No Bellman optimization** | 2.4 Calib | Horizons calibrated independently; misses multi-step consistency | Add Bellman conformal for horizon coherence |
| 10 | **Surrogate explanations unfaithful** | 2.5 | May not reflect actual model behavior | Replace with integrated gradients + attention |
| 11 | **No feature store** | 2.6 | Train/serve skew, versioning issues | Implement Feast or equivalent |
| 12 | **Schema inefficient** | 2.2 | EAV pattern slow for training; can't do differentiable ops | Wide tables + full options surface storage |

### 4.3 Nice-to-Have Enhancements

| # | Issue | Section | Why It Matters | Action |
|---|-------|---------|----------------|--------|
| 13 | **Unified latent architecture** | 2.3 | Avoids sequential error compounding | Consider latent diffusion with structured decoder |
| 14 | **Multimodal prediction regions** | 2.4 | Binary events create bimodal forecasts | Add conformalized normalizing flows |
| 15 | **Scenario consistency learning** | 2.5 | Manual rules miss complex dependencies | Train VAE on feature joint distribution |
| 16 | **Model cards** | 2.7 | Document limitations for governance | Mandatory cards for all deployed models |
| 17 | **Foundation model comparison** | 3.0 | Understand baseline difficulty | Benchmark against Chronos/TimesFM zero-shot |

---

## 5. Unanswered Questions & Clarifications

### Data

1. **Options data source**: Which provider(s) for IV surfaces? Is full strike-by-strike OI available (needed for differentiable Greeks)?

2. **Historical depth**: How far back does synchronized multi-modal data go? Neural Jump SDEs need sufficient jump examples.

3. **Point-in-time capability**: Can you reconstruct exactly what features were available at each historical timestamp? Critical for proper backtesting.

4. **IV surface quality**: Is there an existing arbitrage-free fitting pipeline, or does this need to be built?

### Modeling

5. **Compute budget for training**: What GPU hours are available? FM-GP and Neural Jump SDEs require significant training.

6. **Latency constraints**: Is <1s achievable with 10,000 paths + flow integration + neural SDE? Need to profile.

7. **Cross-asset modeling**: How are BTC/ETH dependencies handled? Joint model or separate with correlation features?

8. **Regime definition**: Is regime discrete (HMM-style) or continuous embedding? This affects calibration design.

### Calibration

9. **NCC training data**: How is the calibration neural network trained? Online only, or pre-trained on historical forecasts?

10. **Coverage target flexibility**: Is 90% fixed, or should system support user-specified coverage levels?

11. **Regime transition protocol**: What happens to calibration state when regime changes? Reset? Interpolate?

### Infrastructure

12. **Experiment tracking**: Is W&B/MLflow already in use? What's the current ML platform?

13. **Retraining frequency**: What's the target for full model retraining? Weekly? Monthly?

14. **Rollback SLA**: How quickly must you be able to revert to previous model if new one fails?

### Product

15. **Downstream latency requirements**: Do consumers need real-time updates or is batch (hourly/daily) sufficient?

16. **Scenario query volume**: How many scenario analyses per day? Affects caching strategy.

17. **Interpretability depth**: Do users need full attention visualizations, or just "top 3 drivers" summary?

18. **Regulatory context**: Is this subject to model risk management requirements (SR 11-7 or equivalent)?

---

## Summary

The Aetheris Oracle v10.0 plan has **strong conceptual foundations** but is **behind state-of-the-art in several critical areas**, particularly calibration and generative modeling.

**Key upgrades for true SOTA:**

1. **Calibration**: Replace ad-hoc regime buffers with Neural Conformal Control
2. **Trend**: Upgrade generic Mamba to MambaTS with variable-aware scanning  
3. **Residuals**: Move from vanilla flow to FM-GP for principled uncertainty
4. **Jumps**: Replace Hawkes with Neural Jump SDEs for learned dynamics
5. **MM Engine**: Implement differentiable Greeks with attention-based aggregation
6. **Vol**: Add neural rough volatility for theoretically grounded vol dynamics

With these upgrades, the system would represent **genuine state-of-the-art** in probabilistic financial forecasting, combining the latest advances in state-space models, generative modeling, conformal prediction, and neural SDEs.

The modular architecture is preserved but each module is upgraded to its most advanced form, with proper integration through shared conditioning and end-to-end training where appropriate.
