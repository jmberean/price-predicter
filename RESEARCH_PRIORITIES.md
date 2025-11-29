# Aetheris Oracle: Research Priorities

Based on analysis of the [Research Roadmap](../Aetheris%20Oracle_%20Research%20Roadmap.md), this document ranks potential improvements by impact to forecast quality and practical utility.

## Current Architecture Baseline

| Roadmap Concept | Current Module | Implementation Status |
|-----------------|----------------|----------------------|
| Flow Matching | `fm_gp_residual.py` | Partial - FM-GP, not TempO/CGFM |
| Neural Jump-Diffusion | `neural_jump_sde.py` | Partial - fixed intensity, not state-dependent |
| Conformal Prediction | `neural_conformal_control.py` | Partial - univariate, not CopulaCPTS |
| Neural SDEs | `neural_rough_vol.py` | Partial - rough vol, not Lévy dynamics |
| Path Signatures | Not implemented | - |
| Graph Neural Networks | Not implemented | - |
| Differentiable Optimization | Not implemented | - |
| JAX/Diffrax Backend | Not implemented | Currently PyTorch |

---

## Tier 1: Transformational Impact

### 1. CopulaCPTS (Copula Conformal Prediction)
**Priority: CRITICAL**

| Aspect | Details |
|--------|---------|
| **Problem Solved** | Current uncertainty is per-timestep independent intervals |
| **Benefit** | Joint path uncertainty - answers "will price hit X within 30 days" |
| **Technical Basis** | Uses copulas to model dependence between future time steps |
| **Key Paper** | [ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/8707924df5e207fa496f729f49069446-Paper-Conference.pdf) |
| **Impact** | Directly fixes overconfidence in multi-step forecasts |

### 2. State-Dependent Jump Intensity λ(Xt)
**Priority: CRITICAL**

| Aspect | Details |
|--------|---------|
| **Problem Solved** | Neural MJD has fixed jump probability regardless of market state |
| **Benefit** | "High volatility → higher crash probability" is learned endogenously |
| **Technical Basis** | Jump intensity becomes a function of regime vector: λ(regime_vol, liquidity, ...) |
| **Key Paper** | [Neural Lévy SDE](https://arxiv.org/html/2509.01041v1) |
| **Impact** | Transforms jump model into a "crash riskometer" |

### 3. Path Signatures as Features
**Priority: CRITICAL**

| Aspect | Details |
|--------|---------|
| **Problem Solved** | OHLC bars lose geometric information about price path |
| **Benefit** | Encodes path *shape* (loops, reversals, momentum) mathematically |
| **Technical Basis** | Rough Path Theory - iterated integrals invariant to time parameterization |
| **Key Paper** | [Signature-Informed Transformer](https://arxiv.org/html/2510.03129v1) |
| **Impact** | Superior conditioning features with minimal architecture change |
| **Implementation** | Use `signatory` library for PyTorch |

**Signature Feature Encoding:**
- Level 1: Net displacement (return)
- Level 2: Signed area (lead-lag, volatility)
- Level 3+: Skewness, kurtosis, higher-order geometry

---

## Tier 2: High Impact

### 4. CGFM (Conditional Guided Flow Matching)
**Priority: HIGH**

| Aspect | Details |
|--------|---------|
| **Problem Solved** | FM-GP generates from scratch, ignoring existing model predictions |
| **Benefit** | Uses legacy model predictions as "guidance" for flow correction |
| **Technical Basis** | Two-sided conditional probability paths interpolate between base and target |
| **Key Paper** | [CGFM](https://arxiv.org/html/2507.07192v1) |
| **Impact** | Upgrades FM-GP to a "corrective" system without discarding prior work |

### 5. Consistency Distillation
**Priority: HIGH**

| Aspect | Details |
|--------|---------|
| **Problem Solved** | ODE integration requires multiple function evaluations |
| **Benefit** | 4-100x faster inference via single-step generation |
| **Technical Basis** | Distill multi-step solver into direct prior→solution mapping |
| **Key Paper** | [Flow Map Learning](https://www.youtube.com/watch?v=ijUly7q0vfo) |
| **Impact** | Enables real-time Monte Carlo (thousands of paths in milliseconds) |

### 6. CPTC (Adaptive Conformal Prediction)
**Priority: HIGH**

| Aspect | Details |
|--------|---------|
| **Problem Solved** | Confidence intervals calibrated in one regime fail in another |
| **Benefit** | Auto-expands/contracts uncertainty when regime changes detected |
| **Technical Basis** | Change point detection + Slowly Varying Dynamical Systems |
| **Key Paper** | [CPTC](https://arxiv.org/html/2509.02844v1) |
| **Impact** | Asymptotic valid coverage without assuming stationarity |

---

## Tier 3: Significant Impact

### 7. CoRel (Relational Conformal Prediction)
**Priority: MEDIUM**

| Aspect | Details |
|--------|---------|
| **Problem Solved** | Uncertainty for correlated assets computed independently |
| **Benefit** | Cross-asset uncertainty correlation ("BTC uncertain → ETH bounds expand") |
| **Technical Basis** | Learns graph topology from residuals, adjusts bounds relationally |
| **Key Paper** | [CoRel](https://arxiv.org/html/2502.09443v1) |
| **Impact** | System-wide coherent risk assessment |

### 8. Dynamic Graph Neural Networks (MDGNN)
**Priority: MEDIUM**

| Aspect | Details |
|--------|---------|
| **Problem Solved** | Static correlation matrix ignores evolving asset relationships |
| **Benefit** | Time-varying network topology captures decoupling and spillovers |
| **Technical Basis** | Transformer-encoded multiplex relation evolution |
| **Key Paper** | [MDGNN](https://arxiv.org/pdf/2402.06633) |
| **Crypto-Native Edges** | Statistical correlation, on-chain flows, volatility spillovers |

### 9. Neural Lévy SDE
**Priority: MEDIUM**

| Aspect | Details |
|--------|---------|
| **Problem Solved** | Rough vol model limited to Gaussian increments |
| **Benefit** | Heavier tails, more flexible jump size distributions |
| **Technical Basis** | Lévy processes with state-dependent dynamics |
| **Key Paper** | [Neural Lévy SDE](https://arxiv.org/html/2509.01041v1) |

### 10. TempO Architecture
**Priority: MEDIUM**

| Aspect | Details |
|--------|---------|
| **Problem Solved** | Standard models suffer from spectral bias (smooth high-frequency signals) |
| **Benefit** | Multi-scale Fourier layers capture fractal volatility across timeframes |
| **Technical Basis** | Latent flow matching with time-conditioned Fourier layers |
| **Key Paper** | [TempO](https://arxiv.org/abs/2510.15101) |

---

## Tier 4: Strategic (Long-term)

### 11. End-to-End Differentiable Optimization
**Priority: LOW (unless trading)**

| Aspect | Details |
|--------|---------|
| **Problem Solved** | Model minimizes MSE, not financial utility |
| **Benefit** | Train to maximize Sharpe ratio / minimize CVaR directly |
| **Technical Basis** | Backprop through convex optimization via KKT conditions |
| **Key Paper** | [E2E Portfolio Opt](https://arxiv.org/abs/2507.01918) |
| **Prerequisite** | Only matters if actually trading on forecasts |

### 12. JAX/Diffrax Migration
**Priority: LOW (unless scale needed)**

| Aspect | Details |
|--------|---------|
| **Problem Solved** | PyTorch SDE solvers not GPU-optimized |
| **Benefit** | 10x compute efficiency, reversible solvers, kernel fusion |
| **Technical Basis** | JAX XLA compilation + Diffrax differential equation library |
| **Key Paper** | [Diffrax](https://arxiv.org/html/2510.25769v1) |
| **Effort** | 6+ month rewrite of entire stack |

### 13. Deep Hedging with IV Surface
**Priority: LOW (options-specific)**

| Aspect | Details |
|--------|---------|
| **Problem Solved** | Delta hedging suboptimal with transaction costs |
| **Benefit** | RL agent learns optimal hedge policy with friction |
| **Technical Basis** | TD3/PPO with IV surface as state input |
| **Key Paper** | [Deep Hedging](https://arxiv.org/html/2504.06208v3) |
| **Prerequisite** | Only relevant for options trading desk |

### 14. JAX-LOB Simulator
**Priority: LOW (research infrastructure)**

| Aspect | Details |
|--------|---------|
| **Problem Solved** | CPU-based LOB simulation too slow for RL |
| **Benefit** | Thousands of parallel trading environments on GPU |
| **Key Paper** | [JAX-LOB](https://arxiv.org/pdf/2308.13289) |
| **Use Case** | RL training at scale, not forecasting improvement |

---

## Implementation Roadmap

### Phase 1: Quick Wins (1-2 months)
1. **Path Signatures** - Add `signatory` library, compute signatures as conditioning
2. **State-Dependent λ** - Modify `neural_jump_sde.py` to take regime as input to intensity network

### Phase 2: Uncertainty Overhaul (2-4 months)
3. **CopulaCPTS** - Implement copula-based joint path uncertainty
4. **CPTC** - Add regime-adaptive conformal wrapper

### Phase 3: Architecture Upgrades (4-6 months)
5. **CGFM** - Upgrade FM-GP to use legacy model guidance
6. **Consistency Distillation** - Distill for fast inference

### Phase 4: Multi-Asset (6-9 months)
7. **CoRel** - Cross-asset uncertainty correlation
8. **MDGNN** - Dynamic graph for asset relationships

### Phase 5: Strategic (9+ months)
9. **Differentiable Optimization** - If trading use case confirmed
10. **JAX Migration** - If scale requirements emerge

---

## Summary

**Top 3 Highest Impact:**
1. **CopulaCPTS** → Fixes uncertainty quantification weakness
2. **State-Dependent λ(Xt)** → Transforms jump model into crash predictor
3. **Path Signatures** → Superior features with minimal architecture change

These three improvements would meaningfully enhance forecast quality without requiring architectural rewrites.
