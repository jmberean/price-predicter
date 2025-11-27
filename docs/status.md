# Aetheris Oracle v10.0 Status Checklist

## Implemented
- [x] Core forecast pipeline with synthetic/CSV/free connectors and stationarity/regime computation
- [x] Trend ensemble upgraded to learned AR + gated neural path with EWAF meta-weights and artifact persistence
- [x] Vol path engine now MLP-based with regime/MM conditioning and learned weights
- [x] Residual generator and jump model use conditional RNN/MLP samplers with artifact save/load
- [x] Market-maker indices standardized (gamma squeeze, inventory unwind, basis pressure)
- [x] Calibration widening with regime/horizon buckets, coverage tracking, and persistence
- [x] Scenario engine with clamping/whitelist plus explicit conditional labeling
- [x] Explainability (top driver ranking + surrogate) and service metadata (coverage, regime values)
- [x] CLI supports scenarios, calibration load/save, and new `train` subcommand for offline fitting
- [x] FastAPI service with API key auth/CORS, logging metrics sink, and batch forecast helper
- [x] Offline training + walk-forward evaluation scaffolding with CRPS/coverage summaries
- [x] Tests for pipeline, calibration, scenarios, connectors, API/service, metrics, and training smoke

## Remaining (per instructions.md)
- [x] Real-ish data connectors (LocalFeatureStore for OHLCV/IV/funding/basis/narratives) with indexing
- [x] Robust trend/vol/jump/residual stubs (adaptive weights, GARCH-like vol, self-exciting jumps, heavy-tail residuals)
- [x] Regime/horizon-aware calibration buffers with empirical coverage tracking/versioning
- [x] Scenario validation across feature sets and explicit conditional labeling in service responses
- [x] Surrogate-based explainability (permutation/sensitivity) tied to cone width
- [ ] Harden monitoring/metrics backend and service auth/logging for production
- [ ] Deployment configs/scripts and production batch inference orchestration
