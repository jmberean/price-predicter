# Progress Notes (v10)

- Learned components: trend ensemble (AR + gated neural), vol path MLP, jump MLP, residual RNN sampler. Meta-weights adapt online via EWAF. Market-maker indices standardized.
- Artifacts: saved to `artifacts/` (trend_state.json, vol_path_state.json, residual_state.json, jump_state.json). Override paths via `TrainConfig.artifact_root` or module configs.
- Training: run `python -m aetheris_oracle.cli train --horizon 7 --samples 24 --artifact-root artifacts` (defaults to synthetic data). Uses stationarity/regime features, MM indices, and saves weights.
- Evaluation: run `python -m aetheris_oracle.start --mode offline_eval` (or call `run_walk_forward`) to produce CRPS/coverage summaries and feed calibration buffers.
- Forecasting: CLI `python -m aetheris_oracle.cli --asset BTC-USD --horizon 7 --paths 500` (artifacts auto-loaded). Service `python -m aetheris_oracle.server`.
- Notes: residual paths are zero-centered; vol/jump models condition on predicted IV/skew + MM pressure; calibration adjusts quantiles per regime/horizon buckets with coverage tracking.
