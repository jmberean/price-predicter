"""
Training data preparation for SOTA components.

Fetches real historical data and prepares it for training:
- NCC: forecast-outcome pairs with calibration errors
- Neural Jump SDE: price paths with detected jump events
- Diff Greeks: IV term structures with market maker state
- FM-GP: residual paths conditioned on regime
"""

import os
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import numpy as np
from pathlib import Path

from ..config import ForecastConfig
from ..data.interfaces import DataConnector
from ..data.free_connectors import FreeDataConnector
from ..features.regime import compute_regime_vector
from ..features.stationarity import StationarityNormalizer
from ..pipeline.forecast import ForecastEngine

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not available, use system env vars only


# Training data configuration from environment
def get_training_config() -> Dict[str, int]:
    """
    Load training data configuration from environment variables.

    IMPORTANT: holdout_days prevents temporal overfitting by excluding
    the most recent N days from training. This ensures the backtest period
    is completely unseen during training.
    """
    return {
        "lookback_days": int(os.getenv("TRAINING_LOOKBACK_DAYS", "180")),
        "window_days": int(os.getenv("TRAINING_WINDOW_DAYS", "90")),
        "holdout_days": int(os.getenv("TRAINING_HOLDOUT_DAYS", "60")),  # Exclude last 60 days
        "samples_ncc": int(os.getenv("TRAINING_SAMPLES_NCC", "300")),
        "samples_fmgp": int(os.getenv("TRAINING_SAMPLES_FMGP", "300")),
        "samples_neural_jump": int(os.getenv("TRAINING_SAMPLES_NEURAL_JUMP", "300")),
        "samples_diff_greeks": int(os.getenv("TRAINING_SAMPLES_DIFF_GREEKS", "200")),
        "samples_neural_vol": int(os.getenv("TRAINING_SAMPLES_NEURAL_VOL", "300")),
        "samples_mamba": int(os.getenv("TRAINING_SAMPLES_MAMBA", "300")),
    }


def fetch_historical_prices(
    connector: DataConnector,
    asset_id: str,
    start_date: datetime,
    end_date: datetime,
    window_days: int = 90,
) -> List[Tuple[datetime, List[float], Dict]]:
    """
    Fetch historical price data with features.

    Returns:
        List of (date, closes, features) tuples
    """
    dates = []
    current = start_date

    while current <= end_date:
        try:
            frame = connector.fetch_window(
                asset_id=asset_id,
                as_of=current,
                window=timedelta(days=window_days),
            )

            if len(frame.closes) >= window_days // 2:  # At least half the window
                features = {
                    "iv_points": frame.iv_points,
                    "funding_rate": frame.funding_rate,
                    "basis": frame.basis,
                    "order_imbalance": frame.order_imbalance,
                    "skew": frame.skew,
                    "narrative_scores": frame.narrative_scores,
                }
                dates.append((current, frame.closes, features))
        except Exception as e:
            print(f"Warning: Failed to fetch data for {current}: {e}")

        current += timedelta(days=1)

    return dates


def prepare_ncc_training_data(
    connector: DataConnector,
    asset_id: str,
    n_samples: int = None,
    horizons: List[int] = [1, 3, 7, 14],
) -> Tuple[List, List, List, List]:
    """
    Prepare training data for Neural Conformal Control.

    Generates historical forecasts and compares to actual outcomes.

    Returns:
        (base_quantiles_list, actuals_list, features_list, horizon_indices_list)
    """
    config = get_training_config()
    n_samples = n_samples or config["samples_ncc"]
    lookback_days = config["lookback_days"]
    window_days = config["window_days"]

    print(f"Preparing NCC training data: {n_samples} samples from {lookback_days} days of history...")

    base_quantiles_list = []
    actuals_list = []
    features_list = []
    horizon_indices_list = []

    # Use SOTA forecast engine to generate base quantiles (without NCC calibration)
    # This trains NCC to calibrate SOTA forecasts, not legacy
    # IMPORTANT: Must enable ALL SOTA components to match Full SOTA inference distribution
    engine = ForecastEngine(
        connector=connector,
        seed=42,
        use_ncc_calibration=False,  # Don't use NCC (we're training it!)
        use_diff_greeks=True,
        use_fm_gp_residuals=True,
        use_neural_rough_vol=True,
        use_neural_jumps=True,  # Must be True to match Full SOTA configuration
        use_mamba_trend=True,
        diff_greeks_artifact_path="artifacts/diff_greeks.pt",
        fmgp_artifact_path="artifacts/fmgp_residuals.pt",
        neural_vol_artifact_path="artifacts/neural_rough_vol.pt",
        neural_jump_artifact_path="artifacts/neural_jump_sde.pt",
        mamba_artifact_path="artifacts/mamba_trend.pt",
    )

    # Fetch historical data (exclude holdout period to prevent temporal overfitting)
    holdout_days = config["holdout_days"]
    end_date = datetime.utcnow() - timedelta(days=holdout_days)
    start_date = end_date - timedelta(days=lookback_days)

    print(f"Training period: {start_date.date()} to {end_date.date()} (excluding last {holdout_days} days)")

    historical_data = fetch_historical_prices(
        connector, asset_id, start_date, end_date, window_days=window_days
    )

    if len(historical_data) < 30:
        print(f"Warning: Only {len(historical_data)} days of data, using synthetic fallback")
        return _prepare_ncc_synthetic_fallback(n_samples, horizons)

    # Sample random dates
    sample_indices = np.random.choice(
        len(historical_data) - 14, size=min(n_samples, len(historical_data) - 14), replace=False
    )

    for idx in sample_indices:
        forecast_date, closes, features = historical_data[idx]

        # Pick random horizon
        horizon = np.random.choice(horizons)
        horizon_idx = horizons.index(horizon)

        # Generate forecast
        try:
            config = ForecastConfig(
                asset_id=asset_id,
                horizon_days=horizon,
                as_of=forecast_date,
                num_paths=1000,
                seed=42,
            )
            result = engine.forecast(config)

            # Extract base quantiles (before calibration)
            base_quantiles = []
            for q in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]:
                base_quantiles.append(result.quantile_paths[horizon][q])

            # Get actual outcome
            if idx + horizon < len(historical_data):
                actual = historical_data[idx + horizon][1][-1]  # Last close
            else:
                continue

            # Extract regime features
            regime = result.regime
            if regime and len(regime.values) >= 2:
                feature_vec = list(regime.values[:10])  # First 10 regime features
                feature_vec = feature_vec + [0.0] * (10 - len(feature_vec))  # Pad to 10
            else:
                feature_vec = [0.0] * 10

            base_quantiles_list.append(np.array(base_quantiles))
            actuals_list.append(actual)
            features_list.append(np.array(feature_vec))
            horizon_indices_list.append(horizon_idx)

        except Exception as e:
            print(f"Warning: Failed to generate forecast for {forecast_date}: {e}")
            continue

    print(f"Generated {len(base_quantiles_list)} NCC training samples")

    if len(base_quantiles_list) < 50:
        print("Warning: Too few samples, supplementing with synthetic data")
        return _prepare_ncc_synthetic_fallback(n_samples, horizons)

    return base_quantiles_list, actuals_list, features_list, horizon_indices_list


def _prepare_ncc_synthetic_fallback(
    n_samples: int, horizons: List[int]
) -> Tuple[List, List, List, List]:
    """Fallback to synthetic data if real data insufficient."""
    print("Using synthetic NCC training data...")

    base_quantiles_list = []
    actuals_list = []
    features_list = []
    horizon_indices_list = []

    for _ in range(n_samples):
        horizon_idx = np.random.choice(len(horizons))

        base_median = np.random.uniform(40000, 60000)
        base_spread = np.random.uniform(1000, 5000)

        base_quantiles = np.array([
            base_median - 2 * base_spread,
            base_median - 1.5 * base_spread,
            base_median - 0.8 * base_spread,
            base_median,
            base_median + 0.8 * base_spread,
            base_median + 1.5 * base_spread,
            base_median + 2 * base_spread,
        ])

        actual = np.random.normal(base_median, base_spread * 0.7)
        features = np.random.randn(10)

        base_quantiles_list.append(base_quantiles)
        actuals_list.append(actual)
        features_list.append(features)
        horizon_indices_list.append(horizon_idx)

    return base_quantiles_list, actuals_list, features_list, horizon_indices_list


def prepare_jump_sde_training_data(
    connector: DataConnector,
    asset_id: str,
    n_samples: int = None,
    horizon: int = 14,
) -> Tuple[List, List, List]:
    """
    Prepare training data for Neural Jump SDE.

    Extracts price paths and detects jump events.

    Returns:
        (x0_list, conditioning_list, target_paths_list)
    """
    config = get_training_config()
    n_samples = n_samples or config["samples_neural_jump"]
    lookback_days = config["lookback_days"]
    window_days = config["window_days"]

    print(f"Preparing Neural Jump SDE training data: {n_samples} samples from {lookback_days} days...")

    x0_list = []
    conditioning_list = []
    target_paths_list = []

    # Fetch historical data (exclude holdout period to prevent temporal overfitting)
    holdout_days = config["holdout_days"]
    end_date = datetime.utcnow() - timedelta(days=holdout_days)
    start_date = end_date - timedelta(days=lookback_days)

    print(f"Training period: {start_date.date()} to {end_date.date()} (excluding last {holdout_days} days)")

    historical_data = fetch_historical_prices(
        connector, asset_id, start_date, end_date, window_days=window_days
    )

    if len(historical_data) < 30:
        print("Warning: Insufficient data, using synthetic fallback")
        return _prepare_jump_sde_synthetic_fallback(n_samples, horizon)

    normalizer = StationarityNormalizer()

    # Sample random windows
    sample_indices = np.random.choice(
        len(historical_data) - horizon - 1, size=min(n_samples, len(historical_data) - horizon - 1), replace=False
    )

    for idx in sample_indices:
        try:
            _, closes, features = historical_data[idx]

            # Normalize
            normalized, stats = normalizer.normalize_and_stats(closes[-horizon-1:])

            if len(normalized) < horizon + 1:
                continue

            x0 = normalized[0]
            path = [normalized[i] - normalized[0] for i in range(1, min(horizon + 1, len(normalized)))]

            if len(path) < horizon:
                continue

            # Compute regime features
            regime = compute_regime_vector(
                closes=closes,
                iv_points=features["iv_points"],
                funding_rate=features["funding_rate"],
                basis=features["basis"],
                order_imbalance=features["order_imbalance"],
                narrative_scores=features["narrative_scores"],
                skew=features["skew"],
            )

            regime_vol = regime.values[0] if regime.values else 0.0
            iv_level = regime.values[1] if len(regime.values) > 1 else 0.0

            conditioning = [
                regime_vol,
                iv_level,
                features["funding_rate"],
                features["basis"],
                features["order_imbalance"],
                sum(features["narrative_scores"].values()) / max(len(features["narrative_scores"]), 1),
            ]

            x0_list.append(x0)
            conditioning_list.append(conditioning)
            target_paths_list.append(path)

        except Exception as e:
            print(f"Warning: Failed to process sample {idx}: {e}")
            continue

    print(f"Generated {len(x0_list)} Neural Jump SDE training samples")

    if len(x0_list) < 50:
        print("Warning: Too few samples, using synthetic fallback")
        return _prepare_jump_sde_synthetic_fallback(n_samples, horizon)

    return x0_list, conditioning_list, target_paths_list


def _prepare_jump_sde_synthetic_fallback(
    n_samples: int, horizon: int
) -> Tuple[List, List, List]:
    """Fallback to synthetic data if real data insufficient."""
    print("Using synthetic Neural Jump SDE training data...")

    x0_list = []
    conditioning_list = []
    target_paths_list = []

    for _ in range(n_samples):
        x0 = 0.0
        conditioning = np.random.randn(6).tolist()

        path = [0.0]
        for t in range(1, horizon):
            diffusion = np.random.randn() * 0.001
            jump = np.random.randn() * 0.01 if np.random.rand() < 0.1 else 0.0
            path.append(path[-1] + diffusion + jump)

        x0_list.append(x0)
        conditioning_list.append(conditioning)
        target_paths_list.append(path)

    return x0_list, conditioning_list, target_paths_list


def prepare_diff_greeks_training_data(
    connector: DataConnector,
    asset_id: str,
    n_samples: int = None,
) -> Tuple[List, List, List]:
    """
    Prepare training data for Differentiable Greeks.

    Extracts IV term structures and market maker state.

    Returns:
        (spot_list, iv_surfaces_list, target_mm_state_list)
    """
    config = get_training_config()
    n_samples = n_samples or config["samples_diff_greeks"]
    lookback_days = config["lookback_days"]
    window_days = min(config["window_days"], 30)  # Diff Greeks needs shorter window

    print(f"Preparing Diff Greeks training data: {n_samples} samples from {lookback_days} days...")

    spot_list = []
    iv_surfaces_list = []
    target_mm_state_list = []

    # Fetch historical data (exclude holdout period to prevent temporal overfitting)
    holdout_days = config["holdout_days"]
    end_date = datetime.utcnow() - timedelta(days=holdout_days)
    start_date = end_date - timedelta(days=lookback_days)

    print(f"Training period: {start_date.date()} to {end_date.date()} (excluding last {holdout_days} days)")

    historical_data = fetch_historical_prices(
        connector, asset_id, start_date, end_date, window_days=window_days
    )

    if len(historical_data) < 10:
        print("Warning: Insufficient data, using synthetic fallback")
        return _prepare_diff_greeks_synthetic_fallback(n_samples)

    # Sample random dates
    sample_indices = np.random.choice(
        len(historical_data), size=min(n_samples, len(historical_data)), replace=False
    )

    for idx in sample_indices:
        try:
            _, closes, features = historical_data[idx]

            spot = closes[-1]
            iv_surface = features["iv_points"]  # Dict of {tenor_days: iv}

            # Compute target MM state from features
            gamma_squeeze = features["funding_rate"] * 100  # Scale funding
            inventory_unwind = features["basis"] * 50  # Scale basis
            basis_pressure = features["order_imbalance"] * 10  # Scale imbalance

            target_mm_state = [gamma_squeeze, inventory_unwind, basis_pressure]

            spot_list.append(spot)
            iv_surfaces_list.append(iv_surface)
            target_mm_state_list.append(target_mm_state)

        except Exception as e:
            print(f"Warning: Failed to process sample {idx}: {e}")
            continue

    print(f"Generated {len(spot_list)} Diff Greeks training samples")

    if len(spot_list) < 20:
        print("Warning: Too few samples, using synthetic fallback")
        return _prepare_diff_greeks_synthetic_fallback(n_samples)

    return spot_list, iv_surfaces_list, target_mm_state_list


def _prepare_diff_greeks_synthetic_fallback(n_samples: int) -> Tuple[List, List, List]:
    """Fallback to synthetic data if real data insufficient."""
    print("Using synthetic Diff Greeks training data...")

    spot_list = []
    iv_surfaces_list = []
    target_mm_state_list = []

    for _ in range(n_samples):
        spot = np.random.uniform(40000, 60000)

        iv_surface = {
            7: np.random.uniform(0.4, 0.8),
            14: np.random.uniform(0.45, 0.85),
            30: np.random.uniform(0.5, 0.9),
            60: np.random.uniform(0.55, 0.95),
        }

        target_mm_state = np.random.randn(3).tolist()

        spot_list.append(spot)
        iv_surfaces_list.append(iv_surface)
        target_mm_state_list.append(target_mm_state)

    return spot_list, iv_surfaces_list, target_mm_state_list


def prepare_fmgp_residual_training_data(
    connector: DataConnector,
    asset_id: str,
    n_samples: int = None,
    horizon: int = 14,
) -> Tuple[List, List]:
    """
    Prepare training data for FM-GP Residual Generator.

    Extracts residual paths after removing trend.

    Returns:
        (conditioning_list, residual_paths_list)
    """
    config = get_training_config()
    n_samples = n_samples or config["samples_fmgp"]
    lookback_days = config["lookback_days"]
    window_days = config["window_days"]

    print(f"Preparing FM-GP Residual training data: {n_samples} samples from {lookback_days} days...")

    conditioning_list = []
    residual_paths_list = []

    # Fetch historical data (exclude holdout period to prevent temporal overfitting)
    holdout_days = config["holdout_days"]
    end_date = datetime.utcnow() - timedelta(days=holdout_days)
    start_date = end_date - timedelta(days=lookback_days)

    print(f"Training period: {start_date.date()} to {end_date.date()} (excluding last {holdout_days} days)")

    historical_data = fetch_historical_prices(
        connector, asset_id, start_date, end_date, window_days=window_days
    )

    if len(historical_data) < 30:
        print("Warning: Insufficient data for FM-GP, using synthetic fallback")
        return _prepare_fmgp_synthetic_fallback(n_samples, horizon)

    normalizer = StationarityNormalizer()

    # Sample random windows
    sample_indices = np.random.choice(
        len(historical_data) - horizon - 1, size=min(n_samples, len(historical_data) - horizon - 1), replace=False
    )

    for idx in sample_indices:
        try:
            _, closes, features = historical_data[idx]

            # Normalize prices
            normalized, stats = normalizer.normalize_and_stats(closes[-horizon-5:])

            if len(normalized) < horizon + 1:
                continue

            # Extract residuals (detrended returns)
            returns = np.diff(normalized)[-horizon:]

            # Remove linear trend to get residuals
            trend = np.linspace(returns[0], returns[-1], len(returns))
            residuals = returns - trend

            # Ensure zero mean
            residuals -= residuals.mean()

            if len(residuals) < horizon:
                continue

            # Compute regime features
            regime = compute_regime_vector(
                closes=closes,
                iv_points=features["iv_points"],
                funding_rate=features["funding_rate"],
                basis=features["basis"],
                order_imbalance=features["order_imbalance"],
                narrative_scores=features["narrative_scores"],
                skew=features["skew"],
            )

            regime_vol = regime.values[0] if regime.values else 0.0
            iv_level = regime.values[1] if len(regime.values) > 1 else 0.0

            conditioning = [
                regime_vol,
                iv_level,
                features["funding_rate"],
                features["basis"],
                features["order_imbalance"],
            ]

            conditioning_list.append(conditioning)
            residual_paths_list.append(residuals.tolist())

        except Exception as e:
            print(f"Warning: Failed to process FM-GP sample {idx}: {e}")
            continue

    print(f"Generated {len(residual_paths_list)} FM-GP training samples")

    if len(residual_paths_list) < 20:
        print("Warning: Too few FM-GP samples, using synthetic fallback")
        return _prepare_fmgp_synthetic_fallback(n_samples, horizon)

    return conditioning_list, residual_paths_list


def _prepare_fmgp_synthetic_fallback(n_samples: int, horizon: int) -> Tuple[List, List]:
    """Fallback to synthetic data for FM-GP if real data insufficient."""
    print("Using synthetic FM-GP training data...")

    conditioning_list = []
    residual_paths_list = []

    for _ in range(n_samples):
        conditioning = np.random.randn(5).tolist()

        vol_scale = np.random.uniform(0.01, 0.05)
        residuals = np.random.randn(horizon) * vol_scale
        residuals -= residuals.mean()

        conditioning_list.append(conditioning)
        residual_paths_list.append(residuals.tolist())

    return conditioning_list, residual_paths_list


def prepare_neural_vol_training_data(
    connector: DataConnector,
    asset_id: str,
    n_samples: int = None,
    horizon: int = 14,
) -> Tuple[List, List, List]:
    """
    Prepare training data for Neural Rough Volatility.

    Extracts volatility paths from historical data.

    Returns:
        (past_vols, conditioning_list, target_vol_paths)
    """
    config = get_training_config()
    n_samples = n_samples or config["samples_neural_vol"]
    lookback_days = config["lookback_days"]
    window_days = config["window_days"]

    print(f"Preparing Neural Vol training data: {n_samples} samples from {lookback_days} days...")

    past_vols = []
    conditioning_list = []
    target_vol_paths = []

    # Fetch historical data (exclude holdout period to prevent temporal overfitting)
    holdout_days = config["holdout_days"]
    end_date = datetime.utcnow() - timedelta(days=holdout_days)
    start_date = end_date - timedelta(days=lookback_days)

    print(f"Training period: {start_date.date()} to {end_date.date()} (excluding last {holdout_days} days)")

    historical_data = fetch_historical_prices(
        connector, asset_id, start_date, end_date, window_days=window_days
    )

    if len(historical_data) < 30:
        print("Warning: Insufficient data for Neural Vol, using synthetic fallback")
        return _prepare_neural_vol_synthetic_fallback(n_samples, horizon)

    # Sample random windows
    sample_indices = np.random.choice(
        len(historical_data) - horizon - 1, size=min(n_samples, len(historical_data) - horizon - 1), replace=False
    )

    for idx in sample_indices:
        try:
            _, closes, features = historical_data[idx]

            # Compute realized volatility from returns
            returns = np.diff(closes) / closes[:-1]
            past_vol = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)

            # Future volatility path
            vol_path = []
            for i in range(idx, min(idx + horizon, len(historical_data))):
                _, future_closes, _ = historical_data[i]
                future_returns = np.diff(future_closes) / future_closes[:-1]
                future_vol = np.std(future_returns[-5:]) if len(future_returns) >= 5 else np.std(future_returns)
                vol_path.append(float(future_vol))

            if len(vol_path) < horizon:
                continue

            # Compute regime features
            regime = compute_regime_vector(
                closes=closes,
                iv_points=features["iv_points"],
                funding_rate=features["funding_rate"],
                basis=features["basis"],
                order_imbalance=features["order_imbalance"],
                narrative_scores=features["narrative_scores"],
                skew=features["skew"],
            )

            # Extract IV features
            iv_values = list(features["iv_points"].values())
            base_iv = iv_values[0] if iv_values else 0.5
            iv_30 = iv_values[1] if len(iv_values) > 1 else base_iv

            conditioning = [
                base_iv,
                iv_30,
                features["skew"],
                regime.values[0] if regime.values else 0.0,
                features["funding_rate"],
                features["basis"],
                features["order_imbalance"],
                sum(features["narrative_scores"].values()) / max(len(features["narrative_scores"]), 1),
            ]

            # Pad to 10 features
            conditioning = conditioning + [0.0] * (10 - len(conditioning))
            conditioning = conditioning[:10]

            past_vols.append(float(past_vol))
            conditioning_list.append(conditioning)
            target_vol_paths.append(vol_path)

        except Exception as e:
            print(f"Warning: Failed to process Neural Vol sample {idx}: {e}")
            continue

    print(f"Generated {len(target_vol_paths)} Neural Vol training samples")

    if len(target_vol_paths) < 20:
        print("Warning: Too few Neural Vol samples, using synthetic fallback")
        return _prepare_neural_vol_synthetic_fallback(n_samples, horizon)

    return past_vols, conditioning_list, target_vol_paths


def _prepare_neural_vol_synthetic_fallback(n_samples: int, horizon: int) -> Tuple[List, List, List]:
    """Fallback to synthetic data for Neural Vol if real data insufficient."""
    print("Using synthetic Neural Vol training data...")

    past_vols = []
    conditioning_list = []
    target_vol_paths = []

    for _ in range(n_samples):
        past_vol = np.random.uniform(0.3, 0.8)
        conditioning = np.random.randn(10).tolist()

        vol_path = [past_vol]
        for t in range(1, horizon):
            mean_reversion = 0.1 * (0.5 - vol_path[-1])
            rough_increment = np.random.randn() * 0.02
            vol_path.append(max(0.1, vol_path[-1] + mean_reversion + rough_increment))

        past_vols.append(past_vol)
        conditioning_list.append(conditioning)
        target_vol_paths.append(vol_path)

    return past_vols, conditioning_list, target_vol_paths


def prepare_mamba_trend_training_data(
    connector: DataConnector,
    asset_id: str,
    n_samples: int = None,
    lookback: int = 20,
    horizon: int = 14,
) -> Tuple[List, List]:
    """
    Prepare training data for MambaTS Trend.

    Extracts price sequences and future trends.

    Returns:
        (input_sequences, target_sequences)
    """
    config = get_training_config()
    n_samples = n_samples or config["samples_mamba"]
    lookback_days = config["lookback_days"]
    window_days = config["window_days"]

    print(f"Preparing MambaTS training data: {n_samples} samples from {lookback_days} days...")

    input_sequences = []
    target_sequences = []

    # Fetch historical data (exclude holdout period to prevent temporal overfitting)
    holdout_days = config["holdout_days"]
    end_date = datetime.utcnow() - timedelta(days=holdout_days)
    start_date = end_date - timedelta(days=lookback_days)

    print(f"Training period: {start_date.date()} to {end_date.date()} (excluding last {holdout_days} days)")

    historical_data = fetch_historical_prices(
        connector, asset_id, start_date, end_date, window_days=window_days
    )

    if len(historical_data) < 30:
        print("Warning: Insufficient data for MambaTS, using synthetic fallback")
        return _prepare_mamba_synthetic_fallback(n_samples, lookback, horizon)

    normalizer = StationarityNormalizer()

    # Sample random windows
    sample_indices = np.random.choice(
        len(historical_data) - horizon - 1, size=min(n_samples, len(historical_data) - horizon - 1), replace=False
    )

    for idx in sample_indices:
        try:
            _, closes, features = historical_data[idx]

            # Need enough history
            if len(closes) < lookback + horizon:
                continue

            # Normalize
            normalized, stats = normalizer.normalize_and_stats(closes[-(lookback + horizon):])

            if len(normalized) < lookback + horizon:
                continue

            # Past returns
            past_prices = normalized[:lookback]
            past_returns = np.diff(past_prices, prepend=past_prices[0])

            # Future returns
            future_prices = normalized[lookback:lookback + horizon]
            future_returns = np.diff(future_prices, prepend=future_prices[0])

            # Compute regime
            regime = compute_regime_vector(
                closes=closes,
                iv_points=features["iv_points"],
                funding_rate=features["funding_rate"],
                basis=features["basis"],
                order_imbalance=features["order_imbalance"],
                narrative_scores=features["narrative_scores"],
                skew=features["skew"],
            )

            regime_strength = regime.values[0] if regime.values else 1.0

            # Build input features: [return, regime, t, t^2, const]
            input_features = []
            for i, ret in enumerate(past_returns):
                t_norm = i / lookback
                feat = [ret, regime_strength, t_norm, t_norm ** 2, 1.0]
                input_features.append(feat)

            if len(input_features) < lookback or len(future_returns) < horizon:
                continue

            input_sequences.append(input_features)
            target_sequences.append(future_returns.tolist())

        except Exception as e:
            print(f"Warning: Failed to process MambaTS sample {idx}: {e}")
            continue

    print(f"Generated {len(input_sequences)} MambaTS training samples")

    if len(input_sequences) < 20:
        print("Warning: Too few MambaTS samples, using synthetic fallback")
        return _prepare_mamba_synthetic_fallback(n_samples, lookback, horizon)

    return input_sequences, target_sequences


def _prepare_mamba_synthetic_fallback(n_samples: int, lookback: int, horizon: int) -> Tuple[List, List]:
    """Fallback to synthetic data for MambaTS if real data insufficient."""
    print("Using synthetic MambaTS training data...")

    input_sequences = []
    target_sequences = []

    for _ in range(n_samples):
        trend = np.random.randn() * 0.001
        returns = np.random.randn(lookback) * 0.02 + trend

        features = []
        for i, ret in enumerate(returns):
            t_norm = i / lookback
            feat = [ret, 1.0, t_norm, t_norm ** 2, 1.0]
            features.append(feat)

        future_returns = np.random.randn(horizon) * 0.02 + trend

        input_sequences.append(features)
        target_sequences.append(future_returns.tolist())

    return input_sequences, target_sequences
