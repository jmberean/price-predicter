"""Tests for input validation (Phase 1 Fix #3)."""

import pytest
from datetime import datetime

from aetheris_oracle.config import ForecastConfig


def test_valid_config_passes():
    """Test that a valid configuration passes validation."""
    config = ForecastConfig()
    config.validate()  # Should not raise


def test_negative_horizon_raises():
    """Test that negative horizon raises ValueError."""
    config = ForecastConfig(horizon_days=-1)
    with pytest.raises(ValueError, match="horizon_days must be positive"):
        config.validate()


def test_zero_horizon_raises():
    """Test that zero horizon raises ValueError."""
    config = ForecastConfig(horizon_days=0)
    with pytest.raises(ValueError, match="horizon_days must be positive"):
        config.validate()


def test_negative_trailing_window_raises():
    """Test that negative trailing window raises ValueError."""
    config = ForecastConfig(trailing_window_days=-1)
    with pytest.raises(ValueError, match="trailing_window_days must be positive"):
        config.validate()


def test_zero_num_paths_raises():
    """Test that zero num_paths raises ValueError."""
    config = ForecastConfig(num_paths=0)
    with pytest.raises(ValueError, match="num_paths must be positive"):
        config.validate()


def test_negative_num_paths_raises():
    """Test that negative num_paths raises ValueError."""
    config = ForecastConfig(num_paths=-100)
    with pytest.raises(ValueError, match="num_paths must be positive"):
        config.validate()


def test_empty_asset_id_raises():
    """Test that empty asset_id raises ValueError."""
    config = ForecastConfig(asset_id="")
    with pytest.raises(ValueError, match="asset_id cannot be empty"):
        config.validate()


def test_whitespace_asset_id_raises():
    """Test that whitespace-only asset_id raises ValueError."""
    config = ForecastConfig(asset_id="   ")
    with pytest.raises(ValueError, match="asset_id cannot be empty"):
        config.validate()


def test_invalid_quantile_too_high_raises():
    """Test that quantile > 1 raises ValueError."""
    config = ForecastConfig(quantiles=[0.5, 1.5])
    with pytest.raises(ValueError, match="Quantile must be in"):
        config.validate()


def test_invalid_quantile_too_low_raises():
    """Test that quantile < 0 raises ValueError."""
    config = ForecastConfig(quantiles=[-0.1, 0.5])
    with pytest.raises(ValueError, match="Quantile must be in"):
        config.validate()


def test_negative_threshold_raises():
    """Test that negative threshold raises ValueError."""
    config = ForecastConfig(thresholds=(-1000.0,))
    with pytest.raises(ValueError, match="Threshold must be non-negative"):
        config.validate()


def test_valid_config_with_custom_params():
    """Test that valid custom config passes."""
    config = ForecastConfig(
        asset_id="ETH-USD",
        horizon_days=14,
        trailing_window_days=180,
        num_paths=5000,
        quantiles=[0.1, 0.5, 0.9],
        thresholds=[1000.0, 2000.0],
    )
    config.validate()  # Should not raise


def test_boundary_values():
    """Test boundary values for validation."""
    # Minimum valid values
    config = ForecastConfig(
        horizon_days=1,
        trailing_window_days=1,
        num_paths=1,
        quantiles=[0.0, 1.0],  # Edge cases
        thresholds=[0.0],
    )
    config.validate()  # Should not raise
