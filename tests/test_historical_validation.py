"""Tests for historical validation script."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from scripts.historical_validation import (
    ValidationConfig,
    ForecastRecord,
    HistoricalValidator,
)


def test_validation_config_defaults():
    """Test ValidationConfig default values."""
    config = ValidationConfig()

    assert config.asset_id == "BTC-USD"
    assert config.horizons == [7]
    assert config.num_paths == 1000
    assert config.test_legacy is True
    assert config.test_sota is True


def test_forecast_record_creation():
    """Test ForecastRecord creation."""
    record = ForecastRecord(
        forecast_date=datetime(2024, 1, 1),
        horizon=7,
        model_type="legacy",
        quantiles={0.1: 90000, 0.5: 95000, 0.9: 100000},
        realized_price=96000,
    )

    assert record.forecast_date == datetime(2024, 1, 1)
    assert record.horizon == 7
    assert record.model_type == "legacy"
    assert record.realized_price == 96000


def test_generate_forecast_dates():
    """Test forecast date generation."""
    config = ValidationConfig(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 31),
        forecast_interval_days=7,
    )

    validator = HistoricalValidator(config)
    dates = validator._generate_forecast_dates()

    # Should generate: Jan 1, 8, 15, 22, 29
    assert len(dates) == 5
    assert dates[0] == datetime(2024, 1, 1)
    assert dates[-1] == datetime(2024, 1, 29)


def test_compute_metrics():
    """Test metrics computation for a forecast record."""
    config = ValidationConfig()
    validator = HistoricalValidator(config)

    # Create a record
    record = ForecastRecord(
        forecast_date=datetime(2024, 1, 1),
        horizon=7,
        model_type="legacy",
        quantiles={
            0.05: 85000,
            0.1: 88000,
            0.5: 95000,
            0.9: 102000,
            0.95: 105000,
        },
        realized_price=96000,  # Falls within P10-P90
    )

    # Mock result object
    mock_result = Mock()
    mock_result.quantile_paths = {7: record.quantiles}

    # Compute metrics
    record_with_metrics = validator._compute_metrics(record, mock_result)

    # Check coverage
    assert record_with_metrics.coverage_p10_p90 is True  # 88k <= 96k <= 102k
    assert record_with_metrics.coverage_p5_p95 is True  # 85k <= 96k <= 105k

    # Check interval width
    expected_width = (102000 - 88000) / 96000
    assert abs(record_with_metrics.interval_width_p10_p90 - expected_width) < 0.01

    # Check CRPS exists
    assert record_with_metrics.crps is not None
    assert record_with_metrics.crps > 0


def test_aggregate_metrics():
    """Test aggregate metrics computation."""
    config = ValidationConfig()
    validator = HistoricalValidator(config)

    # Add some mock records
    for i in range(10):
        record = ForecastRecord(
            forecast_date=datetime(2024, 1, 1) + timedelta(days=i*7),
            horizon=7,
            model_type="legacy",
            quantiles={0.1: 90000, 0.5: 95000, 0.9: 100000},
            realized_price=95000 + i*1000,  # Half will be covered, half won't
            coverage_p10_p90=(i < 5),  # First 5 are covered
            coverage_p5_p95=(i < 8),  # First 8 are covered
            crps=1000.0 + i*100,
            interval_width_p10_p90=0.10,
        )
        validator.records.append(record)

    # Compute aggregates
    results = validator._compute_aggregate_metrics()

    # Check structure
    assert "by_model" in results
    assert "legacy" in results["by_model"]

    legacy_metrics = results["by_model"]["legacy"]
    assert legacy_metrics["coverage_p10_p90"] == 0.5  # 5/10
    assert legacy_metrics["coverage_p5_p95"] == 0.8  # 8/10
    assert legacy_metrics["mean_crps"] is not None
    assert legacy_metrics["mean_interval_width_pct"] == 10.0


def test_create_engine_legacy():
    """Test legacy engine creation."""
    config = ValidationConfig()
    validator = HistoricalValidator(config)

    engine = validator._create_engine("legacy")

    # Should have SOTA flags set to False
    assert engine.use_neural_rough_vol is False
    assert engine.use_fm_gp_residuals is False


def test_create_engine_sota():
    """Test SOTA engine creation."""
    config = ValidationConfig()
    validator = HistoricalValidator(config)

    engine = validator._create_engine("sota")

    # Should have some SOTA flags set to True
    assert engine.use_neural_rough_vol is True
    assert engine.use_fm_gp_residuals is True


@pytest.mark.parametrize("model_type,expected", [
    ("legacy", False),
    ("sota", True),
])
def test_model_type_selection(model_type, expected):
    """Test model type selection."""
    config = ValidationConfig()
    validator = HistoricalValidator(config)

    engine = validator._create_engine(model_type)

    assert engine.use_neural_rough_vol == expected
    assert engine.use_fm_gp_residuals == expected


def test_validation_config_custom_horizons():
    """Test ValidationConfig with custom horizons."""
    config = ValidationConfig(
        horizons=[3, 7, 14]
    )

    assert len(config.horizons) == 3
    assert 3 in config.horizons
    assert 7 in config.horizons
    assert 14 in config.horizons


def test_forecast_record_metrics_none_when_no_realized_price():
    """Test that metrics are None when realized price is not available."""
    record = ForecastRecord(
        forecast_date=datetime(2024, 1, 1),
        horizon=7,
        model_type="legacy",
        quantiles={0.1: 90000, 0.5: 95000, 0.9: 100000},
        realized_price=None,  # No realized price
    )

    # Metrics should be None
    assert record.crps is None
    assert record.coverage_p10_p90 is None
    assert record.coverage_p5_p95 is None


def test_interval_width_calculation():
    """Test interval width percentage calculation."""
    config = ValidationConfig()
    validator = HistoricalValidator(config)

    record = ForecastRecord(
        forecast_date=datetime(2024, 1, 1),
        horizon=7,
        model_type="legacy",
        quantiles={
            0.1: 90000,   # P10
            0.5: 100000,  # P50
            0.9: 110000,  # P90
        },
        realized_price=100000,
    )

    mock_result = Mock()
    mock_result.quantile_paths = {7: record.quantiles}

    record_with_metrics = validator._compute_metrics(record, mock_result)

    # Width = (110k - 90k) / 100k = 0.20 = 20%
    assert abs(record_with_metrics.interval_width_p10_p90 - 0.20) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
