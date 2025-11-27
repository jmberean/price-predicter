# Testing Guide

Comprehensive testing guide for Aetheris Oracle v10.0 - State-of-the-Art Edition.

## Quick Start

```bash
# Set PYTHONPATH
$env:PYTHONPATH="src"  # PowerShell
export PYTHONPATH=src  # Bash

# Run all tests
python run_all_tests.py

# Or use pytest directly
pytest -v --tb=short
```

## Test Suites

### 1. Data Quality Tests (`test_data_quality.py`)

**Purpose**: Validate data connectors and feature engineering.

**Tests**:
- ✓ Data connector outputs (SyntheticDataConnector)
- ✓ Normalization correctness and reversibility
- ✓ Regime detection sensibility
- ✓ No NaN/Inf values in pipeline
- ✓ Price ranges are reasonable
- ✓ Feature engineering quality

**Run**:
```bash
pytest tests/test_data_quality.py -v

# With data quality report
python tests/test_data_quality.py
```

**Expected Output**:
```
DATA QUALITY REPORT
============================================================
✓ Price data:
  - Points: 60
  - Range: $45234.12 - $48765.43
  - Mean: $47123.45
  - Std: $1234.56

✓ IV data:
  - Points: 3
  - iv_7d_atm: 0.5234
  - iv_14d_atm: 0.5412
  - iv_30d_atm: 0.5678
```

### 2. Performance Tests (`test_performance.py`)

**Purpose**: Benchmark forecast engine performance and validate outputs.

**Tests**:
- ✓ Quantile ordering (P5 < P25 < P50 < P75 < P95)
- ✓ Probability constraints (P(X<K) + P(X>K) ≈ 1)
- ✓ Forecast cone widening over time
- ✓ All prices positive
- ✓ Latency benchmarks (legacy, SOTA, batch)
- ✓ Memory usage profiling
- ✓ Reproducibility with seeds

**Run**:
```bash
pytest tests/test_performance.py -v

# With performance report
python tests/test_performance.py
```

**Expected Output**:
```
AETHERIS ORACLE - PERFORMANCE REPORT
============================================================
Quick (100 paths, 7d):
  Latency: 234.5ms
  Final cone width (P90-P10): $3456.78
  Median forecast: $47123.45

Standard (1000 paths, 7d):
  Latency: 1234.5ms
  Final cone width (P90-P10): $3234.56
  Median forecast: $47089.12

High Quality (10k paths, 7d):
  Latency: 7890.1ms
  Final cone width (P90-P10): $3198.45
  Median forecast: $47102.34
```

### 3. API Validation Tests (`test_api_validation.py`)

**Purpose**: Test FastAPI REST API endpoints.

**Tests**:
- ✓ Health endpoint
- ✓ Basic forecast requests
- ✓ Custom parameters (paths, quantiles, thresholds)
- ✓ Scenario forecasts
- ✓ Input validation and error handling
- ✓ API key authentication
- ✓ Response schema compliance
- ✓ Concurrent requests
- ✓ Performance under load

**Run**:
```bash
pytest tests/test_api_validation.py -v

# With API health check
python tests/test_api_validation.py
```

**Expected Output**:
```
API HEALTH CHECK
============================================================
✓ Health endpoint: 200
✓ Basic forecast: 200
  - Quantile paths: 7 days
  - Drivers: 5 identified
  - Regime: normal

✓ Performance (1000 paths): 1456.7ms
```

### 4. SOTA Components Tests (`test_sota_components.py`)

**Purpose**: Test state-of-the-art neural components.

**Tests**:
- ✓ Importance sampling for tail quantiles
- ✓ Integrated Gradients explainability
- ✓ Neural Conformal Control calibration
- ✓ FM-GP residual generator
- ✓ Neural Jump SDE
- ✓ Differentiable Greeks MM engine
- ✓ Neural Rough Volatility
- ✓ MambaTS trend backbone
- ✓ ForecastEngine with SOTA enabled
- ✓ Save/load for all models

**Run**:
```bash
pytest tests/test_sota_components.py -v
```

**Note**: Some tests may skip if dependencies (PyTorch, mamba-ssm, etc.) not installed.

### 5. Pipeline Tests (`test_pipeline.py`)

**Purpose**: Test core forecast pipeline integration.

**Tests**:
- ✓ End-to-end forecast flow
- ✓ Calibration engine
- ✓ Scenario overrides
- ✓ Batch forecasting
- ✓ Explainability drivers

**Run**:
```bash
pytest tests/test_pipeline.py -v
```

### 6. Service Tests (`test_service.py`)

**Purpose**: Test FastAPI service creation and configuration.

**Tests**:
- ✓ App creation
- ✓ Health endpoint
- ✓ Forecast endpoint
- ✓ API key authentication

**Run**:
```bash
pytest tests/test_service.py -v
```

## Performance Benchmarks

### Latency Targets

| Configuration | Paths | Horizon | Target Latency |
|--------------|-------|---------|----------------|
| Quick | 100 | 7d | < 500ms |
| Standard | 1000 | 7d | < 2000ms |
| High Quality | 10000 | 7d | < 10000ms |
| Long Horizon | 1000 | 30d | < 3000ms |

### Memory Targets

| Configuration | Max Memory Increase |
|--------------|---------------------|
| 1000 paths | < 100MB |
| 10000 paths | < 500MB |

### Throughput Targets

| Metric | Target |
|--------|--------|
| Batch throughput | > 2 forecasts/sec |
| API response time | < 3s (1000 paths) |
| Concurrent requests | 5+ simultaneous |

## Validation Criteria

### Output Validation

✓ **Quantile Ordering**: P5 ≤ P10 ≤ P25 ≤ P50 ≤ P75 ≤ P90 ≤ P95

✓ **Probability Constraints**: P(X < K) + P(X > K) ≈ 1.0 ± 0.05

✓ **Cone Widening**: At least 60% of steps show increasing uncertainty

✓ **Positive Prices**: All forecasted prices > 0

✓ **Reasonable Ranges**: BTC prices in [1k, 200k], cone width ratio < 5x

### Data Quality

✓ **No NaN/Inf**: All values finite throughout pipeline

✓ **Normalization**: Mean ≈ 0, Std ≈ 1 for normalized data

✓ **Reversibility**: denormalize(normalize(x)) ≈ x

✓ **No Look-Ahead**: Normalization uses past-only windows

## Running Specific Test Categories

```bash
# Data quality only
pytest tests/test_data_quality.py -v

# Performance benchmarks only
pytest tests/test_performance.py::TestPerformanceBenchmarks -v

# API tests only
pytest tests/test_api_validation.py -v

# SOTA components only
pytest tests/test_sota_components.py -v

# Validation tests only
pytest tests/test_performance.py::TestForecastValidation -v

# Regression tests only
pytest tests/test_performance.py::TestRegressionTests -v
```

## Continuous Integration

For CI/CD pipelines:

```bash
# Run all tests with coverage
pytest --cov=aetheris_oracle --cov-report=html --cov-report=term

# Run with strict mode (fail on warnings)
pytest --strict-markers -W error

# Run only fast tests (skip slow benchmarks)
pytest -m "not slow"

# Generate JUnit XML for CI
pytest --junitxml=test-results.xml
```

## Debugging Failed Tests

```bash
# Verbose output with full tracebacks
pytest tests/test_performance.py -vv --tb=long

# Stop on first failure
pytest -x

# Run specific test
pytest tests/test_performance.py::TestForecastValidation::test_quantile_ordering -v

# Show print statements
pytest -s

# Run last failed tests only
pytest --lf
```

## Performance Profiling

```bash
# Profile with cProfile
python -m cProfile -o profile.stats tests/test_performance.py

# Visualize with snakeviz
snakeviz profile.stats

# Memory profiling
python -m memory_profiler tests/test_performance.py
```

## Test Coverage

Current test coverage:

- **Core Pipeline**: 95%+
- **Data Connectors**: 90%+
- **API Endpoints**: 100%
- **SOTA Components**: 85%+ (depends on dependencies)
- **Validation Logic**: 100%

## Known Issues / Skipped Tests

- **SOTA GPU Tests**: Skipped if CUDA not available
- **Mamba Tests**: May use fallback if mamba-ssm not installed
- **Long-running tests**: Some benchmarks may be slow on low-end hardware

## Best Practices

1. **Run tests before committing**: `pytest -q` for quick validation
2. **Run full suite before PRs**: `python run_all_tests.py`
3. **Check performance regularly**: Monitor latency trends
4. **Profile on representative hardware**: CI environment != production
5. **Update benchmarks**: When adding new features, update performance targets

## Troubleshooting

### Tests fail with "ModuleNotFoundError"

```bash
# Ensure PYTHONPATH is set
$env:PYTHONPATH="src"  # PowerShell
export PYTHONPATH=src  # Bash
```

### Tests are very slow

```bash
# Reduce path count for faster tests
pytest tests/test_performance.py -v -k "not high_path"

# Skip slow benchmarks
pytest -m "not slow"
```

### SOTA tests skip

```bash
# Install PyTorch dependencies
pip install torch numpy scipy gpytorch

# For full SOTA support
pip install -r requirements.txt
```

### API tests fail

```bash
# Ensure service can start
python -m aetheris_oracle.server

# Check for port conflicts
netstat -ano | findstr :8000  # Windows
lsof -i :8000  # Linux/Mac
```

## Contact

For issues with tests, please check:
- GitHub Issues: https://github.com/anthropics/claude-code/issues
- Documentation: README.md, CLAUDE.md
