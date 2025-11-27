# Historical Validation Scripts

Comprehensive walk-forward validation for Aetheris Oracle forecasting models.

## Overview

These scripts run forecasts over historical data to measure:
- **Coverage**: Do P10-P90 intervals contain 80% of outcomes?
- **CRPS**: How accurate are the forecasts? (lower is better)
- **Sharpness**: How tight are the forecast intervals? (sharper is better, but only if calibrated)
- **Tail Accuracy**: Do P5/P95 quantiles capture extreme events?
- **Legacy vs SOTA**: How do traditional and neural models compare?

## Quick Start

### Install Dependencies

```bash
# Install tqdm if not already installed
pip install tqdm>=4.65.0

# Or reinstall all requirements
pip install -r requirements.txt
```

### Run Validation

```bash
# Quick test (7 days, fast)
python scripts/run_validation.py --quick

# Last 6 months (recommended for initial validation)
python scripts/run_validation.py --recent

# Full 2-year validation (comprehensive, takes longer)
python scripts/run_validation.py --full

# Custom date range
python scripts/run_validation.py --start 2023-06-01 --end 2024-12-31
```

### Output

Results are saved to `validation_results/` directory:

```
validation_results/
├── validation.log              # Detailed logs
├── validation_summary.json     # Aggregate metrics
├── forecast_records.jsonl      # All forecast records
└── validation_report.png       # Visualizations (4 charts)
```

## Usage Examples

### Test Only Legacy Models

```bash
python scripts/run_validation.py --recent --legacy-only
```

### Test Only SOTA Models

```bash
python scripts/run_validation.py --recent --sota-only
```

### Test Multiple Horizons

```bash
python scripts/run_validation.py --horizons 3 7 14 --recent
```

### Custom Forecast Interval

```bash
# Run forecasts every 3 days (more data points, slower)
python scripts/run_validation.py --interval 3 --recent

# Run forecasts every 14 days (fewer data points, faster)
python scripts/run_validation.py --interval 14 --full
```

### Test Different Asset

```bash
python scripts/run_validation.py --asset ETH-USD --recent
```

### Reduce Path Count for Speed

```bash
# Use 500 paths instead of default 1000 (faster, less accurate)
python scripts/run_validation.py --paths 500 --recent
```

## Advanced Usage

### Direct Python Script

For maximum control, use the main validation script directly:

```bash
python scripts/historical_validation.py \
  --asset BTC-USD \
  --start 2023-01-01 \
  --end 2024-12-31 \
  --interval 7 \
  --horizons 7 14 \
  --paths 1000 \
  --legacy \
  --sota \
  --output-dir my_validation_results
```

### Programmatic Usage

```python
from datetime import datetime
from scripts.historical_validation import ValidationConfig, HistoricalValidator

# Configure validation
config = ValidationConfig(
    asset_id="BTC-USD",
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 31),
    forecast_interval_days=7,
    horizons=[7],
    num_paths=1000,
    test_legacy=True,
    test_sota=True,
    output_dir="validation_results",
)

# Run validation
validator = HistoricalValidator(config)
results = validator.run_validation()

# Access results
for model_type, metrics in results["by_model"].items():
    print(f"{model_type}: Coverage = {metrics['coverage_p10_p90']:.1%}")
```

## Understanding the Output

### Validation Summary JSON

```json
{
  "config": { ... },
  "total_forecasts": 104,
  "by_model": {
    "legacy": {
      "coverage_p10_p90": 0.78,
      "coverage_p5_p95": 0.91,
      "mean_crps": 1250.45,
      "mean_interval_width_pct": 8.2
    },
    "sota": {
      "coverage_p10_p90": 0.82,
      "coverage_p5_p95": 0.93,
      "mean_crps": 980.32,
      "mean_interval_width_pct": 18.5
    }
  }
}
```

**Key Metrics**:
- `coverage_p10_p90`: Should be ~0.80 (80%)
- `coverage_p5_p95`: Should be ~0.90 (90%)
- `mean_crps`: Lower is better (absolute error)
- `mean_interval_width_pct`: Percentage of price (wider = less overconfident)

### Visualization Report

The `validation_report.png` contains 4 charts:

1. **Coverage Over Time** (top-left)
   - Rolling 30-day P10-P90 coverage rate
   - Should hover around 80% target line
   - Shows if model is over/under-confident over time

2. **CRPS Over Time** (top-right)
   - Rolling 30-day forecast error (normalized by price)
   - Lower is better
   - Shows if accuracy degrades over time

3. **Interval Width Over Time** (bottom-left)
   - Rolling 30-day P10-P90 width (% of price)
   - Shows forecast confidence
   - Wider = more uncertainty

4. **Coverage Comparison** (bottom-right)
   - Bar chart: Legacy vs SOTA
   - P10-P90 and P5-P95 coverage
   - Direct comparison with target lines

## Interpreting Results

### Well-Calibrated Model

✅ **Ideal metrics**:
- Coverage P10-P90: 75-85% (target: 80%)
- Coverage P5-P95: 85-95% (target: 90%)
- CRPS: As low as possible
- Interval width: Reasonable for crypto (15-25%)

### Overconfident Model (Legacy Problem)

⚠️ **Signs**:
- Coverage P10-P90: <70% (too narrow)
- Interval width: <10% (unrealistic for crypto)
- Many realized prices fall outside intervals

**Fix**: Use SOTA models or adjust calibration

### Underconfident Model

⚠️ **Signs**:
- Coverage P10-P90: >90% (too wide)
- Interval width: >40% (not useful)

**Fix**: Reduce uncertainty, tighten calibration

### Well-Performing SOTA

✅ **Expected**:
- Coverage better than legacy (closer to 80%)
- Lower CRPS (better accuracy)
- Wider intervals (20% vs 5%) but appropriately so
- Stable performance over time

## Troubleshooting

### "No evaluated records to visualize"

**Cause**: No realized prices could be fetched (data connector issues)

**Fix**:
- Check internet connection
- Verify asset ID is valid (e.g., "BTC-USD")
- Try using `--connector synthetic` for testing

### "Error forecasting ... h=7"

**Cause**: Forecast engine error (missing data, model failure)

**Fix**:
- Check logs in `validation_results/validation.log`
- Ensure sufficient historical data (need 90+ days before start date)
- Try reducing `--paths` to lower memory usage

### Very Low Coverage (<50%)

**Cause**: Model severely miscalibrated or data issues

**Fix**:
- Check if realized prices are reasonable
- Verify forecast dates are in the past (not future)
- Try shorter time period first (--quick)

### Slow Performance

**Tips**:
- Reduce `--paths` (default 500, try 250)
- Increase `--interval` (default 7, try 14)
- Use `--legacy-only` (faster than SOTA)
- Use `--quick` for testing (just 7 days)

## Performance Expectations

**Quick test** (7 days, 2 forecasts):
- Time: ~30 seconds
- Use for: Testing setup

**Recent** (6 months, ~26 forecasts):
- Time: ~10-15 minutes
- Use for: Initial validation

**Full** (2 years, ~104 forecasts):
- Time: ~45-60 minutes
- Use for: Comprehensive evaluation

*Times assume 500 paths, 7-day interval, both models*

## Next Steps After Validation

1. **If coverage is good (75-85%)**:
   - ✅ Models are well-calibrated
   - Deploy to production
   - Setup monitoring to track ongoing coverage

2. **If coverage is poor (<70%)**:
   - Train Neural Conformal Control (NCC) calibration
   - Adjust calibration hyperparameters
   - Re-run validation

3. **If SOTA underperforms legacy**:
   - SOTA models need training on real data
   - Run `scripts/train_all_sota.py`
   - Re-validate after training

4. **If validation looks good**:
   - Move to next TODO: Train remaining SOTA components
   - Setup monitoring & observability
   - Prepare for production deployment

## Integration with CI/CD

To run validation in CI/CD:

```bash
# In your CI pipeline
python scripts/run_validation.py --quick --output-dir ci_validation
EXIT_CODE=$?

# Check if coverage meets threshold
python -c "
import json
results = json.load(open('ci_validation/validation_summary.json'))
coverage = results['by_model']['sota']['coverage_p10_p90']
assert 0.70 <= coverage <= 0.90, f'Coverage {coverage} outside acceptable range'
"
```

## Contact & Support

For issues or questions:
- Check logs: `validation_results/validation.log`
- Review detailed records: `validation_results/forecast_records.jsonl`
- Open GitHub issue with logs attached

---

**Pro Tip**: Start with `--quick` to test everything works, then run `--recent` for real validation, finally `--full` for comprehensive evaluation.
