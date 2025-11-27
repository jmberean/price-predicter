"""
Convenience wrapper for running historical validation.

Quick usage examples:

# Quick test (7 days of data)
python scripts/run_validation.py --quick

# Last 6 months
python scripts/run_validation.py --recent

# Full 2-year validation
python scripts/run_validation.py --full

# Custom range
python scripts/run_validation.py --start 2024-01-01 --end 2024-12-31
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from historical_validation import ValidationConfig, HistoricalValidator


def main():
    parser = argparse.ArgumentParser(description="Run historical validation with presets")
    parser.add_argument("--quick", action="store_true", help="Quick test (7 days)")
    parser.add_argument("--recent", action="store_true", help="Last 6 months")
    parser.add_argument("--full", action="store_true", help="Full 2-year validation")

    parser.add_argument("--asset", type=str, default="BTC-USD", help="Asset ID")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--interval", type=int, default=7, help="Forecast interval in days")
    parser.add_argument("--horizons", type=int, nargs="+", default=[7], help="Forecast horizons")
    parser.add_argument("--paths", type=int, default=500, help="Number of forecast paths")
    parser.add_argument("--legacy-only", action="store_true", help="Test legacy models only")
    parser.add_argument("--sota-only", action="store_true", help="Test SOTA models only")
    parser.add_argument("--output-dir", type=str, default="validation_results", help="Output directory")

    args = parser.parse_args()

    # Determine date range
    if args.quick:
        # Just 7 days ago for quick testing
        start = datetime.now() - timedelta(days=14)
        end = datetime.now() - timedelta(days=7)
        print("🚀 Quick test mode: 7 days")
    elif args.recent:
        # Last 6 months
        end = datetime.now()
        start = end - timedelta(days=180)
        print("📊 Recent mode: Last 6 months")
    elif args.full:
        # Full 2 years
        end = datetime.now()
        start = end - timedelta(days=730)
        print("📈 Full validation: 2 years")
    else:
        # Custom or default
        if args.start and args.end:
            start = datetime.strptime(args.start, "%Y-%m-%d")
            end = datetime.strptime(args.end, "%Y-%m-%d")
        else:
            # Default: last 3 months
            end = datetime.now()
            start = end - timedelta(days=90)
            print("📊 Default mode: Last 3 months")

    # Model selection
    test_legacy = not args.sota_only
    test_sota = not args.legacy_only

    # Create config
    config = ValidationConfig(
        asset_id=args.asset,
        start_date=start,
        end_date=end,
        forecast_interval_days=args.interval,
        horizons=args.horizons,
        num_paths=args.paths,
        test_legacy=test_legacy,
        test_sota=test_sota,
        output_dir=args.output_dir,
    )

    print(f"\n{'='*60}")
    print("HISTORICAL VALIDATION")
    print(f"{'='*60}")
    print(f"Asset: {config.asset_id}")
    print(f"Period: {start.date()} to {end.date()}")
    print(f"Horizons: {config.horizons} days")
    print(f"Interval: Every {config.forecast_interval_days} days")
    print(f"Paths: {config.num_paths}")
    print(f"Models: {'Legacy' if test_legacy else ''}{' + ' if (test_legacy and test_sota) else ''}{'SOTA' if test_sota else ''}")
    print(f"{'='*60}\n")

    # Run validation
    validator = HistoricalValidator(config)
    results = validator.run_validation()

    # Print summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)

    for model_type, metrics in results.get("by_model", {}).items():
        print(f"\n{model_type.upper()}:")

        coverage_p10_p90 = metrics.get('coverage_p10_p90')
        coverage_p5_p95 = metrics.get('coverage_p5_p95')
        mean_crps = metrics.get('mean_crps')
        mean_width = metrics.get('mean_interval_width_pct')

        if coverage_p10_p90 is not None:
            # Color code based on target
            cov_status = "✅" if 0.75 <= coverage_p10_p90 <= 0.85 else "⚠️"
            print(f"  Coverage P10-P90: {coverage_p10_p90:.1%} (target: 80%) {cov_status}")

        if coverage_p5_p95 is not None:
            cov_status = "✅" if 0.85 <= coverage_p5_p95 <= 0.95 else "⚠️"
            print(f"  Coverage P5-P95: {coverage_p5_p95:.1%} (target: 90%) {cov_status}")

        if mean_crps is not None:
            print(f"  Mean CRPS: {mean_crps:.2f}")

        if mean_width is not None:
            width_status = "✅" if 10 <= mean_width <= 30 else "⚠️"
            print(f"  Mean Interval Width: {mean_width:.1f}% {width_status}")

    print(f"\n📁 Detailed results saved to: {config.output_dir}/")
    print(f"📊 Visualization: {config.output_dir}/validation_report.png")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
