"""
Historical validation script for walk-forward evaluation.

Runs forecasts over historical data and measures:
- Coverage (P10-P90 hit rate)
- CRPS (Continuous Ranked Probability Score)
- Tail accuracy (P5/P95 performance)
- Sharpness (forecast interval width)
- Legacy vs SOTA comparison

Usage:
    python scripts/historical_validation.py --start 2023-01-01 --end 2024-12-31 --asset BTC-USD
"""

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm

from aetheris_oracle.config import ForecastConfig
from aetheris_oracle.pipeline.forecast import ForecastEngine
from aetheris_oracle.data.free_connectors import FreeDataConnector
from aetheris_oracle.monitoring.advanced_metrics import (
    compute_crps,
    compute_qice,
    compute_sharpness,
    AdvancedMetricsCollector,
)


@dataclass
class ValidationConfig:
    """Configuration for historical validation."""
    asset_id: str = "BTC-USD"
    start_date: datetime = datetime(2023, 1, 1)
    end_date: datetime = datetime(2024, 12, 31)
    forecast_interval_days: int = 7  # Run forecast every N days
    horizons: List[int] = None  # Forecast horizons to test
    num_paths: int = 1000
    test_legacy: bool = True
    test_sota: bool = True
    output_dir: str = "validation_results"

    def __post_init__(self):
        if self.horizons is None:
            self.horizons = [7]  # Default to 7-day horizon


@dataclass
class ForecastRecord:
    """Record of a single forecast and its outcome."""
    forecast_date: datetime
    horizon: int
    model_type: str  # "legacy" or "sota"
    quantiles: Dict[float, float]  # P5, P10, P25, P50, P75, P90, P95
    realized_price: Optional[float] = None
    forecast_paths: Optional[List[List[float]]] = None

    # Metrics
    crps: Optional[float] = None
    coverage_p10_p90: Optional[bool] = None
    coverage_p5_p95: Optional[bool] = None
    interval_width_p10_p90: Optional[float] = None
    qice_metrics: Optional[Dict[str, float]] = None


class HistoricalValidator:
    """Main orchestrator for historical validation."""

    def __init__(self, config: ValidationConfig):
        self.config = config
        self.connector = FreeDataConnector()
        self.records: List[ForecastRecord] = []

        # Setup logging
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.output_dir / "validation.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def run_validation(self) -> Dict[str, any]:
        """Run walk-forward validation over the configured period."""
        self.logger.info(f"Starting validation: {self.config.start_date} to {self.config.end_date}")
        self.logger.info(f"Asset: {self.config.asset_id}, Horizons: {self.config.horizons}")

        # Generate forecast dates
        forecast_dates = self._generate_forecast_dates()
        self.logger.info(f"Generated {len(forecast_dates)} forecast dates")

        # Run forecasts for each model type
        model_types = []
        if self.config.test_legacy:
            model_types.append("legacy")
        if self.config.test_sota:
            model_types.append("sota")

        total_forecasts = len(forecast_dates) * len(self.config.horizons) * len(model_types)
        self.logger.info(f"Total forecasts to generate: {total_forecasts}")

        # Progress bar
        pbar = tqdm(total=total_forecasts, desc="Running forecasts")

        for model_type in model_types:
            engine = self._create_engine(model_type)

            for forecast_date in forecast_dates:
                for horizon in self.config.horizons:
                    try:
                        record = self._run_single_forecast(
                            engine, model_type, forecast_date, horizon
                        )
                        self.records.append(record)
                    except Exception as e:
                        self.logger.error(f"Error forecasting {model_type} {forecast_date} h={horizon}: {e}")

                    pbar.update(1)

        pbar.close()

        # Compute aggregate metrics
        results = self._compute_aggregate_metrics()

        # Save results
        self._save_results(results)

        # Generate visualizations
        self._generate_visualizations()

        self.logger.info(f"Validation complete! Results saved to {self.output_dir}")

        return results

    def _generate_forecast_dates(self) -> List[datetime]:
        """Generate list of dates to run forecasts."""
        dates = []
        current = self.config.start_date

        while current <= self.config.end_date:
            dates.append(current)
            current += timedelta(days=self.config.forecast_interval_days)

        return dates

    def _create_engine(self, model_type: str) -> ForecastEngine:
        """Create forecast engine with appropriate model configuration."""
        if model_type == "legacy":
            return ForecastEngine(
                connector=self.connector,
                use_neural_rough_vol=False,
                use_fm_gp_residuals=False,
                use_neural_jumps=False,
                use_diff_greeks=False,
                use_mamba_trend=False,
                use_ncc_calibration=False,
            )
        else:  # sota
            return ForecastEngine(
                connector=self.connector,
                use_neural_rough_vol=True,
                use_fm_gp_residuals=True,
                use_neural_jumps=False,  # Not trained yet
                use_diff_greeks=False,  # Not trained yet
                use_mamba_trend=False,  # Not trained yet
                use_ncc_calibration=False,  # Not trained yet
            )

    def _run_single_forecast(
        self,
        engine: ForecastEngine,
        model_type: str,
        forecast_date: datetime,
        horizon: int
    ) -> ForecastRecord:
        """Run a single forecast and evaluate against realized outcome."""
        # Generate forecast
        config = ForecastConfig(
            asset_id=self.config.asset_id,
            horizon_days=horizon,
            num_paths=self.config.num_paths,
            as_of=forecast_date,
        )

        result = engine.forecast(config)

        # Extract quantiles at final horizon
        quantiles = result.quantile_paths.get(horizon, {})

        # Fetch realized price
        try:
            future_date = forecast_date + timedelta(days=horizon)
            future_frame = self.connector.fetch_window(
                asset_id=self.config.asset_id,
                as_of=future_date,
                window=timedelta(days=1),
            )
            realized_price = future_frame.latest_price()
        except Exception as e:
            self.logger.warning(f"Could not fetch realized price for {future_date}: {e}")
            realized_price = None

        # Create record
        record = ForecastRecord(
            forecast_date=forecast_date,
            horizon=horizon,
            model_type=model_type,
            quantiles=quantiles,
            realized_price=realized_price,
        )

        # Compute metrics if realized price available
        if realized_price is not None:
            record = self._compute_metrics(record, result)

        return record

    def _compute_metrics(self, record: ForecastRecord, result) -> ForecastRecord:
        """Compute metrics for a forecast record."""
        realized = record.realized_price
        quantiles = record.quantiles

        # Coverage P10-P90
        p10 = quantiles.get(0.1, realized)
        p90 = quantiles.get(0.9, realized)
        record.coverage_p10_p90 = p10 <= realized <= p90

        # Coverage P5-P95
        p5 = quantiles.get(0.05, realized)
        p95 = quantiles.get(0.95, realized)
        record.coverage_p5_p95 = p5 <= realized <= p95

        # Interval width
        record.interval_width_p10_p90 = (p90 - p10) / realized if realized > 0 else 0.0

        # CRPS (use median as point forecast)
        p50 = quantiles.get(0.5, realized)
        # Simplified CRPS using quantiles
        record.crps = abs(p50 - realized) + 0.5 * (p90 - p10)

        # QICE
        record.qice_metrics = compute_qice(
            quantiles,
            realized,
            [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
        )

        return record

    def _compute_aggregate_metrics(self) -> Dict[str, any]:
        """Compute aggregate metrics across all forecasts."""
        results = {
            "config": asdict(self.config),
            "total_forecasts": len(self.records),
            "by_model": {},
            "by_horizon": {},
        }

        # Group by model type
        for model_type in ["legacy", "sota"]:
            model_records = [r for r in self.records if r.model_type == model_type]

            if not model_records:
                continue

            # Filter records with realized prices
            evaluated_records = [r for r in model_records if r.realized_price is not None]

            if not evaluated_records:
                continue

            # Coverage metrics
            p10_p90_hits = sum(1 for r in evaluated_records if r.coverage_p10_p90)
            p5_p95_hits = sum(1 for r in evaluated_records if r.coverage_p5_p95)
            total = len(evaluated_records)

            # CRPS
            crps_values = [r.crps for r in evaluated_records if r.crps is not None]
            mean_crps = np.mean(crps_values) if crps_values else None

            # Sharpness
            widths = [r.interval_width_p10_p90 for r in evaluated_records if r.interval_width_p10_p90 is not None]
            mean_width = np.mean(widths) if widths else None

            results["by_model"][model_type] = {
                "total_forecasts": len(model_records),
                "evaluated_forecasts": total,
                "coverage_p10_p90": p10_p90_hits / total if total > 0 else None,
                "coverage_p5_p95": p5_p95_hits / total if total > 0 else None,
                "mean_crps": float(mean_crps) if mean_crps is not None else None,
                "mean_interval_width_pct": float(mean_width * 100) if mean_width is not None else None,
                "target_coverage_p10_p90": 0.80,
                "target_coverage_p5_p95": 0.90,
            }

        # Group by horizon
        for horizon in self.config.horizons:
            horizon_records = [r for r in self.records if r.horizon == horizon and r.realized_price is not None]

            if not horizon_records:
                continue

            p10_p90_hits = sum(1 for r in horizon_records if r.coverage_p10_p90)
            total = len(horizon_records)

            results["by_horizon"][horizon] = {
                "coverage_p10_p90": p10_p90_hits / total if total > 0 else None,
            }

        return results

    def _save_results(self, results: Dict[str, any]):
        """Save validation results to files."""
        # Save summary JSON
        summary_path = self.output_dir / "validation_summary.json"
        with open(summary_path, "w") as f:
            json.dump(results, indent=2, fp=f)

        self.logger.info(f"Saved summary to {summary_path}")

        # Save detailed records as JSONL
        records_path = self.output_dir / "forecast_records.jsonl"
        with open(records_path, "w") as f:
            for record in self.records:
                record_dict = {
                    "forecast_date": record.forecast_date.isoformat(),
                    "horizon": record.horizon,
                    "model_type": record.model_type,
                    "quantiles": record.quantiles,
                    "realized_price": record.realized_price,
                    "coverage_p10_p90": record.coverage_p10_p90,
                    "coverage_p5_p95": record.coverage_p5_p95,
                    "interval_width_p10_p90": record.interval_width_p10_p90,
                    "crps": record.crps,
                }
                f.write(json.dumps(record_dict) + "\n")

        self.logger.info(f"Saved detailed records to {records_path}")

    def _generate_visualizations(self):
        """Generate validation visualizations."""
        self.logger.info("Generating visualizations...")

        # Filter records with realized prices
        evaluated_records = [r for r in self.records if r.realized_price is not None]

        if not evaluated_records:
            self.logger.warning("No evaluated records to visualize")
            return

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Historical Validation: {self.config.asset_id}", fontsize=16, fontweight="bold")

        # Plot 1: Coverage over time
        self._plot_coverage_over_time(axes[0, 0], evaluated_records)

        # Plot 2: CRPS over time
        self._plot_crps_over_time(axes[0, 1], evaluated_records)

        # Plot 3: Interval width over time
        self._plot_interval_width_over_time(axes[1, 0], evaluated_records)

        # Plot 4: Coverage comparison (legacy vs SOTA)
        self._plot_coverage_comparison(axes[1, 1], evaluated_records)

        plt.tight_layout()

        # Save figure
        viz_path = self.output_dir / "validation_report.png"
        plt.savefig(viz_path, dpi=150, bbox_inches="tight")
        self.logger.info(f"Saved visualization to {viz_path}")

        plt.close()

    def _plot_coverage_over_time(self, ax, records):
        """Plot coverage rate over time."""
        for model_type in ["legacy", "sota"]:
            model_records = [r for r in records if r.model_type == model_type]
            if not model_records:
                continue

            # Sort by date
            model_records.sort(key=lambda r: r.forecast_date)

            # Rolling coverage (30-day window)
            window_size = 30 // self.config.forecast_interval_days
            dates = [r.forecast_date for r in model_records]
            coverages = [1.0 if r.coverage_p10_p90 else 0.0 for r in model_records]

            # Compute rolling average
            rolling_cov = []
            for i in range(len(coverages)):
                start = max(0, i - window_size + 1)
                rolling_cov.append(np.mean(coverages[start:i+1]))

            ax.plot(dates, rolling_cov, label=model_type.upper(), linewidth=2)

        ax.axhline(0.80, color="green", linestyle="--", label="Target (80%)", alpha=0.7)
        ax.set_xlabel("Date")
        ax.set_ylabel("P10-P90 Coverage (30d rolling)")
        ax.set_title("Coverage Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    def _plot_crps_over_time(self, ax, records):
        """Plot CRPS over time."""
        for model_type in ["legacy", "sota"]:
            model_records = [r for r in records if r.model_type == model_type and r.crps is not None]
            if not model_records:
                continue

            model_records.sort(key=lambda r: r.forecast_date)

            dates = [r.forecast_date for r in model_records]
            crps = [r.crps for r in model_records]

            # Normalize by price for readability
            crps_normalized = [c / r.realized_price if r.realized_price > 0 else 0 for c, r in zip(crps, model_records)]

            # Rolling average
            window_size = 30 // self.config.forecast_interval_days
            rolling_crps = []
            for i in range(len(crps_normalized)):
                start = max(0, i - window_size + 1)
                rolling_crps.append(np.mean(crps_normalized[start:i+1]))

            ax.plot(dates, rolling_crps, label=model_type.upper(), linewidth=2)

        ax.set_xlabel("Date")
        ax.set_ylabel("CRPS / Price (30d rolling)")
        ax.set_title("Forecast Error Over Time (lower is better)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    def _plot_interval_width_over_time(self, ax, records):
        """Plot interval width over time."""
        for model_type in ["legacy", "sota"]:
            model_records = [r for r in records if r.model_type == model_type and r.interval_width_p10_p90 is not None]
            if not model_records:
                continue

            model_records.sort(key=lambda r: r.forecast_date)

            dates = [r.forecast_date for r in model_records]
            widths = [r.interval_width_p10_p90 * 100 for r in model_records]  # Convert to percentage

            # Rolling average
            window_size = 30 // self.config.forecast_interval_days
            rolling_widths = []
            for i in range(len(widths)):
                start = max(0, i - window_size + 1)
                rolling_widths.append(np.mean(widths[start:i+1]))

            ax.plot(dates, rolling_widths, label=model_type.upper(), linewidth=2)

        ax.set_xlabel("Date")
        ax.set_ylabel("P10-P90 Width (% of price, 30d rolling)")
        ax.set_title("Forecast Sharpness Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    def _plot_coverage_comparison(self, ax, records):
        """Plot coverage comparison bar chart."""
        model_types = []
        p10_p90_coverages = []
        p5_p95_coverages = []

        for model_type in ["legacy", "sota"]:
            model_records = [r for r in records if r.model_type == model_type]
            if not model_records:
                continue

            p10_p90_hits = sum(1 for r in model_records if r.coverage_p10_p90)
            p5_p95_hits = sum(1 for r in model_records if r.coverage_p5_p95)
            total = len(model_records)

            model_types.append(model_type.upper())
            p10_p90_coverages.append(p10_p90_hits / total if total > 0 else 0)
            p5_p95_coverages.append(p5_p95_hits / total if total > 0 else 0)

        x = np.arange(len(model_types))
        width = 0.35

        ax.bar(x - width/2, p10_p90_coverages, width, label="P10-P90 (target 80%)", alpha=0.8)
        ax.bar(x + width/2, p5_p95_coverages, width, label="P5-P95 (target 90%)", alpha=0.8)

        ax.axhline(0.80, color="green", linestyle="--", alpha=0.5)
        ax.axhline(0.90, color="blue", linestyle="--", alpha=0.5)

        ax.set_ylabel("Coverage Rate")
        ax.set_title("Coverage Comparison: Legacy vs SOTA")
        ax.set_xticks(x)
        ax.set_xticklabels(model_types)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, 1.0)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run historical validation")
    parser.add_argument("--asset", type=str, default="BTC-USD", help="Asset ID")
    parser.add_argument("--start", type=str, default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--interval", type=int, default=7, help="Forecast interval in days")
    parser.add_argument("--horizons", type=int, nargs="+", default=[7], help="Forecast horizons")
    parser.add_argument("--paths", type=int, default=1000, help="Number of forecast paths")
    parser.add_argument("--legacy", action="store_true", default=True, help="Test legacy models")
    parser.add_argument("--sota", action="store_true", default=True, help="Test SOTA models")
    parser.add_argument("--output-dir", type=str, default="validation_results", help="Output directory")

    args = parser.parse_args()

    config = ValidationConfig(
        asset_id=args.asset,
        start_date=datetime.strptime(args.start, "%Y-%m-%d"),
        end_date=datetime.strptime(args.end, "%Y-%m-%d"),
        forecast_interval_days=args.interval,
        horizons=args.horizons,
        num_paths=args.paths,
        test_legacy=args.legacy,
        test_sota=args.sota,
        output_dir=args.output_dir,
    )

    validator = HistoricalValidator(config)
    results = validator.run_validation()

    # Print summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)

    for model_type, metrics in results.get("by_model", {}).items():
        print(f"\n{model_type.upper()}:")
        print(f"  Coverage P10-P90: {metrics['coverage_p10_p90']:.1%} (target: 80%)")
        print(f"  Coverage P5-P95: {metrics['coverage_p5_p95']:.1%} (target: 90%)")
        print(f"  Mean CRPS: {metrics['mean_crps']:.2f}")
        print(f"  Mean Interval Width: {metrics['mean_interval_width_pct']:.1f}%")

    print(f"\nDetailed results saved to: {config.output_dir}/")
    print("="*60)


if __name__ == "__main__":
    main()
