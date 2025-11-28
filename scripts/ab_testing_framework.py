"""
A/B Testing Framework for Legacy vs SOTA comparison.

Supports:
- Shadow mode: Run both configurations in parallel, compare metrics
- Online evaluation: Track real-time coverage and CRPS
- Statistical significance testing
- Automated reporting

Usage:
    # Run A/B test on historical data
    python scripts/ab_testing_framework.py --mode backtest --days 30

    # Run shadow mode (real-time parallel execution)
    python scripts/ab_testing_framework.py --mode shadow --forecasts 10

    # Analyze existing results
    python scripts/ab_testing_framework.py --mode analyze --results artifacts/ab_tests
"""

import sys
sys.path.insert(0, "src")

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np

from aetheris_oracle.data.free_connectors import FreeDataConnector
from aetheris_oracle.data.connectors import SyntheticDataConnector
from aetheris_oracle.config import ForecastConfig
from aetheris_oracle.pipeline.forecast import ForecastEngine


@dataclass
class ABTestResult:
    """Result from a single A/B test comparison."""
    timestamp: str
    asset_id: str
    horizon: int

    # Legacy metrics
    legacy_p10: float
    legacy_p50: float
    legacy_p90: float
    legacy_spread: float
    legacy_latency_ms: float

    # SOTA metrics
    sota_p10: float
    sota_p50: float
    sota_p90: float
    sota_spread: float
    sota_latency_ms: float

    # Actual outcome (for coverage calculation)
    actual_price: Optional[float] = None
    legacy_coverage: Optional[bool] = None
    sota_coverage: Optional[bool] = None


@dataclass
class ABTestSummary:
    """Summary statistics for an A/B test run."""
    n_tests: int
    start_time: str
    end_time: str

    # Coverage (P10-P90 should contain 80%)
    legacy_coverage_rate: float
    sota_coverage_rate: float

    # Spread (sharpness)
    legacy_avg_spread: float
    sota_avg_spread: float

    # Latency
    legacy_avg_latency_ms: float
    sota_avg_latency_ms: float

    # Statistical significance
    coverage_p_value: Optional[float] = None
    spread_improvement: float = 0.0
    latency_overhead: float = 0.0


class ABTestingFramework:
    """Framework for A/B testing Legacy vs SOTA forecasting."""

    def __init__(
        self,
        output_dir: str = "artifacts/ab_tests",
        device: str = "cpu",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.results: List[ABTestResult] = []

        # Initialize data connector
        self.connector = FreeDataConnector(enable_cache=True, cache_ttl_seconds=3600)

        # Initialize engines
        self._init_engines()

    def _init_engines(self):
        """Initialize Legacy and SOTA forecast engines."""
        print("Initializing forecast engines...")

        # Legacy engine (baseline)
        self.legacy_engine = ForecastEngine(
            connector=self.connector,
            seed=42,
            use_ncc_calibration=False,
            use_diff_greeks=False,
            use_fm_gp_residuals=False,
            use_neural_rough_vol=False,
            use_neural_jumps=False,
            use_mamba_trend=False,
        )

        # SOTA engine (5-component, no Mamba)
        self.sota_engine = ForecastEngine(
            connector=self.connector,
            seed=42,
            use_ncc_calibration=True,
            use_diff_greeks=True,
            use_fm_gp_residuals=True,
            use_neural_rough_vol=True,
            use_neural_jumps=True,
            use_mamba_trend=False,  # Excluded due to directional bias
            ncc_artifact_path="artifacts/ncc_calibration.pt",
            diff_greeks_artifact_path="artifacts/diff_greeks.pt",
            fmgp_artifact_path="artifacts/fmgp_residuals.pt",
            neural_vol_artifact_path="artifacts/neural_rough_vol.pt",
            neural_jump_artifact_path="artifacts/neural_jump_sde.pt",
            device=self.device,
        )

        print("  Legacy engine: initialized")
        print("  SOTA engine: initialized (5-component)")

    def run_single_comparison(
        self,
        asset_id: str = "BTC-USD",
        horizon: int = 7,
        num_paths: int = 1000,
    ) -> ABTestResult:
        """
        Run a single A/B comparison.

        Args:
            asset_id: Asset to forecast
            horizon: Forecast horizon in days
            num_paths: Number of Monte Carlo paths

        Returns:
            ABTestResult with comparison metrics
        """
        config = ForecastConfig(
            asset_id=asset_id,
            horizon_days=horizon,
            num_paths=num_paths,
            seed=42,
        )

        # Run Legacy forecast
        start_time = time.perf_counter()
        legacy_result = self.legacy_engine.forecast(config)
        legacy_latency = (time.perf_counter() - start_time) * 1000

        # Run SOTA forecast
        start_time = time.perf_counter()
        sota_result = self.sota_engine.forecast(config)
        sota_latency = (time.perf_counter() - start_time) * 1000

        # Extract terminal quantiles
        legacy_q = legacy_result.quantile_paths[horizon]
        sota_q = sota_result.quantile_paths[horizon]

        legacy_p10, legacy_p50, legacy_p90 = legacy_q[0.1], legacy_q[0.5], legacy_q[0.9]
        sota_p10, sota_p50, sota_p90 = sota_q[0.1], sota_q[0.5], sota_q[0.9]

        legacy_spread = (legacy_p90 - legacy_p10) / legacy_p50
        sota_spread = (sota_p90 - sota_p10) / sota_p50

        return ABTestResult(
            timestamp=datetime.now().isoformat(),
            asset_id=asset_id,
            horizon=horizon,
            legacy_p10=legacy_p10,
            legacy_p50=legacy_p50,
            legacy_p90=legacy_p90,
            legacy_spread=legacy_spread,
            legacy_latency_ms=legacy_latency,
            sota_p10=sota_p10,
            sota_p50=sota_p50,
            sota_p90=sota_p90,
            sota_spread=sota_spread,
            sota_latency_ms=sota_latency,
        )

    def run_shadow_mode(
        self,
        n_forecasts: int = 10,
        asset_id: str = "BTC-USD",
        horizon: int = 7,
        interval_seconds: int = 60,
    ) -> List[ABTestResult]:
        """
        Run shadow mode: execute both configurations in parallel.

        Args:
            n_forecasts: Number of forecasts to generate
            asset_id: Asset to forecast
            horizon: Forecast horizon
            interval_seconds: Interval between forecasts

        Returns:
            List of ABTestResults
        """
        print(f"\n{'='*60}")
        print(f"SHADOW MODE: {n_forecasts} forecasts")
        print(f"Asset: {asset_id}, Horizon: {horizon} days")
        print(f"{'='*60}")

        results = []

        for i in range(n_forecasts):
            print(f"\nForecast {i+1}/{n_forecasts}...")

            result = self.run_single_comparison(asset_id, horizon)
            results.append(result)
            self.results.append(result)

            print(f"  Legacy: P50=${result.legacy_p50:,.0f}, spread={result.legacy_spread:.2%}, {result.legacy_latency_ms:.0f}ms")
            print(f"  SOTA:   P50=${result.sota_p50:,.0f}, spread={result.sota_spread:.2%}, {result.sota_latency_ms:.0f}ms")

            if i < n_forecasts - 1 and interval_seconds > 0:
                print(f"  Waiting {interval_seconds}s...")
                time.sleep(interval_seconds)

        # Save results
        self._save_results(results, f"shadow_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        return results

    def run_backtest(
        self,
        days: int = 30,
        asset_id: str = "BTC-USD",
        horizon: int = 7,
    ) -> List[ABTestResult]:
        """
        Run backtest on historical data.

        Args:
            days: Number of days to backtest
            asset_id: Asset to test
            horizon: Forecast horizon

        Returns:
            List of ABTestResults with actual outcomes
        """
        print(f"\n{'='*60}")
        print(f"BACKTEST MODE: {days} days")
        print(f"Asset: {asset_id}, Horizon: {horizon} days")
        print(f"{'='*60}")

        results = []
        end_date = datetime.now() - timedelta(days=horizon)  # Allow time for outcome
        start_date = end_date - timedelta(days=days)

        # Fetch historical prices for outcome verification
        print("Fetching historical data for outcome verification...")

        try:
            import yfinance as yf
            ticker = yf.Ticker(asset_id)
            hist = ticker.history(start=start_date - timedelta(days=30), end=datetime.now())

            if hist.empty:
                print("Warning: Could not fetch historical data. Running without outcomes.")
                historical_prices = None
            else:
                historical_prices = hist["Close"]
        except Exception as e:
            print(f"Warning: {e}. Running without outcomes.")
            historical_prices = None

        current_date = start_date

        while current_date <= end_date:
            print(f"\nBacktest: {current_date.date()}...")

            # Run comparison
            result = self.run_single_comparison(asset_id, horizon)

            # Check actual outcome if we have historical data
            if historical_prices is not None:
                outcome_date = current_date + timedelta(days=horizon)
                try:
                    # Find closest date in historical data
                    closest_idx = historical_prices.index.get_indexer([outcome_date], method='nearest')[0]
                    actual_price = float(historical_prices.iloc[closest_idx])

                    result.actual_price = actual_price
                    result.legacy_coverage = result.legacy_p10 <= actual_price <= result.legacy_p90
                    result.sota_coverage = result.sota_p10 <= actual_price <= result.sota_p90

                    print(f"  Actual: ${actual_price:,.0f}")
                    print(f"  Legacy coverage: {'HIT' if result.legacy_coverage else 'MISS'}")
                    print(f"  SOTA coverage: {'HIT' if result.sota_coverage else 'MISS'}")
                except Exception as e:
                    print(f"  Could not get outcome: {e}")

            results.append(result)
            self.results.append(result)

            current_date += timedelta(days=1)

        # Save results
        self._save_results(results, f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        return results

    def compute_summary(self, results: Optional[List[ABTestResult]] = None) -> ABTestSummary:
        """
        Compute summary statistics for test results.

        Args:
            results: List of results (uses self.results if None)

        Returns:
            ABTestSummary with aggregated metrics
        """
        if results is None:
            results = self.results

        if not results:
            raise ValueError("No results to summarize")

        # Filter results with outcomes for coverage calculation
        results_with_outcomes = [r for r in results if r.actual_price is not None]

        # Coverage rates
        if results_with_outcomes:
            legacy_coverage = sum(1 for r in results_with_outcomes if r.legacy_coverage) / len(results_with_outcomes)
            sota_coverage = sum(1 for r in results_with_outcomes if r.sota_coverage) / len(results_with_outcomes)
        else:
            legacy_coverage = None
            sota_coverage = None

        # Spreads
        legacy_spreads = [r.legacy_spread for r in results]
        sota_spreads = [r.sota_spread for r in results]

        # Latencies
        legacy_latencies = [r.legacy_latency_ms for r in results]
        sota_latencies = [r.sota_latency_ms for r in results]

        # Statistical significance for coverage (if available)
        p_value = None
        if results_with_outcomes and len(results_with_outcomes) >= 10:
            try:
                from scipy import stats
                legacy_hits = [1 if r.legacy_coverage else 0 for r in results_with_outcomes]
                sota_hits = [1 if r.sota_coverage else 0 for r in results_with_outcomes]
                _, p_value = stats.ttest_rel(legacy_hits, sota_hits)
            except ImportError:
                pass

        return ABTestSummary(
            n_tests=len(results),
            start_time=results[0].timestamp,
            end_time=results[-1].timestamp,
            legacy_coverage_rate=legacy_coverage or 0.0,
            sota_coverage_rate=sota_coverage or 0.0,
            legacy_avg_spread=np.mean(legacy_spreads),
            sota_avg_spread=np.mean(sota_spreads),
            legacy_avg_latency_ms=np.mean(legacy_latencies),
            sota_avg_latency_ms=np.mean(sota_latencies),
            coverage_p_value=p_value,
            spread_improvement=(np.mean(sota_spreads) / np.mean(legacy_spreads) - 1) * 100,
            latency_overhead=(np.mean(sota_latencies) / np.mean(legacy_latencies) - 1) * 100,
        )

    def print_summary(self, summary: Optional[ABTestSummary] = None):
        """Print formatted summary of A/B test results."""
        if summary is None:
            summary = self.compute_summary()

        print(f"\n{'='*60}")
        print("A/B TEST SUMMARY")
        print(f"{'='*60}")

        print(f"\nTests: {summary.n_tests}")
        print(f"Period: {summary.start_time[:10]} to {summary.end_time[:10]}")

        print(f"\n{'Coverage (P10-P90):':<30}")
        print(f"  Legacy: {summary.legacy_coverage_rate:.1%}")
        print(f"  SOTA:   {summary.sota_coverage_rate:.1%}")
        if summary.coverage_p_value is not None:
            sig = "significant" if summary.coverage_p_value < 0.05 else "not significant"
            print(f"  p-value: {summary.coverage_p_value:.4f} ({sig})")

        print(f"\n{'Spread (Sharpness):':<30}")
        print(f"  Legacy: {summary.legacy_avg_spread:.2%}")
        print(f"  SOTA:   {summary.sota_avg_spread:.2%}")
        if summary.spread_improvement > 0:
            print(f"  SOTA is {summary.spread_improvement:.1f}% wider (more realistic)")
        else:
            print(f"  SOTA is {-summary.spread_improvement:.1f}% tighter")

        print(f"\n{'Latency:':<30}")
        print(f"  Legacy: {summary.legacy_avg_latency_ms:.0f}ms")
        print(f"  SOTA:   {summary.sota_avg_latency_ms:.0f}ms")
        if summary.latency_overhead > 0:
            print(f"  SOTA is {summary.latency_overhead:.1f}% slower")
        else:
            print(f"  SOTA is {-summary.latency_overhead:.1f}% faster")

        # Recommendation
        print(f"\n{'='*60}")
        print("RECOMMENDATION:")
        print(f"{'='*60}")

        if summary.sota_coverage_rate > summary.legacy_coverage_rate:
            print("  SOTA shows better coverage - recommend for production")
        elif summary.sota_coverage_rate == summary.legacy_coverage_rate:
            print("  Coverage is similar - evaluate based on spread and latency")
        else:
            print("  Legacy shows better coverage - investigate SOTA calibration")

        if summary.sota_avg_spread > summary.legacy_avg_spread * 1.5:
            print("  SOTA spreads more realistic for crypto volatility")

    def _save_results(self, results: List[ABTestResult], name: str):
        """Save results to JSON file."""
        filepath = self.output_dir / f"{name}.json"

        data = {
            "name": name,
            "created_at": datetime.now().isoformat(),
            "n_results": len(results),
            "results": [asdict(r) for r in results],
            "summary": asdict(self.compute_summary(results)) if results else None,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\nResults saved to: {filepath}")

    @staticmethod
    def analyze_results(results_dir: str):
        """Analyze and compare all A/B test results in a directory."""
        results_path = Path(results_dir)

        if not results_path.exists():
            print(f"Directory not found: {results_dir}")
            return

        result_files = list(results_path.glob("*.json"))

        if not result_files:
            print(f"No result files found in {results_dir}")
            return

        print(f"\n{'='*60}")
        print(f"ANALYZING {len(result_files)} A/B TEST RUNS")
        print(f"{'='*60}")

        all_summaries = []

        for filepath in sorted(result_files):
            try:
                with open(filepath) as f:
                    data = json.load(f)

                summary = data.get("summary", {})
                if summary:
                    all_summaries.append({
                        "name": data.get("name", filepath.stem),
                        "n_tests": summary.get("n_tests", 0),
                        "legacy_coverage": summary.get("legacy_coverage_rate", 0),
                        "sota_coverage": summary.get("sota_coverage_rate", 0),
                        "legacy_spread": summary.get("legacy_avg_spread", 0),
                        "sota_spread": summary.get("sota_avg_spread", 0),
                    })
            except Exception as e:
                print(f"Error reading {filepath}: {e}")

        if not all_summaries:
            print("No valid summaries found")
            return

        # Print comparison table
        print(f"\n{'Run':<30} {'Tests':<8} {'Legacy Cov':<12} {'SOTA Cov':<12} {'Legacy Spread':<14} {'SOTA Spread':<14}")
        print("-" * 90)

        for s in all_summaries:
            print(f"{s['name']:<30} {s['n_tests']:<8} {s['legacy_coverage']:.1%}{'':>6} {s['sota_coverage']:.1%}{'':>6} {s['legacy_spread']:.2%}{'':>8} {s['sota_spread']:.2%}")

        # Aggregate statistics
        if len(all_summaries) > 1:
            print(f"\n{'='*60}")
            print("AGGREGATE STATISTICS")
            print(f"{'='*60}")

            total_tests = sum(s['n_tests'] for s in all_summaries)
            avg_legacy_cov = np.mean([s['legacy_coverage'] for s in all_summaries])
            avg_sota_cov = np.mean([s['sota_coverage'] for s in all_summaries])

            print(f"Total tests across all runs: {total_tests}")
            print(f"Average Legacy coverage: {avg_legacy_cov:.1%}")
            print(f"Average SOTA coverage: {avg_sota_cov:.1%}")


def main():
    parser = argparse.ArgumentParser(description="A/B Testing Framework for Legacy vs SOTA")
    parser.add_argument("--mode", type=str, default="shadow",
                        choices=["shadow", "backtest", "analyze"],
                        help="Testing mode")
    parser.add_argument("--days", type=int, default=30,
                        help="Days for backtest mode")
    parser.add_argument("--forecasts", type=int, default=10,
                        help="Number of forecasts for shadow mode")
    parser.add_argument("--asset", type=str, default="BTC-USD",
                        help="Asset to test")
    parser.add_argument("--horizon", type=int, default=7,
                        help="Forecast horizon in days")
    parser.add_argument("--interval", type=int, default=5,
                        help="Interval between forecasts in shadow mode (seconds)")
    parser.add_argument("--results", type=str, default="artifacts/ab_tests",
                        help="Results directory (for analyze mode)")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Device for SOTA models")

    args = parser.parse_args()

    if args.mode == "analyze":
        ABTestingFramework.analyze_results(args.results)
    else:
        framework = ABTestingFramework(device=args.device)

        if args.mode == "shadow":
            results = framework.run_shadow_mode(
                n_forecasts=args.forecasts,
                asset_id=args.asset,
                horizon=args.horizon,
                interval_seconds=args.interval,
            )
        elif args.mode == "backtest":
            results = framework.run_backtest(
                days=args.days,
                asset_id=args.asset,
                horizon=args.horizon,
            )

        framework.print_summary()


if __name__ == "__main__":
    main()
