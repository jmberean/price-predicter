"""
Visualization module for forecast results.

Creates dynamic charts showing:
- Historical price data
- Forecast cone (P5-P95 quantiles)
- Median forecast path
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


def plot_forecast_cone(
    historical_timestamps: List[datetime],
    historical_prices: List[float],
    forecast_start: datetime,
    quantiles_by_day: Dict[int, Dict[float, float]],
    asset_id: str = "BTC-USD",
    save_path: Optional[Path] = None,
    show: bool = True,
) -> Path:
    """
    Create a forecast cone visualization.

    Args:
        historical_timestamps: List of historical timestamps
        historical_prices: List of historical prices
        forecast_start: Start timestamp of forecast
        quantiles_by_day: Dict mapping day offset to quantile dict
            e.g., {1: {0.05: 30000, 0.5: 31000, 0.95: 32000}, ...}
        asset_id: Asset identifier (for title)
        save_path: Path to save image (default: forecast_YYYYMMDD_HHMMSS.png)
        show: Whether to display the plot

    Returns:
        Path to saved image
    """
    # Prepare forecast data
    forecast_days = sorted(quantiles_by_day.keys())
    forecast_timestamps = [forecast_start + timedelta(days=d) for d in forecast_days]

    # Extract quantiles
    p05 = [quantiles_by_day[d][0.05] for d in forecast_days]
    p10 = [quantiles_by_day[d][0.1] for d in forecast_days]
    p25 = [quantiles_by_day[d][0.25] for d in forecast_days]
    p50 = [quantiles_by_day[d][0.5] for d in forecast_days]
    p75 = [quantiles_by_day[d][0.75] for d in forecast_days]
    p90 = [quantiles_by_day[d][0.9] for d in forecast_days]
    p95 = [quantiles_by_day[d][0.95] for d in forecast_days]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot historical prices
    ax.plot(
        historical_timestamps,
        historical_prices,
        color="#2E86AB",
        linewidth=2,
        label="Historical Price",
        zorder=5,
    )

    # Add connection from last historical to first forecast
    if historical_timestamps and forecast_timestamps:
        ax.plot(
            [historical_timestamps[-1], forecast_timestamps[0]],
            [historical_prices[-1], p50[0]],
            color="#A23B72",
            linewidth=2,
            linestyle="--",
            alpha=0.5,
        )

    # Plot median forecast
    ax.plot(
        forecast_timestamps,
        p50,
        color="#A23B72",
        linewidth=3,
        label="Forecast (Median)",
        zorder=5,
    )

    # Plot forecast cone (fill between quantiles)
    # P10-P90 (80% confidence)
    ax.fill_between(
        forecast_timestamps,
        p10,
        p90,
        color="#F18F01",
        alpha=0.3,
        label="80% Confidence (P10-P90)",
        zorder=2,
    )

    # P25-P75 (50% confidence)
    ax.fill_between(
        forecast_timestamps,
        p25,
        p75,
        color="#C73E1D",
        alpha=0.4,
        label="50% Confidence (P25-P75)",
        zorder=3,
    )

    # P5-P95 outer bounds (faint)
    ax.plot(
        forecast_timestamps, p05, color="#999", linewidth=1, linestyle=":", alpha=0.6, zorder=1
    )
    ax.plot(
        forecast_timestamps, p95, color="#999", linewidth=1, linestyle=":", alpha=0.6, zorder=1
    )
    ax.fill_between(
        forecast_timestamps,
        p05,
        p95,
        color="#999",
        alpha=0.1,
        label="90% Confidence (P5-P95)",
        zorder=1,
    )

    # Styling
    ax.set_xlabel("Date", fontsize=12, fontweight="bold")
    ax.set_ylabel("Price (USD)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"{asset_id} Price Forecast\n"
        f"Historical + {len(forecast_days)}-Day Probabilistic Forecast Cone",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(forecast_days) // 7)))
    plt.xticks(rotation=45, ha="right")

    # Format y-axis with currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

    # Grid
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    # Legend
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

    # Add vertical line at forecast start
    ax.axvline(
        x=forecast_start,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="Forecast Start",
    )

    # Add annotation for current price
    if historical_timestamps and historical_prices:
        last_price = historical_prices[-1]
        ax.annotate(
            f"Current: ${last_price:,.0f}",
            xy=(historical_timestamps[-1], last_price),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#2E86AB", alpha=0.7, edgecolor="white"),
            color="white",
            fontweight="bold",
            fontsize=10,
        )

    # Add annotation for forecast median
    if forecast_timestamps and p50:
        last_forecast = p50[-1]
        change_pct = ((last_forecast - historical_prices[-1]) / historical_prices[-1]) * 100
        color = "#00A650" if change_pct > 0 else "#E63946"
        ax.annotate(
            f"Forecast: ${last_forecast:,.0f} ({change_pct:+.1f}%)",
            xy=(forecast_timestamps[-1], last_forecast),
            xytext=(-10, -30),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.8, edgecolor="white"),
            color="white",
            fontweight="bold",
            fontsize=10,
            ha="right",
        )

    # Tight layout
    plt.tight_layout()

    # Save
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = Path(f"forecast_{asset_id.replace('/', '_')}_{timestamp}.png")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    return save_path


def quick_plot_from_cli_result(
    result_dict: Dict,
    asset_id: str = "BTC-USD",
    lookback_days: int = 30,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> Path:
    """
    Quick plot from CLI forecast result.

    Args:
        result_dict: Result from ForecastEngine.forecast()
        asset_id: Asset identifier
        lookback_days: Days of historical data to show
        save_path: Path to save image
        show: Whether to display the plot

    Returns:
        Path to saved image
    """
    # Extract quantiles_by_day from result
    # quantile_paths is already a dict: {1: {0.05: ..., 0.5: ..., 0.95: ...}, 2: {...}, ...}
    quantiles_by_day = result_dict["quantile_paths"]

    # Get historical data (need to fetch it)
    from .data.free_connectors import FreeDataConnector

    connector = FreeDataConnector()
    forecast_start = datetime.fromisoformat(result_dict["metadata"]["as_of"])
    window_start = forecast_start - timedelta(days=lookback_days)

    try:
        frame = connector.fetch_window(asset_id, forecast_start, timedelta(days=lookback_days))
        historical_timestamps = frame.timestamps
        historical_prices = frame.closes
    except Exception:
        # Fallback: just use the forecast
        historical_timestamps = []
        historical_prices = []

    return plot_forecast_cone(
        historical_timestamps=historical_timestamps,
        historical_prices=historical_prices,
        forecast_start=forecast_start,
        quantiles_by_day=quantiles_by_day,
        asset_id=asset_id,
        save_path=save_path,
        show=show,
    )
