import logging
from dataclasses import dataclass
from typing import Dict


@dataclass
class MetricsSink:
    """Interface for metrics emission; swap with real telemetry in prod."""

    def emit_forecast_metrics(
        self,
        latency_ms: float,
        asset_id: str,
        horizon: int,
        num_paths: int,
        regime_bucket: str,
        status: str = "ok",
    ) -> None:
        # In a real deployment, push to Prometheus/StatsD/etc.
        return

    def emit_error(self, name: str, detail: Dict[str, str]) -> None:
        return


class LoggingMetricsSink(MetricsSink):
    """Logs metrics to standard logging for dev/debug."""

    def __init__(self) -> None:
        self._logger = logging.getLogger("aetheris.metrics")

    def emit_forecast_metrics(
        self,
        latency_ms: float,
        asset_id: str,
        horizon: int,
        num_paths: int,
        regime_bucket: str,
        status: str = "ok",
    ) -> None:
        self._logger.info(
            "forecast_metrics latency_ms=%.2f asset=%s horizon=%s paths=%s regime=%s status=%s",
            latency_ms,
            asset_id,
            horizon,
            num_paths,
            regime_bucket,
            status,
        )

    def emit_error(self, name: str, detail: Dict[str, str]) -> None:
        self._logger.error("forecast_error name=%s detail=%s", name, detail)


__all__ = ["MetricsSink", "LoggingMetricsSink"]
