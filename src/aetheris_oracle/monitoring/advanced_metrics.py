"""
Advanced metrics for probabilistic forecasting evaluation.

Implements:
- QICE (Quantile Interval Coverage Error)
- Conditional FID (Fréchet Inception Distance for time series)
- ProbCorr (Probabilistic Correlation)
- CRPS (Continuous Ranked Probability Score)
- Sharpness metrics
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from scipy.linalg import sqrtm
from scipy.spatial.distance import cdist


def compute_crps(
    predictions: Sequence[float], actual: float
) -> float:
    """
    Continuous Ranked Probability Score.

    CRPS = E|X - y| - 0.5 * E|X - X'|
    where X, X' are independent samples from forecast distribution, y is actual.

    Args:
        predictions: Ensemble of predictions
        actual: Realized value

    Returns:
        CRPS value (lower is better)
    """
    predictions = np.array(predictions)
    n = len(predictions)

    # E|X - y|
    term1 = np.mean(np.abs(predictions - actual))

    # E|X - X'| (pairwise differences)
    if n > 1:
        pairwise_diffs = np.abs(predictions[:, None] - predictions[None, :])
        term2 = 0.5 * np.mean(pairwise_diffs)
    else:
        term2 = 0.0

    crps = term1 - term2
    return float(crps)


def compute_qice(
    predicted_quantiles: Dict[float, float],
    actual: float,
    quantile_levels: Sequence[float],
) -> Dict[str, float]:
    """
    Quantile Interval Coverage Error.

    Measures per-quantile calibration error.

    Args:
        predicted_quantiles: {quantile_level: predicted_value}
        actual: Realized value
        quantile_levels: List of quantile levels to evaluate

    Returns:
        Dictionary of QICE metrics
    """
    qice_values = {}

    for q in quantile_levels:
        if q not in predicted_quantiles:
            continue

        pred_q = predicted_quantiles[q]

        # Check if actual is below predicted quantile
        indicator = 1.0 if actual <= pred_q else 0.0

        # QICE: |indicator - target_quantile|
        error = abs(indicator - q)
        qice_values[f"qice_{q}"] = error

    # Overall QICE
    if qice_values:
        qice_values["qice_mean"] = np.mean(list(qice_values.values()))

    return qice_values


def compute_conditional_fid(
    generated_paths: List[List[float]],
    real_paths: List[List[float]],
    conditioning: Optional[np.ndarray] = None,
) -> float:
    """
    Conditional Fréchet Inception Distance for time series.

    Measures distributional similarity between generated and real paths,
    optionally conditioned on features.

    FID = ||μ_1 - μ_2||^2 + Tr(Σ_1 + Σ_2 - 2*sqrt(Σ_1*Σ_2))

    Args:
        generated_paths: Sampled forecast paths
        real_paths: Historical paths
        conditioning: Optional conditioning features

    Returns:
        FID value (lower is better)
    """
    generated = np.array(generated_paths)
    real = np.array(real_paths)

    # Extract statistical features
    gen_features = _extract_path_features(generated)
    real_features = _extract_path_features(real)

    # If conditioning provided, concatenate
    if conditioning is not None:
        gen_cond = np.tile(conditioning, (gen_features.shape[0], 1))
        real_cond = np.tile(conditioning, (real_features.shape[0], 1))
        gen_features = np.hstack([gen_features, gen_cond])
        real_features = np.hstack([real_features, real_cond])

    # Compute means and covariances
    mu_gen = np.mean(gen_features, axis=0)
    mu_real = np.mean(real_features, axis=0)

    sigma_gen = np.cov(gen_features, rowvar=False)
    sigma_real = np.cov(real_features, rowvar=False)

    # Handle 1D case
    if sigma_gen.ndim == 0:
        sigma_gen = np.array([[sigma_gen]])
        sigma_real = np.array([[sigma_real]])

    # Compute FID
    mean_diff = np.sum((mu_gen - mu_real) ** 2)

    # Matrix square root
    covmean = sqrtm(sigma_gen @ sigma_real)

    # Numerical stability
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    trace_term = np.trace(sigma_gen + sigma_real - 2 * covmean)

    fid = mean_diff + trace_term
    return float(fid)


def _extract_path_features(paths: np.ndarray) -> np.ndarray:
    """
    Extract statistical features from paths for FID computation.

    Features:
    - Mean, std, min, max
    - Autocorrelation at lag 1
    - Skewness, kurtosis
    """
    n_paths, horizon = paths.shape

    features = []

    for path in paths:
        feat = [
            np.mean(path),
            np.std(path),
            np.min(path),
            np.max(path),
        ]

        # Autocorrelation at lag 1
        if horizon > 1:
            autocorr = np.corrcoef(path[:-1], path[1:])[0, 1]
            feat.append(autocorr if not np.isnan(autocorr) else 0.0)
        else:
            feat.append(0.0)

        # Skewness and kurtosis
        if horizon > 2:
            feat.append(stats.skew(path))
            feat.append(stats.kurtosis(path))
        else:
            feat.extend([0.0, 0.0])

        features.append(feat)

    return np.array(features)


def compute_prob_corr(
    predicted_paths: List[List[float]],
    actual_paths: List[List[float]],
    lag: int = 1,
) -> Dict[str, float]:
    """
    Probabilistic Correlation.

    Measures how well the forecast ensemble captures temporal correlations
    in the actual data.

    Args:
        predicted_paths: Ensemble of forecast paths
        actual_paths: Realized paths
        lag: Lag for autocorrelation

    Returns:
        Dictionary of correlation metrics
    """
    predicted = np.array(predicted_paths)
    actual = np.array(actual_paths)

    # Compute autocorrelations
    pred_autocorrs = []
    for path in predicted:
        if len(path) > lag:
            ac = np.corrcoef(path[:-lag], path[lag:])[0, 1]
            if not np.isnan(ac):
                pred_autocorrs.append(ac)

    actual_autocorrs = []
    for path in actual:
        if len(path) > lag:
            ac = np.corrcoef(path[:-lag], path[lag:])[0, 1]
            if not np.isnan(ac):
                actual_autocorrs.append(ac)

    if not pred_autocorrs or not actual_autocorrs:
        return {"prob_corr": 0.0, "autocorr_error": 1.0}

    # Mean autocorrelations
    mean_pred_ac = np.mean(pred_autocorrs)
    mean_actual_ac = np.mean(actual_autocorrs)

    # Std of autocorrelations (captures uncertainty)
    std_pred_ac = np.std(pred_autocorrs)
    std_actual_ac = np.std(actual_autocorrs)

    # ProbCorr: correlation between distributions
    autocorr_error = abs(mean_pred_ac - mean_actual_ac)
    std_error = abs(std_pred_ac - std_actual_ac)

    prob_corr = 1.0 / (1.0 + autocorr_error + 0.5 * std_error)

    return {
        "prob_corr": float(prob_corr),
        "autocorr_error": float(autocorr_error),
        "pred_autocorr_mean": float(mean_pred_ac),
        "pred_autocorr_std": float(std_pred_ac),
        "actual_autocorr_mean": float(mean_actual_ac),
        "actual_autocorr_std": float(std_actual_ac),
    }


def compute_sharpness(
    quantile_paths: Dict[int, Dict[float, float]],
    lower_q: float = 0.1,
    upper_q: float = 0.9,
) -> Dict[str, float]:
    """
    Compute sharpness metrics (prediction interval width).

    Sharpness measures the concentration of the predictive distribution.
    Sharper forecasts are better, but only if calibrated.

    Args:
        quantile_paths: {horizon: {quantile: value}}
        lower_q: Lower quantile for interval
        upper_q: Upper quantile for interval

    Returns:
        Dictionary of sharpness metrics
    """
    widths = []

    for horizon, quantiles in quantile_paths.items():
        if lower_q in quantiles and upper_q in quantiles:
            width = quantiles[upper_q] - quantiles[lower_q]
            widths.append(width)

    if not widths:
        return {"mean_width": 0.0, "median_width": 0.0, "max_width": 0.0}

    return {
        "mean_width": float(np.mean(widths)),
        "median_width": float(np.median(widths)),
        "max_width": float(np.max(widths)),
        "min_width": float(np.min(widths)),
        "width_std": float(np.std(widths)),
    }


def compute_quantile_crossing_rate(
    quantile_paths: Dict[int, Dict[float, float]]
) -> float:
    """
    Compute rate of quantile crossing violations.

    Quantiles must be monotonically increasing. This checks violations.

    Args:
        quantile_paths: {horizon: {quantile: value}}

    Returns:
        Crossing rate (should be 0)
    """
    total_checks = 0
    crossings = 0

    for horizon, quantiles in quantile_paths.items():
        sorted_q = sorted(quantiles.items())

        for i in range(len(sorted_q) - 1):
            q1, val1 = sorted_q[i]
            q2, val2 = sorted_q[i + 1]

            total_checks += 1
            if val1 > val2:  # Violation
                crossings += 1

    if total_checks == 0:
        return 0.0

    return crossings / total_checks


def compute_energy_score(
    predicted_paths: List[List[float]],
    actual: List[float],
) -> float:
    """
    Energy Score: multivariate generalization of CRPS.

    ES = E||X - y|| - 0.5 * E||X - X'||
    where || || is Euclidean distance.

    Args:
        predicted_paths: Ensemble of forecast paths
        actual: Realized path

    Returns:
        Energy score (lower is better)
    """
    predicted = np.array(predicted_paths)
    actual = np.array(actual)

    # E||X - y||
    term1 = np.mean(np.linalg.norm(predicted - actual[None, :], axis=1))

    # E||X - X'||
    n = len(predicted)
    if n > 1:
        pairwise = cdist(predicted, predicted, metric="euclidean")
        term2 = 0.5 * np.mean(pairwise)
    else:
        term2 = 0.0

    es = term1 - term2
    return float(es)


class AdvancedMetricsCollector:
    """
    Collector for comprehensive probabilistic forecast evaluation.
    """

    def __init__(self):
        self.metrics_history: List[Dict[str, float]] = []

    def evaluate_forecast(
        self,
        predicted_paths: List[List[float]],
        quantile_paths: Dict[int, Dict[float, float]],
        actual: List[float],
        historical_paths: Optional[List[List[float]]] = None,
        conditioning: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Compute comprehensive metrics for a single forecast.

        Args:
            predicted_paths: Ensemble of forecast paths
            quantile_paths: Extracted quantile paths
            actual: Realized path
            historical_paths: Historical paths for FID
            conditioning: Conditioning features

        Returns:
            Dictionary of all metrics
        """
        metrics = {}

        # CRPS per horizon
        for horizon in range(len(actual)):
            horizon_preds = [path[horizon] for path in predicted_paths if horizon < len(path)]
            if horizon_preds:
                crps = compute_crps(horizon_preds, actual[horizon])
                metrics[f"crps_h{horizon+1}"] = crps

        # Mean CRPS
        crps_values = [v for k, v in metrics.items() if k.startswith("crps_h")]
        if crps_values:
            metrics["crps_mean"] = np.mean(crps_values)

        # Energy Score (multivariate)
        metrics["energy_score"] = compute_energy_score(predicted_paths, actual)

        # QICE
        for horizon, quantiles in quantile_paths.items():
            if horizon - 1 < len(actual):
                qice = compute_qice(
                    quantiles, actual[horizon - 1], list(quantiles.keys())
                )
                for k, v in qice.items():
                    metrics[f"{k}_h{horizon}"] = v

        # Sharpness
        sharpness = compute_sharpness(quantile_paths)
        metrics.update({f"sharpness_{k}": v for k, v in sharpness.items()})

        # Quantile crossing rate
        metrics["quantile_crossing_rate"] = compute_quantile_crossing_rate(quantile_paths)

        # ProbCorr
        if historical_paths and len(historical_paths) > 0:
            prob_corr = compute_prob_corr(predicted_paths, historical_paths)
            metrics.update({f"prob_corr_{k}": v for k, v in prob_corr.items()})

        # Conditional FID
        if historical_paths and len(historical_paths) > 0:
            try:
                fid = compute_conditional_fid(
                    predicted_paths, historical_paths, conditioning
                )
                metrics["conditional_fid"] = fid
            except Exception:
                # FID can fail with small samples
                metrics["conditional_fid"] = float("nan")

        self.metrics_history.append(metrics)
        return metrics

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics across all evaluated forecasts."""
        if not self.metrics_history:
            return {}

        summary = {}

        # Aggregate each metric
        all_keys = set()
        for m in self.metrics_history:
            all_keys.update(m.keys())

        for key in all_keys:
            values = [m.get(key) for m in self.metrics_history if key in m]
            values = [v for v in values if not np.isnan(v)]

            if values:
                summary[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "median": float(np.median(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }

        return summary

    def save_history(self, path: str) -> None:
        """Save metrics history to JSON."""
        import json
        from pathlib import Path

        Path(path).write_text(json.dumps(self.metrics_history, indent=2))

    def load_history(self, path: str) -> None:
        """Load metrics history from JSON."""
        import json
        from pathlib import Path

        self.metrics_history = json.loads(Path(path).read_text())
