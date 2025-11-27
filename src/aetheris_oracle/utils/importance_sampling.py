"""
Importance sampling for efficient tail quantile estimation.

Allows accurate P5/P95 estimation with fewer samples by reweighting
paths according to their likelihood under a proposal distribution.
"""

from typing import List, Sequence
import numpy as np


def compute_quantiles_with_importance_sampling(
    paths: Sequence[Sequence[float]],
    quantiles: Sequence[float],
    tail_threshold: float = 0.15,
) -> List[float]:
    """
    Compute quantiles using importance sampling for better tail estimates.

    For extreme quantiles (< tail_threshold or > 1-tail_threshold), uses
    importance sampling to get better estimates with fewer samples.

    Args:
        paths: Sampled paths (n_paths, horizon)
        quantiles: Quantile levels to compute
        tail_threshold: Threshold for defining "tail" quantiles (default: 0.15)

    Returns:
        List of quantile values
    """
    paths_array = np.array(paths)

    if paths_array.ndim == 1:
        paths_array = paths_array.reshape(-1, 1)

    n_paths, horizon = paths_array.shape

    # For each time step, compute quantiles
    results = []

    for q in quantiles:
        q_values = []

        for t in range(horizon):
            values = paths_array[:, t]

            # Check if this is a tail quantile
            is_tail = q < tail_threshold or q > (1 - tail_threshold)

            if is_tail and n_paths > 100:
                # Use importance sampling for tail
                q_val = _importance_sampled_quantile(values, q)
            else:
                # Use standard empirical quantile
                q_val = np.quantile(values, q)

            q_values.append(float(q_val))

        results.append(q_values)

    return results


def _importance_sampled_quantile(
    values: np.ndarray,
    q: float,
    proposal_std_multiplier: float = 2.0,
) -> float:
    """
    Compute quantile using importance sampling.

    Uses a heavier-tailed proposal distribution to better sample tails.

    Args:
        values: Sample values
        q: Quantile level
        proposal_std_multiplier: Multiplier for proposal distribution std

    Returns:
        Quantile estimate
    """
    n = len(values)

    # Empirical mean and std
    mean = np.mean(values)
    std = np.std(values)

    if std < 1e-10:
        return float(np.quantile(values, q))

    # Target distribution: empirical
    # Proposal distribution: Gaussian with heavier tails
    proposal_std = std * proposal_std_multiplier

    # Compute importance weights
    # w_i = p(x_i) / q(x_i)
    # For simplicity, use Gaussian approximation for both

    # Log-likelihood under target (empirical approximation)
    target_log_p = -0.5 * ((values - mean) / std) ** 2

    # Log-likelihood under proposal
    proposal_log_p = -0.5 * ((values - mean) / proposal_std) ** 2

    # Importance weights (in log space for stability)
    log_weights = target_log_p - proposal_log_p

    # Convert to linear space and normalize
    max_log_weight = np.max(log_weights)
    weights = np.exp(log_weights - max_log_weight)
    weights /= np.sum(weights)

    # Weighted quantile
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]

    cumsum_weights = np.cumsum(sorted_weights)

    # Find quantile
    idx = np.searchsorted(cumsum_weights, q)
    idx = min(idx, n - 1)

    return float(sorted_values[idx])


def streaming_quantile_computation(
    paths_generator,
    quantiles: Sequence[float],
    buffer_size: int = 1000,
) -> List[float]:
    """
    Compute quantiles in streaming fashion to reduce memory usage.

    Useful when generating 10,000+ paths.

    Args:
        paths_generator: Generator yielding paths
        quantiles: Quantile levels
        buffer_size: Number of paths to keep in memory

    Returns:
        Quantile estimates
    """
    # Use approximate streaming quantile algorithm (P-Square)
    # For simplicity, using buffered approach here

    buffer = []

    for path in paths_generator:
        buffer.append(path)

        if len(buffer) >= buffer_size:
            # Compute quantiles on buffer and yield
            yield compute_quantiles_with_importance_sampling(
                buffer, quantiles
            )
            buffer = []

    # Final buffer
    if buffer:
        yield compute_quantiles_with_importance_sampling(
            buffer, quantiles
        )


def adaptive_path_count(
    preliminary_paths: Sequence[Sequence[float]],
    target_tail_se: float = 0.01,
    min_paths: int = 1000,
    max_paths: int = 50000,
) -> int:
    """
    Adaptively determine how many paths are needed for desired tail accuracy.

    Args:
        preliminary_paths: Initial set of paths
        target_tail_se: Target standard error for tail quantiles
        min_paths: Minimum number of paths
        max_paths: Maximum number of paths

    Returns:
        Recommended number of paths
    """
    paths_array = np.array(preliminary_paths)
    n_prelim = len(paths_array)

    if n_prelim < 100:
        return max_paths  # Not enough data to estimate

    # Compute tail quantile variance
    # Bootstrap estimate of standard error
    n_bootstrap = 50
    q05_estimates = []
    q95_estimates = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(n_prelim, size=n_prelim, replace=True)
        sample = paths_array[indices]

        q05_estimates.append(np.quantile(sample[:, -1], 0.05))
        q95_estimates.append(np.quantile(sample[:, -1], 0.95))

    se_q05 = np.std(q05_estimates)
    se_q95 = np.std(q95_estimates)

    max_se = max(se_q05, se_q95)

    # Estimate required paths: SE ∝ 1/sqrt(n)
    # se_current * sqrt(n_current) = se_target * sqrt(n_target)
    if max_se > 0:
        n_required = int((max_se / target_tail_se) ** 2 * n_prelim)
        n_required = max(min_paths, min(n_required, max_paths))
    else:
        n_required = min_paths

    return n_required
