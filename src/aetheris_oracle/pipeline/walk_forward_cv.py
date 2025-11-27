"""
Walk-Forward Cross-Validation for Time-Series Training.

Implements proper time-series CV to prevent temporal overfitting.
Never uses future data for training.
"""

from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any
import numpy as np


class WalkForwardCV:
    """
    Walk-forward cross-validation for time series.

    Prevents temporal overfitting by:
    1. Never using future data for training
    2. Creating multiple train/validation splits over time
    3. Ensuring validation windows are always after training windows
    """

    def __init__(
        self,
        train_window_days: int = 120,
        val_window_days: int = 30,
        step_days: int = 30,
        min_train_days: int = 90,
    ):
        """
        Initialize walk-forward CV.

        Args:
            train_window_days: Size of training window in days
            val_window_days: Size of validation window in days
            step_days: How many days to step forward between folds
            min_train_days: Minimum training window size
        """
        self.train_window_days = train_window_days
        self.val_window_days = val_window_days
        self.step_days = step_days
        self.min_train_days = min_train_days

    def create_splits(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """
        Create walk-forward train/validation splits.

        Args:
            start_date: Earliest date in dataset
            end_date: Latest date in dataset

        Returns:
            List of (train_start, train_end, val_start, val_end) tuples
        """
        splits = []

        # Start with initial training window
        train_start = start_date
        train_end = start_date + timedelta(days=self.train_window_days)

        while True:
            # Validation window immediately follows training window
            val_start = train_end + timedelta(days=1)
            val_end = val_start + timedelta(days=self.val_window_days)

            # Stop if validation window exceeds available data
            if val_end > end_date:
                break

            # Ensure minimum training window size
            actual_train_days = (train_end - train_start).days
            if actual_train_days >= self.min_train_days:
                splits.append((train_start, train_end, val_start, val_end))

            # Step forward (expanding window approach)
            train_end = val_end

            # Optional: Use sliding window instead of expanding
            # Uncomment to use fixed-size sliding window:
            # train_start = train_start + timedelta(days=self.step_days)

        return splits

    def filter_data_by_dates(
        self,
        data: List[Tuple[datetime, Any, Any]],
        start_date: datetime,
        end_date: datetime,
    ) -> List[Tuple[datetime, Any, Any]]:
        """
        Filter historical data by date range.

        Args:
            data: List of (date, closes, features) tuples
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            Filtered data within date range
        """
        return [
            (date, closes, features)
            for date, closes, features in data
            if start_date <= date <= end_date
        ]

    def get_fold_info(self, fold_idx: int, total_folds: int) -> str:
        """Get human-readable fold information."""
        return f"Fold {fold_idx + 1}/{total_folds}"


def split_historical_data_cv(
    historical_data: List[Tuple[datetime, Any, Any]],
    train_ratio: float = 0.8,
    min_val_samples: int = 20,
) -> Tuple[List, List]:
    """
    Simple train/validation split for historical time-series data.

    Args:
        historical_data: List of (date, closes, features) tuples
        train_ratio: Fraction of data to use for training (0.8 = 80%)
        min_val_samples: Minimum number of validation samples

    Returns:
        (train_data, val_data) tuple
    """
    if len(historical_data) < min_val_samples * 2:
        # Not enough data, return all as training
        return historical_data, []

    # Calculate split point (ensuring minimum validation size)
    split_idx = int(len(historical_data) * train_ratio)
    split_idx = min(split_idx, len(historical_data) - min_val_samples)
    split_idx = max(split_idx, min_val_samples)

    train_data = historical_data[:split_idx]
    val_data = historical_data[split_idx:]

    return train_data, val_data


def validate_temporal_ordering(data: List[Tuple[datetime, Any, Any]]) -> bool:
    """
    Validate that data is sorted chronologically.

    Args:
        data: List of (date, closes, features) tuples

    Returns:
        True if properly ordered, False otherwise
    """
    if len(data) < 2:
        return True

    for i in range(len(data) - 1):
        if data[i][0] >= data[i + 1][0]:
            return False

    return True


def compute_cv_scores(
    fold_scores: List[float],
) -> Dict[str, float]:
    """
    Compute summary statistics across CV folds.

    Args:
        fold_scores: List of scores from each fold

    Returns:
        Dictionary with mean, std, min, max scores
    """
    if not fold_scores:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    return {
        "mean": float(np.mean(fold_scores)),
        "std": float(np.std(fold_scores)),
        "min": float(np.min(fold_scores)),
        "max": float(np.max(fold_scores)),
    }
