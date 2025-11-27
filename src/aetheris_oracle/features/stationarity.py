import math
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class NormalizationStats:
    mean: float
    std: float

    def denormalize(self, value: float) -> float:
        return value * self.std + self.mean


class StationarityNormalizer:
    """RevIN-like normalizer using trailing window statistics."""

    def fit(self, series: List[float]) -> NormalizationStats:
        if not series:
            raise ValueError("series must not be empty")
        mean = sum(series) / len(series)
        variance = sum((x - mean) ** 2 for x in series) / max(len(series) - 1, 1)
        std = math.sqrt(variance) or 1.0
        return NormalizationStats(mean=mean, std=std)

    def normalize(self, series: List[float], stats: NormalizationStats) -> List[float]:
        return [(x - stats.mean) / stats.std for x in series]

    def denormalize_series(
        self, series: List[float], stats: NormalizationStats
    ) -> List[float]:
        return [stats.denormalize(x) for x in series]

    def normalize_and_stats(self, series: List[float]) -> Tuple[List[float], NormalizationStats]:
        stats = self.fit(series)
        return self.normalize(series, stats), stats
