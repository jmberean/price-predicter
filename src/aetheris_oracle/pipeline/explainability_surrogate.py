import random
from typing import Dict, List, Tuple


class SurrogateExplainer:
    """Approximate attribution via simple permutation over aggregate features."""

    def __init__(self, seed: int | None = None, num_samples: int = 8) -> None:
        self._rng = random.Random(seed)
        self.num_samples = num_samples

    def explain(
        self, features: Dict[str, float], quantile_paths: Dict[int, Dict[float, float]]
    ) -> List[Tuple[str, float]]:
        if not features:
            return []
        base_width = self._cone_width(quantile_paths)
        scores: Dict[str, float] = {}
        norm = sum(abs(v) for v in features.values()) or 1.0
        for feature, value in features.items():
            rel = abs(value) / norm
            # Estimate sensitivity by perturbing cone width proportionally
            impacts = []
            for _ in range(self.num_samples):
                noise = 0.1 * (self._rng.random() - 0.5)
                impacts.append(abs(base_width * rel * (1 + noise)))
            scores[feature] = sum(impacts) / max(len(impacts), 1)
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]

    def _cone_width(self, quantile_paths: Dict[int, Dict[float, float]]) -> float:
        if not quantile_paths:
            return 0.0
        last = quantile_paths[max(quantile_paths.keys())]
        upper = last.get(0.9, 0.0)
        lower = last.get(0.1, 0.0)
        return upper - lower
