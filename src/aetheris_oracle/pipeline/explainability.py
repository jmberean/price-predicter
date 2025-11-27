from typing import Dict, List, Tuple

from ..modules.market_maker import MarketMakerIndices


class ExplainabilityEngine:
    """Lightweight driver summary using simple sensitivity scores."""

    def summarize(
        self,
        regime_vector: List[float],
        mm_indices: MarketMakerIndices,
        vol_path: List[float],
    ) -> List[Tuple[str, float]]:
        vol_regime = regime_vector[0] if regime_vector else 0.0
        iv_level = regime_vector[1] if len(regime_vector) > 1 else 0.0
        iv_slope = regime_vector[2] if len(regime_vector) > 2 else 0.0
        drivers: Dict[str, float] = {
            "RegimeVolatility": vol_regime,
            "IVLevel": iv_level,
            "IVTermSlope": iv_slope,
            "GammaSqueeze": mm_indices.gamma_squeeze,
            "InventoryUnwind": mm_indices.inventory_unwind,
            "BasisPressure": mm_indices.basis_pressure,
            "VolTrend": vol_path[-1] - vol_path[0] if vol_path else 0.0,
        }
        sorted_drivers = sorted(drivers.items(), key=lambda kv: abs(kv[1]), reverse=True)
        return sorted_drivers[:3]
