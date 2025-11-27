from dataclasses import dataclass
from typing import Dict


@dataclass
class MarketMakerIndices:
    gamma_squeeze: float
    inventory_unwind: float
    basis_pressure: float


class MarketMakerEngine:
    """Summarizes derivatives/microstructure pressure into compact, deterministic scores."""

    def compute_indices(
        self,
        iv_term_structure: Dict[str, float],
        funding_rate: float,
        basis: float,
        order_imbalance: float,
        skew: float,
        option_oi_ratio: float | None = None,
        stablecoin_flow: float | None = None,
    ) -> MarketMakerIndices:
        iv7 = iv_term_structure.get("iv_7d_atm", 0.5)
        iv30 = iv_term_structure.get("iv_30d_atm", iv7)
        term_slope = iv30 - iv7
        gamma_squeeze = 1.5 * max(term_slope, 0) + max(order_imbalance, 0)
        if option_oi_ratio is not None:
            gamma_squeeze += 0.5 * option_oi_ratio
        inventory_unwind = max(-order_imbalance, 0) + max(skew, 0)
        if stablecoin_flow is not None:
            inventory_unwind += -stablecoin_flow
        basis_pressure = basis + funding_rate
        return MarketMakerIndices(
            gamma_squeeze=_clip(_zscore_like(gamma_squeeze)),
            inventory_unwind=_clip(_zscore_like(inventory_unwind)),
            basis_pressure=_clip(_zscore_like(basis_pressure)),
        )


def _clip(value: float, low: float = -3.0, high: float = 3.0) -> float:
    return max(min(value, high), low)


def _zscore_like(value: float, scale: float = 0.1) -> float:
    if scale <= 0:
        return value
    return value / scale
