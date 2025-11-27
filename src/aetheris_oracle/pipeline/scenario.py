from ..config import ScenarioOverrides
from ..data.schemas import MarketFeatureFrame


def apply_scenario(frame: MarketFeatureFrame, scenario: ScenarioOverrides) -> MarketFeatureFrame:
    """Returns a shallow adjusted copy for what-if runs with whitelist and clamps."""

    iv_mult = _clamp(scenario.iv_multiplier, 0.5, 3.0)
    funding_shift = _clamp(scenario.funding_shift, -0.05, 0.05)
    basis_shift = _clamp(scenario.basis_shift, -0.05, 0.05)
    iv_points = {k: v * iv_mult for k, v in frame.iv_points.items()}

    narrative_scores = dict(frame.narrative_scores)
    for key, value in scenario.narrative_overrides.items():
        if key not in narrative_scores:
            continue
        narrative_scores[key] = _clamp(value, 0.0, 1.0)

    validate_scenario(
        iv_mult=iv_mult,
        funding_shift=funding_shift,
        basis_shift=basis_shift,
        narrative_scores=narrative_scores,
    )

    return MarketFeatureFrame(
        timestamps=frame.timestamps,
        closes=frame.closes,
        volumes=frame.volumes,
        iv_points=iv_points,
        funding_rate=frame.funding_rate + funding_shift,
        basis=frame.basis + basis_shift,
        order_imbalance=frame.order_imbalance,
        narrative_scores=narrative_scores,
        skew=frame.skew,
    )


def _clamp(value: float, low: float, high: float) -> float:
    return max(min(value, high), low)


def validate_scenario(
    iv_mult: float,
    funding_shift: float,
    basis_shift: float,
    narrative_scores: dict,
) -> None:
    if iv_mult <= 0:
        raise ValueError("iv_multiplier must be positive")
    if abs(funding_shift) > 0.1 or abs(basis_shift) > 0.1:
        raise ValueError("funding/basis shift too large")
    if any(v < 0 or v > 1 for v in narrative_scores.values()):
        raise ValueError("narrative override outside [0,1]")
