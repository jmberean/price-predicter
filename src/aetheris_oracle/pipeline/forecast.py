import os
import random
from dataclasses import dataclass, field
import time
from datetime import timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from ..config import ForecastConfig
from ..data.connectors import SyntheticDataConnector
from ..data.interfaces import DataConnector
from ..data.schemas import JumpPath, MarketFeatureFrame, RegimeVector, ResidualPaths, TrendPath, VolPath
from ..features.regime import compute_regime_vector
from ..features.stationarity import NormalizationStats, StationarityNormalizer
from ..monitoring import LoggingMetricsSink, MetricsSink
from ..modules.jump import JumpModel
from ..modules.market_maker import MarketMakerEngine, MarketMakerIndices
from ..modules.residual import ResidualGenerator
from ..modules.trend import TrendEnsemble
from ..modules.vol_path import VolPathEngine
from ..utils.device import get_available_device
from .calibration import CalibrationEngine
from .explainability import ExplainabilityEngine
from .explainability_surrogate import SurrogateExplainer
from .scenario import apply_scenario


# Explainability cone width contribution weights
# These weights determine how much each factor contributes to forecast uncertainty
CONE_WIDTH_VOL_WEIGHT = 0.4  # Volatility is the primary driver of cone width
CONE_WIDTH_IV_WEIGHT = 0.3  # Implied volatility indicates market expectations
CONE_WIDTH_GAMMA_WEIGHT = 0.2  # Gamma squeeze affects short-term uncertainty
CONE_WIDTH_SKEW_WEIGHT = 0.1  # Skew captures tail risk asymmetry


def _getenv_bool(key: str, default: bool = False) -> bool:
    """Helper to read boolean from environment variable."""
    val = os.getenv(key, "").lower()
    if val in ("true", "1", "yes"):
        return True
    elif val in ("false", "0", "no", ""):
        return default
    return default

# SOTA component imports (conditional based on feature flags)
try:
    from .neural_conformal_control import NCCCalibrationEngine
    NCC_AVAILABLE = True
except ImportError:
    NCC_AVAILABLE = False

try:
    from ..modules.fm_gp_residual import FMGPResidualEngine
    FMGP_AVAILABLE = True
except ImportError:
    FMGP_AVAILABLE = False

try:
    from ..modules.neural_jump_sde import NeuralJumpSDEEngine
    NEURAL_JUMP_AVAILABLE = True
except ImportError:
    NEURAL_JUMP_AVAILABLE = False

try:
    from ..modules.differentiable_greeks import DifferentiableMMEngineWrapper
    DIFF_GREEKS_AVAILABLE = True
except ImportError:
    DIFF_GREEKS_AVAILABLE = False

try:
    from ..modules.neural_rough_vol import NeuralRoughVolWrapper
    NEURAL_VOL_AVAILABLE = True
except ImportError:
    NEURAL_VOL_AVAILABLE = False

try:
    from ..modules.mamba_trend import MambaTrendWrapper
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False

try:
    from .integrated_gradients import IntegratedGradientsExplainer
    IG_AVAILABLE = True
except ImportError:
    IG_AVAILABLE = False

try:
    from ..utils.importance_sampling import compute_quantiles_with_importance_sampling
    IS_AVAILABLE = True
except ImportError:
    IS_AVAILABLE = False


@dataclass
class ForecastResult:
    quantile_paths: Dict[int, Dict[float, float]]
    threshold_probabilities: Dict[float, Dict[str, float]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    drivers: List[Tuple[str, float]] = field(default_factory=list)
    regime: RegimeVector | None = None


class ForecastEngine:
    """Orchestrates forecasting across modules with optional SOTA components."""

    def __init__(
        self,
        seed: int | None = None,
        connector: DataConnector | None = None,
        calibration: CalibrationEngine | None = None,
        metrics: MetricsSink | None = None,
        # SOTA feature flags (None = read from env, explicit bool overrides env)
        use_ncc_calibration: Optional[bool] = None,
        use_fm_gp_residuals: Optional[bool] = None,
        use_neural_jumps: Optional[bool] = None,
        use_diff_greeks: Optional[bool] = None,
        use_neural_rough_vol: Optional[bool] = None,
        use_mamba_trend: Optional[bool] = None,
        use_integrated_gradients: Optional[bool] = None,
        use_importance_sampling: Optional[bool] = None,
        device: Optional[str] = None,
        # Artifact paths for loading pretrained models (None = read from env)
        ncc_artifact_path: Optional[str] = None,
        fmgp_artifact_path: Optional[str] = None,
        neural_jump_artifact_path: Optional[str] = None,
        diff_greeks_artifact_path: Optional[str] = None,
        neural_vol_artifact_path: Optional[str] = None,
        mamba_artifact_path: Optional[str] = None,
    ) -> None:
        self._base_seed = seed

        # Read device from env if not provided, then validate availability
        requested_device = device if device is not None else os.getenv("TORCH_DEVICE", "cpu")
        self.device = get_available_device(requested_device)

        self.data_connector = connector or SyntheticDataConnector(seed=seed)
        self.normalizer = StationarityNormalizer()
        self.metrics = metrics or LoggingMetricsSink()

        # Feature flags - read from environment if not explicitly provided
        use_ncc_calibration = _getenv_bool("USE_NCC_CALIBRATION") if use_ncc_calibration is None else use_ncc_calibration
        use_fm_gp_residuals = _getenv_bool("USE_FM_GP_RESIDUALS") if use_fm_gp_residuals is None else use_fm_gp_residuals
        use_neural_jumps = _getenv_bool("USE_NEURAL_JUMPS") if use_neural_jumps is None else use_neural_jumps
        use_diff_greeks = _getenv_bool("USE_DIFF_GREEKS") if use_diff_greeks is None else use_diff_greeks
        use_neural_rough_vol = _getenv_bool("USE_NEURAL_ROUGH_VOL") if use_neural_rough_vol is None else use_neural_rough_vol
        use_mamba_trend = _getenv_bool("USE_MAMBA_TREND") if use_mamba_trend is None else use_mamba_trend
        use_integrated_gradients = _getenv_bool("USE_INTEGRATED_GRADIENTS") if use_integrated_gradients is None else use_integrated_gradients
        use_importance_sampling = _getenv_bool("USE_IMPORTANCE_SAMPLING") if use_importance_sampling is None else use_importance_sampling

        # Artifact paths - read from environment if not provided
        ncc_artifact_path = ncc_artifact_path or os.getenv("NCC_CALIBRATION_PATH")
        fmgp_artifact_path = fmgp_artifact_path or os.getenv("FMGP_RESIDUALS_PATH")
        neural_jump_artifact_path = neural_jump_artifact_path or os.getenv("NEURAL_JUMP_PATH")
        diff_greeks_artifact_path = diff_greeks_artifact_path or os.getenv("DIFF_GREEKS_PATH")
        neural_vol_artifact_path = neural_vol_artifact_path or os.getenv("NEURAL_ROUGH_VOL_PATH")
        mamba_artifact_path = mamba_artifact_path or os.getenv("MAMBA_TREND_PATH")

        # Set final feature flags (only enable if module is available)
        self.use_ncc_calibration = use_ncc_calibration and NCC_AVAILABLE
        self.use_fm_gp_residuals = use_fm_gp_residuals and FMGP_AVAILABLE
        self.use_neural_jumps = use_neural_jumps and NEURAL_JUMP_AVAILABLE
        self.use_diff_greeks = use_diff_greeks and DIFF_GREEKS_AVAILABLE
        self.use_neural_rough_vol = use_neural_rough_vol and NEURAL_VOL_AVAILABLE
        self.use_mamba_trend = use_mamba_trend and MAMBA_AVAILABLE
        self.use_integrated_gradients = use_integrated_gradients and IG_AVAILABLE
        self.use_importance_sampling = use_importance_sampling and IS_AVAILABLE

        # Initialize calibration engine
        if self.use_ncc_calibration:
            if ncc_artifact_path:
                from pathlib import Path
                self.calibration = NCCCalibrationEngine.load(Path(ncc_artifact_path), device=device)
            else:
                self.calibration = NCCCalibrationEngine(device=device)
        else:
            self.calibration = calibration or CalibrationEngine()

        # Initialize trend model
        if self.use_mamba_trend:
            if mamba_artifact_path:
                from pathlib import Path
                self.trend = MambaTrendWrapper.load(Path(mamba_artifact_path), device=device)
            else:
                self.trend = MambaTrendWrapper(device=device)
        else:
            self.trend = TrendEnsemble()

        # Initialize volatility engine
        if self.use_neural_rough_vol:
            if neural_vol_artifact_path:
                from pathlib import Path
                self.vol_engine = NeuralRoughVolWrapper.load(Path(neural_vol_artifact_path), device=device)
            else:
                self.vol_engine = NeuralRoughVolWrapper(device=device)
        else:
            self.vol_engine = VolPathEngine()

        # Initialize market maker engine
        if self.use_diff_greeks:
            if diff_greeks_artifact_path:
                from pathlib import Path
                self.mm_engine = DifferentiableMMEngineWrapper.load(Path(diff_greeks_artifact_path), device=device)
            else:
                self.mm_engine = DifferentiableMMEngineWrapper(device=device)
        else:
            self.mm_engine = MarketMakerEngine()

        # Initialize residual generator
        if self.use_fm_gp_residuals:
            from pathlib import Path
            if fmgp_artifact_path:
                self.residual_engine = FMGPResidualEngine.load(Path(fmgp_artifact_path), device=self.device)
            else:
                self.residual_engine = FMGPResidualEngine(device=self.device)
        else:
            self.residual_engine = None  # Will create per-forecast for legacy

        # Initialize jump model
        if self.use_neural_jumps:
            from pathlib import Path
            if neural_jump_artifact_path:
                self.jump_engine = NeuralJumpSDEEngine.load(Path(neural_jump_artifact_path), device=self.device)
            else:
                self.jump_engine = NeuralJumpSDEEngine(device=self.device)
        else:
            self.jump_engine = None  # Will create per-forecast for legacy

        # Initialize explainability
        self.explainability = ExplainabilityEngine()
        self.surrogate_explainer = SurrogateExplainer(seed=seed)

        if self.use_integrated_gradients:
            self.ig_explainer = IntegratedGradientsExplainer()
        else:
            self.ig_explainer = None

    def forecast(self, config: ForecastConfig) -> ForecastResult:
        # Validate configuration
        config.validate()

        start = time.perf_counter()
        seed = config.seed if config.seed is not None else self._base_seed
        rng = random.Random(seed)

        # Fetch and normalize data
        frame, normalized_closes, stats = self._fetch_and_normalize_data(config)

        # Compute regime state
        regime, regime_strength = self._compute_regime_state(frame)

        # Compute trend, market maker indices, volatility, residuals, and jumps
        trend_path = self._compute_trend_path(normalized_closes, config.horizon_days, regime_strength)
        mm_indices = self._compute_mm_indices(frame)
        vol_path = self._compute_vol_path(frame, config.horizon_days, regime_strength, mm_indices)
        residuals = self._generate_residuals(config, vol_path, regime_strength, mm_indices, seed)
        jump_model = self._initialize_jump_model(seed)

        # Assemble paths and compute quantiles
        paths = self._assemble_paths(
            trend_path,
            vol_path,
            residuals,
            regime,
            mm_indices,
            config.num_paths,
            config.horizon_days,
            frame,
            stats,
            jump_model,
            rng,
            regime_strength,
        )

        quantile_paths = self._compute_quantiles(paths, config, regime)
        threshold_probabilities = self._compute_threshold_probs(paths, config.thresholds)

        # Explainability (legacy + surrogate or Integrated Gradients)
        drivers = self.explainability.summarize(
            regime_vector=list(regime.values),
            mm_indices=mm_indices,
            vol_path=vol_path.path,
        )

        if self.use_integrated_gradients and self.ig_explainer:
            # Use Integrated Gradients for more faithful attribution
            features = {
                "regime_volatility": regime.values[0] if regime.values else 0.0,
                "iv_level": regime.values[1] if len(regime.values) > 1 else 0.0,
                "gamma_squeeze": mm_indices.gamma_squeeze,
                "inventory_unwind": mm_indices.inventory_unwind,
                "basis_pressure": mm_indices.basis_pressure,
                "funding_rate": frame.funding_rate,
                "skew": frame.skew,
                "order_imbalance": frame.order_imbalance,
            }

            # Create a simple model function that predicts cone width from features
            def cone_width_fn(feature_tensor):
                import torch
                # Weighted sum of features contributing to forecast cone width
                vol_contrib = feature_tensor[0] * CONE_WIDTH_VOL_WEIGHT
                iv_contrib = feature_tensor[1] * CONE_WIDTH_IV_WEIGHT
                gamma_contrib = feature_tensor[2] * CONE_WIDTH_GAMMA_WEIGHT
                skew_contrib = feature_tensor[6] * CONE_WIDTH_SKEW_WEIGHT
                return vol_contrib + iv_contrib + gamma_contrib + skew_contrib

            ig_result = self.ig_explainer.explain_forecast(
                features=features,
                model_fn=cone_width_fn,
            )
            drivers = ig_result.top_drivers[:5]  # Top 5 drivers
        else:
            # Use surrogate explainer
            surrogate_drivers = self.surrogate_explainer.explain(
                features={
                    "RegimeVolatility": regime.values[0] if regime.values else 0.0,
                    "IVLevel": regime.values[1] if len(regime.values) > 1 else 0.0,
                    "GammaSqueeze": mm_indices.gamma_squeeze,
                    "BasisPressure": mm_indices.basis_pressure,
                },
                quantile_paths=quantile_paths,
            )
            drivers = drivers or surrogate_drivers
        coverage_stats = self.calibration.state.coverage.get(
            (self.calibration._regime_bucket(regime), self.calibration._horizon_bucket(config.horizon_days)),
            {"hits": 0, "total": 0},
        )
        metadata = {
            "as_of": config.as_of.isoformat(),
            "asset_id": config.asset_id,
            "paths": str(config.num_paths),
            "scenario": config.scenario.description if config.scenario else "base",
            "regime_bucket": self.calibration._regime_bucket(regime),
            "horizon_bucket": self.calibration._horizon_bucket(config.horizon_days),
            "scenario_label": "conditional" if config.scenario else "base",
            "coverage_hits": int(coverage_stats.get("hits", 0)),
            "coverage_total": int(coverage_stats.get("total", 0)),
            # SOTA component flags
            "sota_enabled": {
                "ncc_calibration": self.use_ncc_calibration,
                "fm_gp_residuals": self.use_fm_gp_residuals,
                "neural_jumps": self.use_neural_jumps,
                "diff_greeks": self.use_diff_greeks,
                "neural_rough_vol": self.use_neural_rough_vol,
                "mamba_trend": self.use_mamba_trend,
                "integrated_gradients": self.use_integrated_gradients,
                "importance_sampling": self.use_importance_sampling,
            },
        }
        result = ForecastResult(
            quantile_paths=quantile_paths,
            threshold_probabilities=threshold_probabilities,
            drivers=drivers or surrogate_drivers,
            metadata=metadata,
            regime=regime,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        self.metrics.emit_forecast_metrics(
            latency_ms=latency_ms,
            asset_id=config.asset_id,
            horizon=config.horizon_days,
            num_paths=config.num_paths,
            regime_bucket=metadata["regime_bucket"],
            status="ok",
        )
        return result

    def update_calibration_with_realized(
        self, result: ForecastResult, realized_price: float, horizon: int | None = None
    ) -> None:
        if result.regime is None:
            return
        target_horizon = horizon or max(result.quantile_paths.keys())
        quantiles = result.quantile_paths.get(target_horizon)
        if not quantiles:
            return

        # Check which calibration engine we're using and pass appropriate parameters
        from .neural_conformal_control import NCCCalibrationEngine

        if isinstance(self.calibration, NCCCalibrationEngine):
            # NCC expects features dict
            features = None
            if result.regime and hasattr(result.regime, 'values'):
                features = {f"f{i}": float(v) for i, v in enumerate(result.regime.values[:10])}

            self.calibration.update_with_outcome(
                predicted_quantiles=quantiles,
                realized=realized_price,
                features=features,
                horizon=target_horizon,
            )
        else:
            # Legacy CalibrationEngine expects regime object
            self.calibration.update_with_outcome(
                predicted_quantiles=quantiles,
                realized=realized_price,
                regime=result.regime,
                horizon=target_horizon,
            )

    def forecast_batch(self, configs: Sequence[ForecastConfig]) -> Dict[str, ForecastResult]:
        results: Dict[str, ForecastResult] = {}
        for cfg in configs:
            key = f"{cfg.asset_id}:{cfg.as_of.isoformat()}:{cfg.horizon_days}"
            results[key] = self.forecast(cfg)
        return results

    def _fetch_and_normalize_data(
        self, config: ForecastConfig
    ) -> Tuple[MarketFeatureFrame, List[float], NormalizationStats]:
        """Fetch market data and normalize closes."""
        frame = self.data_connector.fetch_window(
            asset_id=config.asset_id,
            as_of=config.as_of,
            window=timedelta(days=config.trailing_window_days),
        )

        if config.scenario:
            frame = apply_scenario(frame, config.scenario)

        normalized_closes, stats = self.normalizer.normalize_and_stats(frame.closes)
        return frame, normalized_closes, stats

    def _compute_regime_state(
        self, frame: MarketFeatureFrame
    ) -> Tuple[RegimeVector, float]:
        """Compute regime vector and strength from market data."""
        regime = compute_regime_vector(
            closes=frame.closes,
            iv_points=frame.iv_points,
            funding_rate=frame.funding_rate,
            basis=frame.basis,
            order_imbalance=frame.order_imbalance,
            narrative_scores=frame.narrative_scores,
            skew=frame.skew,
        )
        regime_strength = 1.0 + abs(regime.values[0]) if regime.values else 1.0
        return regime, regime_strength

    def _compute_trend_path(
        self, normalized_closes: List[float], horizon_days: int, regime_strength: float
    ) -> TrendPath:
        """Compute trend path using legacy or Mamba model."""
        if self.use_mamba_trend:
            trend_forecast = self.trend.predict_trend(
                normalized_closes, horizon_days, regime_strength
            )
            return TrendPath(path=trend_forecast)
        else:
            return self.trend.predict_trend(normalized_closes, horizon_days, regime_strength)

    def _compute_mm_indices(self, frame: MarketFeatureFrame) -> MarketMakerIndices:
        """Compute market maker indices using legacy or differentiable Greeks."""
        if self.use_diff_greeks:
            mm_state_tensor, attn_weights = self.mm_engine.compute_indices(
                spot=frame.closes[-1] if frame.closes else 1.0,
                iv_term_structure=frame.iv_points,
                funding_rate=frame.funding_rate,
                basis=frame.basis,
                order_imbalance=frame.order_imbalance,
                skew=frame.skew,
            )
            # Extract scalar indices from embedding for compatibility
            mm_state = mm_state_tensor.cpu().numpy()
            return MarketMakerIndices(
                gamma_squeeze=float(mm_state[0]) if len(mm_state) > 0 else 0.0,
                inventory_unwind=float(mm_state[1]) if len(mm_state) > 1 else 0.0,
                basis_pressure=float(mm_state[2]) if len(mm_state) > 2 else 0.0,
            )
        else:
            return self.mm_engine.compute_indices(
                iv_term_structure=frame.iv_points,
                funding_rate=frame.funding_rate,
                basis=frame.basis,
                order_imbalance=frame.order_imbalance,
                skew=frame.skew,
            )

    def _compute_vol_path(
        self,
        frame: MarketFeatureFrame,
        horizon_days: int,
        regime_strength: float,
        mm_indices: MarketMakerIndices,
    ) -> VolPath:
        """Compute volatility path using legacy or neural rough vol."""
        mm_tuple = (mm_indices.gamma_squeeze, mm_indices.inventory_unwind, mm_indices.basis_pressure)

        if self.use_neural_rough_vol:
            vol_forecast = self.vol_engine.forecast(
                frame.iv_points,
                horizon_days,
                regime_strength,
                mm_indices=mm_tuple,
            )
            return VolPath(path=vol_forecast)
        else:
            return self.vol_engine.forecast(
                frame.iv_points, horizon_days, regime_strength, mm_indices=mm_tuple
            )

    def _generate_residuals(
        self,
        config: ForecastConfig,
        vol_path: VolPath,
        regime_strength: float,
        mm_indices: MarketMakerIndices,
        seed: int,
    ) -> ResidualPaths:
        """Generate residual paths using legacy or FM-GP."""
        if self.use_fm_gp_residuals and self.residual_engine is not None:
            # Use pre-loaded FM-GP engine
            mm_features = [
                mm_indices.gamma_squeeze,
                mm_indices.inventory_unwind,
                mm_indices.basis_pressure,
            ]

            residual_paths = self.residual_engine.sample_paths(
                horizon=config.horizon_days,
                num_paths=config.num_paths,
                vol_path=vol_path.path,
                regime_strength=regime_strength,
                mm_features=mm_features,
            )
            return ResidualPaths(paths=residual_paths)
        else:
            # Use legacy residual generator
            return ResidualGenerator(seed=seed).sample_paths(
                horizon=config.horizon_days,
                num_paths=config.num_paths,
                vol_path=vol_path.path,
                regime_strength=regime_strength,
                mm_features=(mm_indices.gamma_squeeze, mm_indices.inventory_unwind, mm_indices.basis_pressure),
            )

    def _initialize_jump_model(self, seed: int):
        """Initialize jump model (legacy or Neural Jump SDE)."""
        if self.use_neural_jumps and self.jump_engine is not None:
            return self.jump_engine
        else:
            return JumpModel(seed=seed)

    def _compute_quantiles(
        self, paths: List[List[float]], config: ForecastConfig, regime: RegimeVector
    ) -> Dict[int, Dict[float, float]]:
        """Compute quantile paths using legacy or importance sampling."""
        quantile_paths: Dict[int, Dict[float, float]] = {}

        if self.use_importance_sampling:
            # Transpose paths for importance sampling: (horizon, num_paths)
            paths_by_time = [[paths[i][t] for i in range(len(paths))] for t in range(config.horizon_days)]

            # Compute quantiles with importance sampling
            quantile_results = compute_quantiles_with_importance_sampling(
                paths_by_time, config.quantiles
            )

            # Build quantile_paths dict
            for q_idx, q in enumerate(config.quantiles):
                q_values = quantile_results[q_idx]
                for t in range(config.horizon_days):
                    if (t + 1) not in quantile_paths:
                        quantile_paths[t + 1] = {}
                    quantile_paths[t + 1][q] = q_values[t]

            # Apply calibration
            for t in range(config.horizon_days):
                quantile_paths[t + 1] = self.calibration.calibrate_quantiles(
                    quantile_paths[t + 1], regime=regime, horizon=t + 1
                )
        else:
            # Legacy quantile computation (optimized with numpy)
            for t in range(config.horizon_days):
                step_values = [path[t] for path in paths]
                # Convert quantiles to percentiles (0-100 scale) for numpy
                percentiles = [q * 100 for q in config.quantiles]
                quantile_values = np.percentile(step_values, percentiles)
                quantiles = dict(zip(config.quantiles, quantile_values))
                quantiles = self.calibration.calibrate_quantiles(
                    quantiles, regime=regime, horizon=t + 1
                )
                quantile_paths[t + 1] = quantiles

        return quantile_paths

    def _assemble_paths(
        self,
        trend_path: TrendPath,
        vol_path: VolPath,
        residuals: ResidualPaths,
        regime: RegimeVector,
        mm_indices: MarketMakerIndices,
        num_paths: int,
        horizon: int,
        frame: MarketFeatureFrame,
        stats: NormalizationStats,
        jump_model,  # Can be JumpModel or NeuralJumpSDEEngine
        rng: random.Random,
        regime_strength: float,
    ) -> List[List[float]]:
        paths: List[List[float]] = []
        narrative_intensity = sum(frame.narrative_scores.values()) / max(
            len(frame.narrative_scores), 1
        )

        # Check if using Neural Jump SDE
        is_neural_jump = self.use_neural_jumps and hasattr(jump_model, 'sample_sde_paths')

        if is_neural_jump:
            # Use Neural Jump SDE for all paths at once
            import torch

            # Build conditioning
            conditioning = [
                regime.values[0] if regime.values else 0.0,
                regime.values[1] if len(regime.values) > 1 else 0.0,
                mm_indices.gamma_squeeze,
                mm_indices.inventory_unwind,
                mm_indices.basis_pressure,
                narrative_intensity,
            ]

            # Sample jump paths
            x0 = torch.zeros(num_paths, device=self.device)
            jump_paths = jump_model.sample_sde_paths(
                x0=x0,
                conditioning=conditioning,
                horizon=horizon,
                vol_path=vol_path.path,
            )  # (num_paths, horizon)

            jump_paths_list = jump_paths.cpu().numpy().tolist()

            # Combine with trend and residuals
            for i in range(num_paths):
                combined: List[float] = []
                for t in range(horizon):
                    base = trend_path.path[t]
                    noise = residuals.paths[i][t]
                    jump = jump_paths_list[i][t]
                    normalized = base + noise + jump
                    combined.append(stats.denormalize(normalized))
                paths.append(combined)
        else:
            # Use legacy JumpModel
            for i in range(num_paths):
                jump_path = jump_model.sample_path(
                    horizon=horizon,
                    vol_path=vol_path.path,
                    narrative_score=narrative_intensity,
                    gamma_squeeze=mm_indices.gamma_squeeze,
                    regime_strength=regime_strength,
                    basis_pressure=mm_indices.basis_pressure,
                )
                combined: List[float] = []
                for t in range(horizon):
                    base = trend_path.path[t]
                    noise = residuals.paths[i][t]
                    jump = jump_path.path[t]
                    normalized = base + noise + jump
                    combined.append(stats.denormalize(normalized))
                paths.append(combined)

        return paths

    def _compute_threshold_probs(
        self, paths: List[List[float]], thresholds: Sequence[float]
    ) -> Dict[float, Dict[str, float]]:
        terminal_prices = [path[-1] for path in paths]
        results: Dict[float, Dict[str, float]] = {}
        for k in thresholds:
            below = sum(1 for p in terminal_prices if p < k) / len(terminal_prices)
            above = sum(1 for p in terminal_prices if p > k) / len(terminal_prices)
            results[k] = {"lt": below, "gt": above}
        return results
