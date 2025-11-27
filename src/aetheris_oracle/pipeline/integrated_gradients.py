"""
Integrated Gradients for faithful model attribution.

Implements the approach from "Axiomatic Attribution for Deep Networks" (Sundararajan et al., 2017).
Provides more faithful attribution than permutation importance or SHAP for neural models.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class AttributionResult:
    """Result of attribution analysis."""

    feature_attributions: Dict[str, float]  # feature_name -> attribution score
    top_drivers: List[Tuple[str, float]]  # Sorted by absolute attribution
    baseline_output: float
    actual_output: float
    concept_explanations: List[str]  # Human-readable explanations


class IntegratedGradientsExplainer:
    """
    Integrated Gradients explainer for forecast models.

    Computes feature attributions via path integral of gradients from baseline to input.
    More faithful than permutation importance as it uses actual gradients.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        baseline_strategy: str = "zero",
        n_steps: int = 50,
    ):
        """
        Args:
            model: PyTorch model to explain (optional, can explain any differentiable function)
            baseline_strategy: How to choose baseline ('zero', 'mean', 'random')
            n_steps: Number of interpolation steps
        """
        self.model = model
        self.baseline_strategy = baseline_strategy
        self.n_steps = n_steps

    def explain_forecast(
        self,
        features: Dict[str, float],
        model_fn: Optional[Callable] = None,
        baseline: Optional[Dict[str, float]] = None,
    ) -> AttributionResult:
        """
        Explain a forecast using Integrated Gradients.

        Args:
            features: Input features used for forecast
            model_fn: Function mapping features dict to scalar output (e.g., P50 or cone width)
            baseline: Baseline features (if None, uses baseline_strategy)

        Returns:
            AttributionResult with feature attributions
        """
        if model_fn is None and self.model is None:
            raise ValueError("Must provide either model_fn or model")

        # Convert to tensor
        feature_names = sorted(features.keys())
        input_values = torch.tensor(
            [features[k] for k in feature_names], dtype=torch.float32, requires_grad=True
        )

        # Get baseline
        if baseline is None:
            baseline = self._get_baseline(features)

        baseline_values = torch.tensor(
            [baseline.get(k, 0.0) for k in feature_names], dtype=torch.float32
        )

        # Compute integrated gradients
        attributions_tensor = self._integrated_gradients(
            input_values, baseline_values, model_fn or self._default_model_fn
        )

        attributions = {
            name: float(attr)
            for name, attr in zip(feature_names, attributions_tensor.detach().numpy())
        }

        # Get outputs
        baseline_output = float(
            (model_fn or self._default_model_fn)(baseline_values).detach()
        )
        actual_output = float((model_fn or self._default_model_fn)(input_values).detach())

        # Sort by absolute attribution
        top_drivers = sorted(attributions.items(), key=lambda x: abs(x[1]), reverse=True)

        # Generate concept explanations
        concept_explanations = self._generate_concept_explanations(
            attributions, features, top_k=3
        )

        return AttributionResult(
            feature_attributions=attributions,
            top_drivers=top_drivers,
            baseline_output=baseline_output,
            actual_output=actual_output,
            concept_explanations=concept_explanations,
        )

    def _integrated_gradients(
        self,
        inputs: torch.Tensor,
        baseline: torch.Tensor,
        model_fn: Callable,
    ) -> torch.Tensor:
        """
        Compute integrated gradients.

        IG = (x - x') ∫₀¹ ∂f(x' + α(x - x'))/∂x dα

        Args:
            inputs: Input features
            baseline: Baseline features
            model_fn: Model function

        Returns:
            Attribution scores per feature
        """
        # Create interpolated inputs
        alphas = torch.linspace(0, 1, self.n_steps + 1)
        interpolated = [
            baseline + alpha * (inputs - baseline) for alpha in alphas
        ]

        # Compute gradients at each interpolated point
        gradients = []
        for interp in interpolated:
            interp.requires_grad_(True)
            output = model_fn(interp)
            grad = torch.autograd.grad(output, interp, create_graph=False)[0]
            gradients.append(grad)

        # Average gradients (trapezoidal rule)
        avg_gradients = torch.stack(gradients).mean(dim=0)

        # Integrated gradients = (input - baseline) * avg_gradients
        integrated_grads = (inputs - baseline) * avg_gradients

        return integrated_grads

    def _default_model_fn(self, features: torch.Tensor) -> torch.Tensor:
        """Default model function using self.model."""
        if self.model is None:
            raise ValueError("No model provided")
        return self.model(features.unsqueeze(0)).squeeze()

    def _get_baseline(self, features: Dict[str, float]) -> Dict[str, float]:
        """Get baseline features based on strategy."""
        if self.baseline_strategy == "zero":
            return {k: 0.0 for k in features.keys()}
        elif self.baseline_strategy == "mean":
            # Use typical/mean values (would need historical data in practice)
            return {
                "regime_volatility": 0.0,  # Normalized
                "iv_level": 0.0,
                "gamma_squeeze": 0.0,
                "inventory_unwind": 0.0,
                "basis_pressure": 0.0,
                "funding_rate": 0.0,
                "skew": 0.0,
                "narrative_score": 0.0,
                "order_imbalance": 0.0,
                "horizon_norm": features.get("horizon_norm", 0.0),
            }
        else:  # random
            return {k: np.random.normal(0, 0.1) for k in features.keys()}

    def _generate_concept_explanations(
        self, attributions: Dict[str, float], features: Dict[str, float], top_k: int = 3
    ) -> List[str]:
        """
        Generate human-readable concept-level explanations.

        Args:
            attributions: Feature attributions
            features: Actual feature values
            top_k: Number of top explanations

        Returns:
            List of explanation strings
        """
        explanations = []

        # Sort by absolute attribution
        sorted_attrs = sorted(
            attributions.items(), key=lambda x: abs(x[1]), reverse=True
        )

        for feature_name, attribution in sorted_attrs[:top_k]:
            feature_value = features.get(feature_name, 0.0)

            # Generate concept explanation
            if "gamma_squeeze" in feature_name.lower():
                if attribution > 0:
                    expl = f"Elevated gamma squeeze (value: {feature_value:.2f}) → dealer covering drives upside risk"
                else:
                    expl = f"Low gamma squeeze (value: {feature_value:.2f}) → reduced upside pressure"

            elif "inventory" in feature_name.lower():
                if attribution > 0:
                    expl = f"Inventory unwind pressure (value: {feature_value:.2f}) → mean reversion expected"
                else:
                    expl = f"Low inventory pressure (value: {feature_value:.2f}) → trend continuation"

            elif "basis" in feature_name.lower():
                if attribution > 0:
                    expl = f"Positive basis/funding (value: {feature_value:.2f}) → bullish sentiment"
                else:
                    expl = f"Negative basis/funding (value: {feature_value:.2f}) → bearish sentiment"

            elif "volatility" in feature_name.lower() or "iv" in feature_name.lower():
                if attribution > 0:
                    expl = f"Elevated volatility (value: {feature_value:.2f}) → wider forecast cone"
                else:
                    expl = f"Low volatility (value: {feature_value:.2f}) → tighter forecast range"

            elif "skew" in feature_name.lower():
                if attribution > 0:
                    expl = f"Put skew dominance (value: {feature_value:.2f}) → downside tail risk"
                else:
                    expl = f"Call skew or flat (value: {feature_value:.2f}) → balanced tails"

            elif "narrative" in feature_name.lower():
                if attribution > 0:
                    expl = f"High narrative intensity (value: {feature_value:.2f}) → event risk elevated"
                else:
                    expl = f"Low narrative activity (value: {feature_value:.2f}) → quiet period"

            else:
                # Generic explanation
                direction = "increases" if attribution > 0 else "decreases"
                expl = f"{feature_name} (value: {feature_value:.2f}) {direction} forecast uncertainty"

            explanations.append(expl)

        return explanations


class AttentionAttributionExtractor:
    """
    Extracts attributions from attention weights in models.

    Useful when models (like MM engine or trend model) use attention mechanisms.
    """

    def __init__(self):
        self.attention_weights_cache: Dict[str, torch.Tensor] = {}

    def register_attention_hook(self, model: nn.Module, layer_name: str):
        """
        Register hook to capture attention weights.

        Args:
            model: Model with attention layers
            layer_name: Name of attention layer to hook
        """

        def hook_fn(module, input, output):
            # Assuming output includes attention weights
            if isinstance(output, tuple) and len(output) > 1:
                self.attention_weights_cache[layer_name] = output[1].detach()
            else:
                self.attention_weights_cache[layer_name] = output.detach()

        # Find and register hook
        for name, module in model.named_modules():
            if layer_name in name and isinstance(module, nn.MultiheadAttention):
                module.register_forward_hook(hook_fn)

    def get_attention_attributions(
        self, layer_name: str, feature_names: Sequence[str]
    ) -> Dict[str, float]:
        """
        Get attention-based attributions.

        Args:
            layer_name: Name of attention layer
            feature_names: Names of input features

        Returns:
            Attribution scores per feature
        """
        if layer_name not in self.attention_weights_cache:
            return {}

        attn_weights = self.attention_weights_cache[layer_name]

        # Average over heads and sequences
        if attn_weights.dim() == 3:  # (batch, n_heads, seq_len, seq_len)
            attn_weights = attn_weights.mean(dim=1)  # Average over heads

        # Get self-attention scores (diagonal or last position)
        if attn_weights.dim() == 2:
            scores = attn_weights[-1, :].cpu().numpy()  # Last position attention
        else:
            scores = torch.diag(attn_weights).cpu().numpy()

        # Map to feature names
        n_features = min(len(feature_names), len(scores))
        attributions = {
            feature_names[i]: float(scores[i]) for i in range(n_features)
        }

        return attributions


def combine_attributions(
    integrated_grad_attrs: Dict[str, float],
    attention_attrs: Dict[str, float],
    weights: Tuple[float, float] = (0.7, 0.3),
) -> Dict[str, float]:
    """
    Combine multiple attribution methods.

    Args:
        integrated_grad_attrs: Attributions from integrated gradients
        attention_attrs: Attributions from attention weights
        weights: (ig_weight, attention_weight)

    Returns:
        Combined attributions
    """
    combined = {}

    all_features = set(integrated_grad_attrs.keys()) | set(attention_attrs.keys())

    ig_weight, attn_weight = weights
    total_weight = ig_weight + attn_weight

    for feature in all_features:
        ig_score = integrated_grad_attrs.get(feature, 0.0)
        attn_score = attention_attrs.get(feature, 0.0)

        combined[feature] = (ig_weight * ig_score + attn_weight * attn_score) / total_weight

    return combined
