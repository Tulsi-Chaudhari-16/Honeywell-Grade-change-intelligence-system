"""
GCIS ML Pipeline — Shared Pydantic Types (Interface Contract)

These are the typed contracts between:
  - My ML code (producer)
  - Partner's agents (consumer)

The partner's Prediction Agent, Root Cause Agent, and Historical Retrieval Agent
import from this module. Changes here must be coordinated.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


# ─── Input ────────────────────────────────────────────────────────────────────


class FeatureVector(BaseModel):
    """
    A single feature snapshot computed by the Feature Engineering Agent.
    Consumed by the Prediction Agent and passed to my inference wrappers.
    """

    episode_id: str
    ts: datetime
    features: dict[str, float] = Field(
        description="All engineered features keyed by name, e.g. stock_flow_mean_30s"
    )
    imputation_flags: dict[str, bool] = Field(
        default_factory=dict,
        description="True for any feature that was imputed (sensor dropout). "
        "Prediction Agent downgrades confidence when flags are present.",
    )

    model_config = {"frozen": True}


# ─── Outputs ──────────────────────────────────────────────────────────────────


class Prediction(BaseModel):
    """
    The primary output of the Prediction Agent (my inference wrapper).
    Consumed by the Root Cause Agent and Alert Agent.
    """

    episode_id: str
    ts: datetime

    p_offspec: float = Field(
        ge=0.0,
        le=1.0,
        description="Probability that Basis Weight will exceed ±2.5% of setpoint "
        "within the forecast horizon.",
    )
    trajectory: list[float] = Field(
        description="Predicted Basis Weight values for each step in the forecast horizon."
    )
    trajectory_lower: list[float] = Field(
        description="Lower uncertainty bound (5th percentile) of trajectory."
    )
    trajectory_upper: list[float] = Field(
        description="Upper uncertainty bound (95th percentile) of trajectory."
    )
    eta_stabilize_min: float | None = Field(
        default=None,
        description="Estimated minutes until Basis Weight stabilises within ±2.5% band. "
        "None if the model predicts it will not stabilise in the horizon.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Blended confidence: model calibration confidence weighted with "
        "historical retrieval support score.",
    )
    model_version: str
    degraded_mode: bool = Field(
        default=False,
        description="True if falling back to rule-based / Isolation Forest because "
        "primary LightGBM model is unavailable.",
    )


class FactorAttribution(BaseModel):
    """A single feature's contribution to a predicted Basis Weight excursion."""

    feature_name: str
    shap_value: float = Field(
        description="SHAP value — signed magnitude of contribution to the model output."
    )
    feature_value: float = Field(description="Actual current value of this feature.")
    direction: Literal["too_high", "too_low", "too_fast", "too_slow", "neutral"]
    rank: int = Field(ge=1, description="1 = highest contributor.")
    envelope_deviation: float | None = Field(
        default=None,
        description="How far this feature deviates from the mean successful-transition "
        "envelope for this grade pair, in standard deviations.",
    )


class RootCauseAttribution(BaseModel):
    """
    SHAP-based root cause analysis output.
    Consumed by the Root Cause Agent (partner) which narrates it with Claude.
    The LLM receives this struct verbatim — it never recomputes SHAP values.
    """

    prediction_id: str
    episode_id: str
    ts: datetime

    ranked_factors: list[FactorAttribution] = Field(
        description="Top contributing features, ranked by |shap_value| descending."
    )
    attribution_method: Literal["shap", "permutation_fallback"] = Field(
        default="shap",
        description="'permutation_fallback' if SHAP timed out and we fell back to "
        "precomputed permutation importance.",
    )
    model_expected_value: float = Field(
        description="SHAP baseline (mean model output). Helps the LLM contextualise "
        "individual SHAP magnitudes."
    )


class DriftReport(BaseModel):
    """
    Output of the drift detection module. Consumed by the Learning Agent.
    """

    computed_at: datetime
    feature_psi: dict[str, float] = Field(
        description="PSI value per feature. PSI > 0.2 is the retrain trigger threshold."
    )
    top_drifted_features: list[str] = Field(
        description="Features with PSI > 0.2, sorted by PSI descending."
    )
    triggered_retrain: bool = Field(
        description="True if any feature PSI > PSI_RETRAIN_THRESHOLD."
    )
    new_episodes_since_last_train: int = 0


class AnomalyReport(BaseModel):
    """Per-tag anomaly scores from Isolation Forest + z-score ensemble."""

    ts: datetime
    anomaly_scores: dict[str, float] = Field(
        description="Score per tag. Higher = more anomalous. Scores are normalised to [0, 1]."
    )
    flagged_tags: list[str] = Field(
        description="Tags whose anomaly score exceeds the alert threshold."
    )
    overall_anomaly: bool
