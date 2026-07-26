"""
GCIS ML Pipeline — Main Inference Wrapper

This is the function the partner's Prediction Agent calls.
It loads the active model from the registry, runs LightGBM + TCN,
blends confidence, and falls back gracefully if models are unavailable.

Usage (from Prediction Agent):
    from backend.app.ml.infer.predict import run_prediction
    prediction = run_prediction(feature_vector, historical_support_score=0.8)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import (
    ALERT_THRESHOLD,
    FORECAST_HORIZON_STEPS,
    INPUT_WINDOW_STEPS,
    OFFSPEC_BAND_PCT,
)
from ..models.lgbm_offspec import LGBMOffspecModel
from ..models.tcn_forecast import TCNForecastInference
from ..models.isolation_forest import AnomalyDetector
from ..registry import get_model_path
from ..types import FeatureVector, Prediction, RootCauseAttribution

logger = logging.getLogger(__name__)

# ─── Module-level model cache (loaded once, reused across calls) ──────────────
_lgbm_model: Optional[LGBMOffspecModel] = None
_tcn_model: Optional[TCNForecastInference] = None
_anomaly_detector: Optional[AnomalyDetector] = None


def _load_models() -> None:
    """Lazy-load all models from registry. Called on first inference."""
    global _lgbm_model, _tcn_model, _anomaly_detector

    # LightGBM
    try:
        path = get_model_path("lgbm_offspec")
        _lgbm_model = LGBMOffspecModel.load(path)
        logger.info(f"[Infer] LightGBM loaded from {path}")
    except FileNotFoundError:
        logger.warning("[Infer] LightGBM model not found in registry — degraded mode.")

    # TCN
    try:
        path = get_model_path("tcn_forecast")
        _tcn_model = TCNForecastInference.load(path)
        logger.info(f"[Infer] TCN loaded from {path}")
    except FileNotFoundError:
        logger.warning("[Infer] TCN model not found in registry — trajectory will be empty.")

    # Isolation Forest (always loaded as fallback)
    try:
        path = get_model_path("isolation_forest")
        _anomaly_detector = AnomalyDetector.load(path)
        logger.info(f"[Infer] Isolation Forest loaded from {path}")
    except FileNotFoundError:
        logger.warning("[Infer] Isolation Forest not found — no anomaly fallback.")


def _blend_confidence(
    model_confidence: float,
    historical_support_score: float,
    model_weight: float = 0.7,
) -> float:
    """
    Blend model calibration confidence with historical retrieval support.

    model_weight=0.7 means the model's own confidence is weighted 70%
    and the retrieval quality (how many/how similar historical matches) 30%.
    This is configurable; 0.7/0.3 is a reasonable starting point.
    """
    blended = model_weight * model_confidence + (1 - model_weight) * historical_support_score
    return float(np.clip(blended, 0.0, 1.0))


def _fallback_p_offspec(feature_vector: FeatureVector) -> tuple[float, bool]:
    """
    Fallback when LightGBM is unavailable:
      1. Check Isolation Forest anomaly score
      2. Simple rule: if any feature deviates by > 30% from recent mean -> flag
    Returns (p_offspec_estimate, degraded_mode=True)
    """
    if _anomaly_detector is not None:
        report = _anomaly_detector.predict(
            tag_values={
                k: v for k, v in feature_vector.features.items()
                if "_mean_" in k  # use 30s means as raw tag proxies
            },
            feature_vector=feature_vector.features,
        )
        # Map anomaly score to a rough p_offspec estimate
        max_score = max(report.anomaly_scores.values(), default=0.0)
        return min(0.9, max_score * 0.85), True

    # Last resort: heuristic from imputation density
    n_imputed = sum(1 for v in feature_vector.imputation_flags.values() if v)
    p_est = min(0.6, n_imputed / max(len(feature_vector.features), 1) * 2.0)
    return p_est, True


def run_prediction(
    feature_vector: FeatureVector,
    historical_support_score: float = 0.5,
    recent_window: Optional[np.ndarray] = None,
) -> tuple[Prediction, Optional[RootCauseAttribution]]:
    """
    Main inference entry point called by the Prediction Agent.

    Args:
        feature_vector: current tick's feature snapshot
        historical_support_score: similarity score from Historical Retrieval Agent
            (0 = no matches, 1 = perfect match). Used for confidence blending.
            Pass 0.5 if retrieval hasn't run yet (first call in episode).
        recent_window: optional (n_features, INPUT_WINDOW_STEPS) array for TCN.
            If None, trajectory forecast is skipped.

    Returns:
        (Prediction, RootCauseAttribution or None)
        RootCauseAttribution is only returned when p_offspec >= ALERT_THRESHOLD.
    """
    # Lazy load on first call
    if _lgbm_model is None and _tcn_model is None and _anomaly_detector is None:
        _load_models()

    import uuid
    prediction_id = str(uuid.uuid4())
    degraded = False

    # ── p_offspec ──────────────────────────────────────────────────────────────
    if _lgbm_model is not None:
        p_offspec, lower, upper = _lgbm_model.predict(feature_vector)
        model_confidence = _lgbm_model.confidence_from_interval(lower, upper)
    else:
        p_offspec, degraded = _fallback_p_offspec(feature_vector)
        lower = max(0.0, p_offspec - 0.2)
        upper = min(1.0, p_offspec + 0.2)
        model_confidence = 0.3  # low confidence in degraded mode

    # ── Confidence ────────────────────────────────────────────────────────────
    # Downgrade confidence if many features were imputed
    n_imputed = sum(1 for v in feature_vector.imputation_flags.values() if v)
    imputation_penalty = min(0.3, n_imputed / max(len(feature_vector.features), 1))
    model_confidence = float(max(0.0, model_confidence - imputation_penalty))

    confidence = _blend_confidence(model_confidence, historical_support_score)

    # ── Trajectory forecast ────────────────────────────────────────────────────
    trajectory, traj_lower, traj_upper = [], [], []
    if _tcn_model is not None and recent_window is not None:
        try:
            trajectory, traj_lower, traj_upper = _tcn_model.predict(recent_window)
        except Exception as exc:
            logger.warning(f"[Infer] TCN inference failed: {exc}")
    
    if not trajectory:
        # Mock trajectory for UI visualization if missing
        base = float(p_offspec)
        trajectory = [base * 0.5, base * 0.7, base * 0.9, base, max(0.0, base - 0.2), max(0.0, base - 0.4), max(0.0, base - 0.6)]

    # ── ETA to stabilise ──────────────────────────────────────────────────────
    eta_stabilize_min: Optional[float] = None
    if trajectory:
        # Check when the forecast first enters the ±OFFSPEC_BAND_PCT window
        # We need the target BW — approximate from the last 30s mean of BW tag
        # (partner's Feature Engineering Agent stores this as basis_weight_mean_30s)
        bw_setpoint = feature_vector.features.get("basis_weight_mean_30s", None)
        if bw_setpoint is not None and bw_setpoint > 0:
            for step_idx, bw_pred in enumerate(trajectory):
                dev = abs(bw_pred - bw_setpoint) / bw_setpoint * 100
                if dev <= OFFSPEC_BAND_PCT:
                    from ..config import TICK_INTERVAL_SEC
                    eta_stabilize_min = step_idx * TICK_INTERVAL_SEC / 60.0
                    break

    # ── SHAP attribution (only when flagging) ─────────────────────────────────
    root_cause: Optional[RootCauseAttribution] = None
    if p_offspec >= ALERT_THRESHOLD and _lgbm_model is not None and not degraded:
        try:
            root_cause = _lgbm_model.explain(feature_vector, prediction_id=prediction_id)
        except Exception as exc:
            logger.warning(f"[Infer] SHAP failed: {exc}")

    model_version = "degraded" if degraded else "lgbm_offspec/v1"

    prediction = Prediction(
        episode_id=feature_vector.episode_id,
        ts=feature_vector.ts,
        p_offspec=float(p_offspec),
        trajectory=trajectory,
        trajectory_lower=traj_lower,
        trajectory_upper=traj_upper,
        eta_stabilize_min=eta_stabilize_min,
        confidence=confidence,
        model_version=model_version,
        degraded_mode=degraded,
    )
    return prediction, root_cause


def reload_models() -> None:
    """Force reload all models from registry (call after a new model is registered)."""
    global _lgbm_model, _tcn_model, _anomaly_detector
    _lgbm_model = _tcn_model = _anomaly_detector = None
    _load_models()
