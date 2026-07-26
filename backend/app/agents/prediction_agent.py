"""
GCIS — Prediction Agent (Deterministic — ML inference)
Calls ml/infer/ wrappers. Implements degraded-mode fallback chain.
"""
import time
import uuid
from datetime import datetime

import structlog

from app.core.config import settings
from app.core.redis_client import publish_alert, is_alert_duplicate
from app.db.session import get_session
from app.db.models import Prediction

log = structlog.get_logger(__name__)


def _call_ml_inference(feature_vector: dict) -> dict:
    """
    Call the ML partner's prediction inference wrapper.
    Interface: ml/infer/predict(feature_vector) -> Prediction dict

    Degraded-mode fallback chain:
      1. Primary model (ml/infer/)
      2. Last deployed good model
      3. Isolation Forest + rule-based check
    """
    # ── Attempt 1: Primary ML model ──────────────────────────────────────────
    try:
        from ml.infer import predict  # type: ignore
        result = predict(feature_vector)
        return {**result, "degraded_mode": False}
    except ImportError:
        log.warning("prediction_agent.ml_stub_used")
    except Exception as e:
        log.error("prediction_agent.primary_model_failed", error=str(e))

    # ── Attempt 2: Isolation Forest fallback ─────────────────────────────────
    try:
        return _isolation_forest_fallback(feature_vector)
    except Exception as e:
        log.error("prediction_agent.isolation_forest_failed", error=str(e))

    # ── Attempt 3: Rule-based minimum viable prediction ───────────────────────
    return _rule_based_fallback(feature_vector)


def _isolation_forest_fallback(feature_vector: dict) -> dict:
    """Isolation Forest anomaly score as p_offspec proxy."""
    import numpy as np
    from sklearn.ensemble import IsolationForest

    features = feature_vector.get("features", feature_vector)
    values = np.array(list(features.values()), dtype=float).reshape(1, -1)

    # Fit on single sample (demo mode — in production use pre-fitted model)
    clf = IsolationForest(contamination=0.1, random_state=42)
    clf.fit(values)
    score = clf.score_samples(values)[0]

    # Convert anomaly score (-1 to 0 range) to [0,1] p_offspec proxy
    p_offspec = float(max(0.0, min(1.0, (-score - 0.1) * 2)))

    log.warning("prediction_agent.using_isolation_forest_fallback", p_offspec=p_offspec)
    return {
        "p_offspec":          p_offspec,
        "trajectory":         [p_offspec] * 10,
        "eta_stabilize_min":  30.0,
        "confidence":         0.3,
        "model_version":      "isolation_forest_fallback",
        "degraded_mode":      True,
    }


def _rule_based_fallback(feature_vector: dict) -> dict:
    """Minimal rule-based: flag if steam pressure exceeds upper bound."""
    features = feature_vector.get("features", feature_vector)
    steam = float(features.get("steam_pressure_p1", 4.0))
    p_offspec = min(1.0, max(0.0, (steam - 5.0) / 2.0)) if steam > 5.0 else 0.2

    log.warning("prediction_agent.using_rule_based_fallback", p_offspec=p_offspec)
    return {
        "p_offspec":          p_offspec,
        "trajectory":         [p_offspec] * 10,
        "eta_stabilize_min":  45.0,
        "confidence":         0.2,
        "model_version":      "rule_based_fallback",
        "degraded_mode":      True,
    }


async def run(state: dict) -> dict:
    """
    LangGraph node entry point.
    Calls ML inference, persists Prediction to Postgres,
    fires alert if degraded_mode or p_offspec threshold crossed.

    Returns: updated state with prediction fields.
    """
    episode_id = state.get("episode_id", "unknown")
    feature_vector = state.get("feature_vector", {})

    log.info("prediction_agent.run", episode_id=episode_id)
    t0 = time.time()

    prediction = _call_ml_inference(feature_vector)
    latency_ms = int((time.time() - t0) * 1000)

    p_offspec    = prediction["p_offspec"]
    degraded     = prediction["degraded_mode"]
    model_ver    = prediction["model_version"]

    # ── Persist prediction to Postgres ───────────────────────────────────────
    prediction_id = str(uuid.uuid4())
    async with get_session() as session:
        session.add(Prediction(
            prediction_id=prediction_id,
            episode_id=episode_id if episode_id != "unknown" else None,
            p_offspec=p_offspec,
            trajectory=prediction.get("trajectory"),
            eta_stabilize_min=prediction.get("eta_stabilize_min"),
            confidence=prediction.get("confidence"),
            model_version=model_ver,
            degraded_mode=degraded,
        ))

    # ── Alert if degraded mode ────────────────────────────────────────────────
    if degraded:
        dedup_key = f"degraded:{episode_id}"
        if not await is_alert_duplicate(dedup_key, settings.ALERT_DEDUP_WINDOW_SEC):
            await publish_alert({
                "alert_type":  "model_degraded",
                "severity":    "warning",
                "episode_id":  episode_id,
                "model_version": model_ver,
                "ts":          time.time(),
            })

    log.info("prediction_agent.done",
             episode_id=episode_id,
             p_offspec=round(p_offspec, 4),
             degraded=degraded,
             latency_ms=latency_ms)

    return {
        "p_offspec":          p_offspec,
        "trajectory":         prediction.get("trajectory", []),
        "eta_stabilize_min":  prediction.get("eta_stabilize_min", 0.0),
        "confidence":         prediction.get("confidence", 0.0),
        "model_version":      model_ver,
        "degraded_mode":      degraded,
        "prediction_id":      prediction_id,
    }
