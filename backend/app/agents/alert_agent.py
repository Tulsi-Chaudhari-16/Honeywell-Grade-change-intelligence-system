"""
GCIS — Alert Agent (Deterministic)
Fires on: p_offspec threshold, sensor staleness, model degradation, agent failures.
De-duplication window prevents re-firing same alert continuously.
No external dependencies beyond Redis — must never fail silently.
"""
import time
import uuid
import structlog

from app.core.config import settings
from app.core.redis_client import is_alert_duplicate, publish_alert, get_redis
from app.db.session import get_session
from app.db.models import Alert

log = structlog.get_logger(__name__)


async def fire_alert(
    alert_type: str,
    severity: str,
    message: str,
    episode_id: str | None = None,
    dedup_key: str | None = None,
) -> str | None:
    """
    Fire an alert:
    1. Check deduplication window
    2. Persist to Postgres alerts table
    3. Publish to Redis for WebSocket push

    Returns alert_id if fired, None if deduplicated.
    Never raises — logs errors internally.
    """
    try:
        key = dedup_key or f"{alert_type}:{episode_id or 'global'}"
        if await is_alert_duplicate(key, settings.ALERT_DEDUP_WINDOW_SEC):
            log.debug("alert_agent.deduplicated", alert_type=alert_type, key=key)
            return None

        alert_id = str(uuid.uuid4())

        # Persist to Postgres
        try:
            async with get_session() as session:
                session.add(Alert(
                    alert_id=alert_id,
                    episode_id=episode_id,
                    severity=severity,
                    alert_type=alert_type,
                    message=message,
                    dedup_key=key,
                ))
        except Exception as db_err:
            log.error("alert_agent.db_persist_failed", error=str(db_err))
            # Continue — still try to publish to Redis

        # Publish to Redis for WebSocket
        try:
            await publish_alert({
                "alert_id":   alert_id,
                "alert_type": alert_type,
                "severity":   severity,
                "message":    message,
                "episode_id": episode_id,
                "ts":         time.time(),
            })
        except Exception as pub_err:
            log.error("alert_agent.redis_publish_failed", error=str(pub_err))

        log.info("alert_agent.fired",
                 alert_id=alert_id, alert_type=alert_type, severity=severity)
        return alert_id

    except Exception as e:
        # Never fail silently
        log.error("alert_agent.unexpected_error", error=str(e), alert_type=alert_type)
        return None


async def check_prediction_threshold(state: dict) -> None:
    """Fire alert if p_offspec crosses ALERT_THRESHOLD."""
    p_offspec  = state.get("p_offspec", 0.0)
    episode_id = state.get("episode_id")

    if p_offspec >= settings.ALERT_THRESHOLD:
        await fire_alert(
            alert_type="p_offspec_threshold",
            severity="critical" if p_offspec >= 0.85 else "warning",
            message=(
                f"p_offspec={round(p_offspec, 4)} exceeds threshold "
                f"{settings.ALERT_THRESHOLD} for episode {episode_id}"
            ),
            episode_id=episode_id,
            dedup_key=f"p_offspec:{episode_id}",
        )


async def check_model_degraded(state: dict) -> None:
    """Fire alert if model is in degraded mode."""
    if state.get("degraded_mode", False):
        await fire_alert(
            alert_type="model_degraded",
            severity="warning",
            message=(
                f"Prediction model degraded (version: {state.get('model_version','unknown')}). "
                "Using fallback inference."
            ),
            episode_id=state.get("episode_id"),
            dedup_key=f"degraded:{state.get('episode_id')}",
        )


async def check_agent_failure(agent_name: str, error: str, episode_id: str | None) -> None:
    """Fire alert on any agent failure."""
    await fire_alert(
        alert_type="agent_failure",
        severity="critical",
        message=f"Agent '{agent_name}' failed: {error}",
        episode_id=episode_id,
        dedup_key=f"agent_failure:{agent_name}:{episode_id}",
    )


async def run(state: dict) -> dict:
    """LangGraph node entry point — checks all alert conditions."""
    await check_prediction_threshold(state)
    await check_model_degraded(state)
    return {}
