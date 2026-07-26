"""
GCIS — Operator Feedback Agent (Deterministic)
Captures Accept/Reject/Modify + optional note from UI.
Write failure → exponential backoff retry (3 attempts).
UI shows optimistic confirmation; reconciles on retry result.
"""
import asyncio
import time
import uuid
import structlog

from app.core.redis_client import get_redis
from app.db.session import get_session
from app.db.models import Feedback

log = structlog.get_logger(__name__)

MAX_RETRIES    = 3
BACKOFF_BASE   = 1.0   # seconds


async def _write_feedback_with_retry(
    recommendation_id: str,
    operator_id: str,
    action: str,
    modified_value: float | None = None,
    note: str | None = None,
) -> str:
    """
    Persist feedback to Postgres with exponential backoff retry.
    Returns feedback_id on success. Raises on all retries exhausted.
    """
    feedback_id = str(uuid.uuid4())
    last_err = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with get_session() as session:
                session.add(Feedback(
                    feedback_id=feedback_id,
                    recommendation_id=recommendation_id if recommendation_id else None,
                    operator_id=operator_id if operator_id else None,
                    action=action,
                    modified_value=modified_value,
                    note=note,
                ))
            log.info("operator_feedback_agent.written",
                     feedback_id=feedback_id, action=action, attempt=attempt)
            return feedback_id
        except Exception as e:
            last_err = e
            wait = BACKOFF_BASE * (2 ** (attempt - 1))
            log.warning("operator_feedback_agent.retry",
                        attempt=attempt, wait_sec=wait, error=str(e))
            await asyncio.sleep(wait)

    log.error("operator_feedback_agent.all_retries_exhausted", error=str(last_err))
    raise RuntimeError(f"Feedback write failed after {MAX_RETRIES} retries: {last_err}")


async def record_feedback(state: dict, action: str) -> dict:
    """
    Called by supervisor after operator action (Applied, Dismissed, Modified).
    Writes feedback to Postgres and publishes confirmation to Redis.
    """
    episode_id        = state.get("episode_id", "unknown")
    operator_id       = state.get("operator_id", "")
    modified_value    = state.get("modified_value")
    note              = state.get("operator_note")

    # Find the top approved recommendation for this episode
    recommendation_id = await _get_top_recommendation_id(episode_id)

    try:
        feedback_id = await _write_feedback_with_retry(
            recommendation_id=recommendation_id,
            operator_id=operator_id,
            action=action,
            modified_value=modified_value,
            note=note,
        )
        # Signal UI confirmation via Redis
        r = await get_redis()
        import json
        await r.setex(
            f"feedback_confirm:{episode_id}",
            300,
            json.dumps({"feedback_id": feedback_id, "action": action, "status": "confirmed"}),
        )
        return {"feedback_id": feedback_id}
    except Exception as e:
        log.error("operator_feedback_agent.failed", episode_id=episode_id, error=str(e))
        # Signal failure to UI
        r = await get_redis()
        import json
        await r.setex(
            f"feedback_confirm:{episode_id}",
            60,
            json.dumps({"status": "error", "error": str(e)}),
        )
        return {}


async def capture_from_api(
    episode_id: str,
    recommendation_id: str,
    operator_id: str,
    action: str,
    modified_value: float | None = None,
    note: str | None = None,
) -> str:
    """
    Called directly from the REST API endpoint (POST /feedback).
    Also publishes operator response to Redis so the supervisor can pick it up.
    """
    # Write feedback to Postgres
    feedback_id = await _write_feedback_with_retry(
        recommendation_id=recommendation_id,
        operator_id=operator_id,
        action=action,
        modified_value=modified_value,
        note=note,
    )

    # Signal supervisor's AwaitingOperator node via Redis
    r = await get_redis()
    import json
    await r.setex(
        f"operator_response:{episode_id}",
        settings_ttl := 300,
        json.dumps({
            "operator_action":  action,
            "operator_id":      operator_id,
            "modified_value":   modified_value,
            "operator_note":    note,
        }),
    )

    log.info("operator_feedback_agent.api_captured",
             episode_id=episode_id, action=action, feedback_id=feedback_id)
    return feedback_id


async def _get_top_recommendation_id(episode_id: str) -> str | None:
    """Fetch the most recent approved recommendation for this episode."""
    from app.db.models import Recommendation
    from sqlalchemy import select, desc

    async with get_session() as session:
        result = await session.execute(
            select(Recommendation)
            .where(Recommendation.episode_id == episode_id)
            .where(Recommendation.safety_status.in_(["approved", "clamped"]))
            .order_by(desc(Recommendation.created_at))
            .limit(1)
        )
        rec = result.scalar_one_or_none()
    return str(rec.recommendation_id) if rec else None
