"""
GCIS — Learning Agent (Mostly Deterministic)
On episode close: indexes to Qdrant, writes training data, updates drift metrics.
Emits RetrainRequested event to offline pipeline (not to any live agent).
"""
import time
import structlog
from datetime import datetime

from app.core.config import settings
from app.db.session import get_session
from app.db.models import Transition, FeatureSnapshot
from app.vectorstore.qdrant_client import upsert_episode
from app.agents.feature_engineering_agent import _call_ml_features

log = structlog.get_logger(__name__)


async def run(state: dict) -> dict:
    """
    LangGraph node entry point — called after EpisodeClosed.

    Steps:
    1. Determine outcome from operator action
    2. Upsert episode embedding to Qdrant
    3. Update Postgres transition outcome
    4. Emit RetrainRequested to offline pipeline
    """
    episode_id     = state.get("episode_id", "unknown")
    machine_id     = state.get("machine_id", "PM1")
    operator_action = state.get("operator_action", "timed_out")
    feature_vector  = state.get("feature_vector", {})
    from_recipe_id  = state.get("from_recipe_id", "")
    to_recipe_id    = state.get("to_recipe_id", "")

    log.info("learning_agent.run", episode_id=episode_id, operator_action=operator_action)
    t0 = time.time()

    # ── Determine outcome ─────────────────────────────────────────────────────
    outcome = _determine_outcome(state)

    # ── Generate embedding from feature vector ────────────────────────────────
    try:
        vector = await _get_embedding(feature_vector)
    except Exception as e:
        log.error("learning_agent.embed_failed", error=str(e))
        vector = None

    # ── Upsert to Qdrant ──────────────────────────────────────────────────────
    if vector:
        try:
            grade_pair = f"{from_recipe_id}_{to_recipe_id}"
            payload = {
                "episode_id":     episode_id,
                "machine_id":     machine_id,
                "grade_pair":     grade_pair,
                "outcome":        outcome,
                "operator_action": operator_action,
                "recovery_time_min": state.get("eta_stabilize_min"),
                "p_offspec":      state.get("p_offspec"),
                "predicted":      outcome == "unconfirmed",
                "indexed_at":     time.time(),
                "operator_actions": _extract_operator_actions(state),
            }
            await upsert_episode(episode_id, vector, payload)
            log.info("learning_agent.qdrant_indexed", episode_id=episode_id, outcome=outcome)
        except Exception as e:
            log.error("learning_agent.qdrant_failed", error=str(e))

    # ── Update Postgres transition outcome ────────────────────────────────────
    try:
        async with get_session() as session:
            from sqlalchemy import update
            await session.execute(
                update(Transition)
                .where(Transition.episode_id == episode_id)
                .values(
                    outcome=outcome,
                    ended_at=datetime.utcnow(),
                    embedding_id=episode_id if vector else None,
                    current_state="Learning",
                )
            )
        log.info("learning_agent.postgres_updated", episode_id=episode_id, outcome=outcome)
    except Exception as e:
        log.error("learning_agent.postgres_failed", error=str(e))

    # ── Emit RetrainRequested to offline pipeline (Kafka) ────────────────────
    await _emit_retrain_event(episode_id, outcome)

    latency_ms = int((time.time() - t0) * 1000)
    log.info("learning_agent.done", episode_id=episode_id, latency_ms=latency_ms)

    return {"learning_done": True}


def _determine_outcome(state: dict) -> str:
    """
    Determine episode outcome from operator action.
    'unconfirmed' if no lab-result confirmation available (archived but excluded from supervised retraining).
    """
    action = state.get("operator_action", "timed_out")
    if action == "accepted":
        return "success"
    elif action in ("rejected", "timed_out"):
        return "offspec"
    elif action == "modified":
        return "success"  # Operator took corrective action
    return "unconfirmed"


def _extract_operator_actions(state: dict) -> list[dict]:
    """Extract what the operator did — stored in Qdrant payload for future retrieval."""
    actions = []
    candidates = state.get("validated_candidates", [])
    operator_action = state.get("operator_action", "")
    modified_value = state.get("modified_value")

    for c in candidates:
        actions.append({
            "variable_name": c.get("variable_name"),
            "direction":     c.get("direction"),
            "proposed_value": c.get("proposed_value"),
            "operator_decision": operator_action,
            "final_value":   modified_value or c.get("clamped_value") or c.get("proposed_value"),
        })
    return actions


async def _get_embedding(feature_vector: dict) -> list[float]:
    """Get embedding from ML partner or stub."""
    from app.agents.historical_retrieval_agent import _call_ml_embed, _embed_stub
    vector = _call_ml_embed(feature_vector)
    if vector is None:
        vector = _embed_stub(feature_vector)
    return vector


async def _emit_retrain_event(episode_id: str, outcome: str) -> None:
    """
    Publish RetrainRequested event to Kafka offline topic.
    NOT consumed by any live agent — only by the offline ML retraining pipeline.
    """
    if outcome == "unconfirmed":
        log.info("learning_agent.retrain_skipped_unconfirmed", episode_id=episode_id)
        return

    try:
        from aiokafka import AIOKafkaProducer
        import json
        producer = AIOKafkaProducer(bootstrap_servers=settings.KAFKA_BROKERS)
        await producer.start()
        try:
            await producer.send_and_wait(
                "gcis.retrain_requests",
                json.dumps({
                    "event":      "RetrainRequested",
                    "episode_id": episode_id,
                    "outcome":    outcome,
                    "ts":         time.time(),
                }).encode(),
            )
            log.info("learning_agent.retrain_event_emitted", episode_id=episode_id)
        finally:
            await producer.stop()
    except Exception as e:
        log.warning("learning_agent.retrain_event_failed", error=str(e))
