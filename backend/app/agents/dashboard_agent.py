"""
GCIS — Dashboard Agent (Deterministic)
Aggregates episode state and publishes to WebSocket via Redis pub/sub.
Called independently or hooked into state changes.
"""
import structlog

from app.core.redis_client import publish_episode_state

log = structlog.get_logger(__name__)


async def push_state(state: dict, override_node: str | None = None) -> None:
    """
    Format and publish current episode state to WebSocket clients.
    The supervisor calls this indirectly via `_log_state`.
    If an upstream agent fails, the state retains the last-known-good
    with a 'stale' or 'error' flag passed in.
    """
    episode_id = state.get("episode_id")
    if not episode_id:
        return

    payload = {
        "episode_id":         episode_id,
        "machine_id":         state.get("machine_id"),
        "current_node":       override_node or state.get("current_node"),
        "p_offspec":          state.get("p_offspec"),
        "trajectory":         state.get("trajectory"),
        "eta_stabilize_min":  state.get("eta_stabilize_min"),
        "confidence":         state.get("confidence"),
        "degraded_mode":      state.get("degraded_mode"),
        "explanation_card":   state.get("explanation_card"),
        "operator_action":    state.get("operator_action"),
        "errors":             state.get("errors", []),
    }
    
    await publish_episode_state(episode_id, payload)
    log.debug("dashboard_agent.state_pushed", episode_id=episode_id, node=payload["current_node"])
