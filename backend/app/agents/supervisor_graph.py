"""
GCIS — Supervisor Agent
LangGraph episode state machine. Routes all 13 agents through the episode lifecycle.
"""
import asyncio
import time
import uuid
from datetime import datetime
from typing import Any, Literal, TypedDict

import structlog
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from app.core.config import settings
from app.core.redis_client import get_redis, publish_episode_state
from app.db.session import get_session
from app.db.models import Transition, AuditLog

log = structlog.get_logger(__name__)


# ─── Episode State Schema ─────────────────────────────────────────────────────

class EpisodeState(TypedDict, total=False):
    episode_id:         str
    machine_id:         str
    from_recipe_id:     str
    to_recipe_id:       str
    current_node:       str

    # Prediction Agent output
    p_offspec:          float
    trajectory:         list[float]
    eta_stabilize_min:  float
    confidence:         float
    model_version:      str
    degraded_mode:      bool
    prediction_id:      str

    # Root Cause Agent output
    root_cause_report:  dict

    # Historical Retrieval Agent output
    historical_matches: list[dict]

    # Recommendation Agent output
    raw_candidates:     list[dict]

    # Safety Validation Agent output
    validated_candidates: list[dict]
    safety_rejected:    bool

    # Explanation Agent output
    explanation_card:   dict

    # Operator Feedback Agent output
    operator_action:    str | None   # accepted | rejected | modified | timed_out
    operator_note:      str | None
    modified_value:     float | None
    await_start_ts:     float | None

    # Learning Agent output
    learning_done:      bool

    # Error tracking
    errors:             list[str]


# ─── Guard conditions ─────────────────────────────────────────────────────────

def should_escalate(state: EpisodeState) -> Literal["RootCause", "Monitoring"]:
    p = state.get("p_offspec", 0.0)
    c = state.get("confidence", 0.0)
    if p >= settings.ALERT_THRESHOLD and c >= settings.MIN_CONFIDENCE:
        log.info("supervisor.escalating", p_offspec=p, confidence=c)
        return "RootCause"
    return "Monitoring"


def safety_gate(state: EpisodeState) -> Literal["Explaining", "Rejected"]:
    if state.get("safety_rejected", False):
        log.warning("supervisor.safety_rejected", episode_id=state.get("episode_id"))
        return "Rejected"
    return "Explaining"


def operator_response(state: EpisodeState) -> Literal["Applied", "Dismissed", "Modified"]:
    action = state.get("operator_action", "timed_out")
    if action == "accepted":
        return "Applied"
    elif action == "modified":
        return "Modified"
    return "Dismissed"


def episode_done(state: EpisodeState) -> Literal["Learning", END]:
    return "Learning"


# ─── Node stubs (each imports its own agent module) ──────────────────────────

async def node_monitoring(state: EpisodeState) -> EpisodeState:
    """Idle monitoring — wait for new feature data."""
    log.info("supervisor.node", node="Monitoring", episode_id=state.get("episode_id"))
    await _log_state(state, "Monitoring")
    return {**state, "current_node": "Monitoring"}


async def node_feature_building(state: EpisodeState) -> EpisodeState:
    from app.agents.feature_engineering_agent import run as fe_run
    log.info("supervisor.node", node="FeatureBuilding", episode_id=state.get("episode_id"))
    result = await fe_run(state)
    await _log_state({**state, **result}, "FeatureBuilding")
    return {**state, **result, "current_node": "FeatureBuilding"}


async def node_predicting(state: EpisodeState) -> EpisodeState:
    from app.agents.prediction_agent import run as pred_run
    log.info("supervisor.node", node="Predicting", episode_id=state.get("episode_id"))
    result = await pred_run(state)
    await _log_state({**state, **result}, "Predicting")
    return {**state, **result, "current_node": "Predicting"}


async def node_root_cause(state: EpisodeState) -> EpisodeState:
    from app.agents.root_cause_agent import run as rc_run
    log.info("supervisor.node", node="RootCause", episode_id=state.get("episode_id"))
    result = await rc_run(state)
    await _log_state({**state, **result}, "RootCause")
    return {**state, **result, "current_node": "RootCause"}


async def node_retrieving(state: EpisodeState) -> EpisodeState:
    from app.agents.historical_retrieval_agent import run as hr_run
    log.info("supervisor.node", node="Retrieving", episode_id=state.get("episode_id"))
    result = await hr_run(state)
    await _log_state({**state, **result}, "Retrieving")
    return {**state, **result, "current_node": "Retrieving"}


async def node_recommending(state: EpisodeState) -> EpisodeState:
    from app.agents.recommendation_agent import run as rec_run
    log.info("supervisor.node", node="Recommending", episode_id=state.get("episode_id"))
    result = await rec_run(state)
    await _log_state({**state, **result}, "Recommending")
    return {**state, **result, "current_node": "Recommending"}


async def node_validating(state: EpisodeState) -> EpisodeState:
    from app.agents.safety_validation_agent import run as sv_run
    log.info("supervisor.node", node="Validating", episode_id=state.get("episode_id"))
    result = await sv_run(state)
    await _log_state({**state, **result}, "Validating")
    return {**state, **result, "current_node": "Validating"}


async def node_rejected(state: EpisodeState) -> EpisodeState:
    log.warning("supervisor.node", node="Rejected", episode_id=state.get("episode_id"))
    await _audit(state, "SafetyValidationAgent", "recommendation_rejected",
                 before={"candidates": state.get("raw_candidates")},
                 after={"reason": "safety_gate_failed"})
    await _log_state(state, "Rejected")
    return {**state, "current_node": "Rejected"}


async def node_explaining(state: EpisodeState) -> EpisodeState:
    from app.agents.explanation_agent import run as exp_run
    log.info("supervisor.node", node="Explaining", episode_id=state.get("episode_id"))
    result = await exp_run(state)
    await _log_state({**state, **result}, "Explaining")
    return {**state, **result, "current_node": "Explaining"}


async def node_awaiting_operator(state: EpisodeState) -> EpisodeState:
    log.info("supervisor.node", node="AwaitingOperator", episode_id=state.get("episode_id"))
    # Record when we entered AwaitingOperator for timeout tracking
    state = {**state, "await_start_ts": time.time(), "current_node": "AwaitingOperator"}
    await _log_state(state, "AwaitingOperator")
    # Wait for operator action or timeout
    timeout = settings.OPERATOR_AWAIT_TIMEOUT_SEC
    start = time.time()
    r = await get_redis()
    key = f"operator_response:{state['episode_id']}"
    while time.time() - start < timeout:
        response = await r.get(key)
        if response:
            import json
            data = json.loads(response)
            await r.delete(key)
            return {**state, **data, "current_node": "AwaitingOperator"}
        await asyncio.sleep(1)
    log.warning("supervisor.operator_timeout", episode_id=state.get("episode_id"))
    return {**state, "operator_action": "timed_out", "current_node": "AwaitingOperator"}


async def node_applied(state: EpisodeState) -> EpisodeState:
    from app.agents.operator_feedback_agent import record_feedback
    log.info("supervisor.node", node="Applied", episode_id=state.get("episode_id"))
    await record_feedback(state, "accepted")
    await _log_state(state, "Applied")
    return {**state, "current_node": "Applied"}


async def node_dismissed(state: EpisodeState) -> EpisodeState:
    from app.agents.operator_feedback_agent import record_feedback
    log.info("supervisor.node", node="Dismissed", episode_id=state.get("episode_id"))
    await record_feedback(state, state.get("operator_action", "rejected"))
    await _log_state(state, "Dismissed")
    return {**state, "current_node": "Dismissed"}


async def node_modified(state: EpisodeState) -> EpisodeState:
    from app.agents.operator_feedback_agent import record_feedback
    log.info("supervisor.node", node="Modified", episode_id=state.get("episode_id"))
    await record_feedback(state, "modified")
    await _log_state(state, "Modified")
    return {**state, "current_node": "Modified"}


async def node_episode_closed(state: EpisodeState) -> EpisodeState:
    log.info("supervisor.node", node="EpisodeClosed", episode_id=state.get("episode_id"))
    await _close_episode_in_db(state)
    await _log_state(state, "EpisodeClosed")
    return {**state, "current_node": "EpisodeClosed"}


async def node_learning(state: EpisodeState) -> EpisodeState:
    from app.agents.learning_agent import run as learn_run
    log.info("supervisor.node", node="Learning", episode_id=state.get("episode_id"))
    result = await learn_run(state)
    await _log_state({**state, **result}, "Learning")
    return {**state, **result, "current_node": "Learning", "learning_done": True}


# ─── Graph Construction ───────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    g = StateGraph(EpisodeState)

    # Register nodes
    g.add_node("Monitoring",       node_monitoring)
    g.add_node("FeatureBuilding",  node_feature_building)
    g.add_node("Predicting",       node_predicting)
    g.add_node("RootCause",        node_root_cause)
    g.add_node("Retrieving",       node_retrieving)
    g.add_node("Recommending",     node_recommending)
    g.add_node("Validating",       node_validating)
    g.add_node("Rejected",         node_rejected)
    g.add_node("Explaining",       node_explaining)
    g.add_node("AwaitingOperator", node_awaiting_operator)
    g.add_node("Applied",          node_applied)
    g.add_node("Dismissed",        node_dismissed)
    g.add_node("Modified",         node_modified)
    g.add_node("EpisodeClosed",    node_episode_closed)
    g.add_node("Learning",         node_learning)

    # Entry point
    g.set_entry_point("Monitoring")

    # Edges
    g.add_edge("Monitoring",       "FeatureBuilding")
    g.add_edge("FeatureBuilding",  "Predicting")
    g.add_conditional_edges("Predicting", should_escalate,
                            {"Monitoring": "Monitoring", "RootCause": "RootCause"})
    g.add_edge("RootCause",        "Retrieving")
    g.add_edge("Retrieving",       "Recommending")
    g.add_edge("Recommending",     "Validating")
    g.add_conditional_edges("Validating", safety_gate,
                            {"Explaining": "Explaining", "Rejected": "Rejected"})
    g.add_edge("Rejected",         "Monitoring")
    g.add_edge("Explaining",       "AwaitingOperator")
    g.add_conditional_edges("AwaitingOperator", operator_response,
                            {"Applied": "Applied", "Dismissed": "Dismissed", "Modified": "Modified"})
    g.add_edge("Applied",          "EpisodeClosed")
    g.add_edge("Dismissed",        "EpisodeClosed")
    g.add_edge("Modified",         "EpisodeClosed")
    g.add_edge("EpisodeClosed",    "Learning")
    g.add_edge("Learning",         END)

    return g


# ─── Helper utilities ─────────────────────────────────────────────────────────

async def _log_state(state: EpisodeState, node: str) -> None:
    """Push current state to WebSocket clients via Redis pub/sub."""
    await publish_episode_state(state.get("episode_id", "unknown"), {
        "current_node": node,
        "episode_id": state.get("episode_id"),
        "p_offspec": state.get("p_offspec"),
        "explanation_card": state.get("explanation_card"),
        "operator_action": state.get("operator_action"),
    })


async def _audit(state: EpisodeState, actor: str, action: str,
                 before: dict, after: dict) -> None:
    """Write an immutable audit log entry."""
    from app.db.models import AuditLog
    async with get_session() as session:
        session.add(AuditLog(
            episode_id=state.get("episode_id"),
            actor=actor,
            action=action,
            before_state=before,
            after_state=after,
        ))


async def _close_episode_in_db(state: EpisodeState) -> None:
    """Mark episode as ended in Postgres."""
    async with get_session() as session:
        from sqlalchemy import update
        await session.execute(
            update(Transition)
            .where(Transition.episode_id == state["episode_id"])
            .values(ended_at=datetime.utcnow(), current_state="EpisodeClosed")
        )


# ─── Public API ───────────────────────────────────────────────────────────────

async def start_episode(
    machine_id: str,
    from_recipe_id: str,
    to_recipe_id: str,
) -> str:
    """Create a new episode in Postgres and start the LangGraph state machine."""
    episode_id = str(uuid.uuid4())
    async with get_session() as session:
        transition = Transition(
            episode_id=episode_id,
            machine_id=machine_id,
            from_recipe_id=from_recipe_id,
            to_recipe_id=to_recipe_id,
            started_at=datetime.utcnow(),
            current_state="Monitoring",
        )
        session.add(transition)

    graph = build_graph().compile()
    initial_state: EpisodeState = {
        "episode_id": episode_id,
        "machine_id": machine_id,
        "from_recipe_id": from_recipe_id,
        "to_recipe_id": to_recipe_id,
        "current_node": "Monitoring",
        "errors": [],
    }

    # Run graph asynchronously (non-blocking)
    asyncio.create_task(graph.ainvoke(initial_state))
    log.info("supervisor.episode_started", episode_id=episode_id)
    return episode_id
