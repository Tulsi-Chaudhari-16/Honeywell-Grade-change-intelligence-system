"""
GCIS — Safety Validation Agent (CRITICAL GATE — Deterministic, NO LLM)
Validates every recommendation candidate against:
  - Recipe limits
  - Machine hard limits
  - Rate-of-change limits
  - Interaction rules

HARD RULE: Any rule evaluation error → reject entirely (fail closed).
Rejected candidates → audit_logs only. NEVER shown as actionable to operators.
"""
import time
import uuid
import structlog

from app.core.config import settings
from app.db.session import get_session
from app.db.models import Recommendation, AuditLog

log = structlog.get_logger(__name__)

# ─── Interaction Rules ────────────────────────────────────────────────────────
# Pairs of variables that cannot both be increased simultaneously beyond thresholds.
INTERACTION_RULES: list[dict] = [
    {
        "name": "speed_and_steam",
        "variables": ["machine_speed", "steam_pressure_p1"],
        "condition": "both_increase",
        "max_combined_delta_pct": 10.0,
        "reason": "Simultaneous speed and steam increases beyond 10% risk web breaks.",
    },
    {
        "name": "consistency_and_headbox",
        "variables": ["headbox_consistency", "headbox_pressure"],
        "condition": "both_increase",
        "max_combined_delta_pct": 15.0,
        "reason": "Simultaneous consistency and headbox pressure increase causes formation issues.",
    },
]


async def _load_machine_limits(machine_id: str) -> dict:
    """
    Load machine hard limits from Postgres. Short TTL cache via Redis.
    Never uses stale data beyond 60s.
    """
    from app.core.redis_client import get_redis
    import json

    r = await get_redis()
    cache_key = f"machine_limits:{machine_id}"
    cached = await r.get(cache_key)
    if cached:
        return json.loads(cached)

    from app.db.models import MachineLimit
    from sqlalchemy import select

    async with get_session() as session:
        result = await session.execute(
            select(MachineLimit).where(MachineLimit.machine_id == machine_id)
        )
        rows = result.scalars().all()

    limits = {
        row.variable_name: {
            "min": float(row.min_value or float("-inf")),
            "max": float(row.max_value or float("inf")),
            "max_rate": float(row.max_rate_of_change or float("inf")),
            "unit": row.unit or "",
        }
        for row in rows
    }

    await r.setex(cache_key, 60, json.dumps(limits))
    return limits


def _validate_candidate(candidate: dict, machine_limits: dict) -> tuple[str, str | None]:
    """
    Validate a single candidate.

    Returns:
        ("approved", None)          — passes all rules
        ("clamped", clamped_value)  — value out of range but clampable
        ("rejected", reason)        — fails hard rule, no clamp possible
    """
    var   = candidate.get("variable_name", "")
    value = candidate.get("proposed_value")
    direction = candidate.get("direction", "")

    if value is None:
        return "rejected", "No proposed_value provided."

    limits = machine_limits.get(var)
    if limits is None:
        # Variable not in machine limits — reject (unknown variable = unsafe)
        return "rejected", f"Variable '{var}' has no machine limit definition. Cannot validate."

    lo = limits["min"]
    hi = limits["max"]

    if lo <= value <= hi:
        return "approved", None

    # Out of range — attempt clamp
    clamped = max(lo, min(hi, value))
    log.warning("safety_validation_agent.clamping",
                var=var, proposed=value, clamped=clamped)
    return "clamped", str(clamped)


def _check_interaction_rules(candidates: list[dict]) -> tuple[bool, str | None]:
    """
    Check interaction rules across ALL candidates simultaneously.
    Returns (passed: bool, rejection_reason: str | None).
    """
    variable_directions = {
        c["variable_name"]: c.get("direction", "") for c in candidates
    }

    for rule in INTERACTION_RULES:
        vars_in_rule = rule["variables"]
        if rule["condition"] == "both_increase":
            if all(variable_directions.get(v) == "increase" for v in vars_in_rule):
                return False, rule["reason"]

    return True, None


async def _persist_candidates(
    episode_id: str,
    prediction_id: str,
    candidates: list[dict],
) -> list[str]:
    """Persist all candidates (approved, clamped, rejected) to Postgres."""
    ids = []
    async with get_session() as session:
        for c in candidates:
            rec_id = str(uuid.uuid4())
            clamped_val = c.get("clamped_value")
            session.add(Recommendation(
                recommendation_id=rec_id,
                episode_id=episode_id if episode_id != "unknown" else None,
                prediction_id=prediction_id if prediction_id else None,
                variable_name=c.get("variable_name"),
                direction=c.get("direction"),
                proposed_value=c.get("proposed_value"),
                clamped_value=float(clamped_val) if clamped_val else None,
                confidence=c.get("confidence", 0.0),
                historical_support=c.get("historical_support", []),
                safety_status=c.get("safety_status", "rejected"),
                rejection_reason=c.get("rejection_reason"),
            ))
            ids.append(rec_id)
    return ids


async def _write_audit(episode_id: str, candidates: list[dict]) -> None:
    """Write rejected candidates to immutable audit log."""
    rejected = [c for c in candidates if c.get("safety_status") == "rejected"]
    if not rejected:
        return
    async with get_session() as session:
        session.add(AuditLog(
            episode_id=episode_id if episode_id != "unknown" else None,
            actor="SafetyValidationAgent",
            action="candidates_rejected",
            before_state={"candidates": [c.get("variable_name") for c in rejected]},
            after_state={"reasons": [c.get("rejection_reason") for c in rejected]},
        ))


async def run(state: dict) -> dict:
    """
    LangGraph node entry point — CRITICAL GATE.
    Any exception here causes safety_rejected=True (fail closed).

    Returns: validated_candidates (approved+clamped only), safety_rejected flag.
    """
    episode_id    = state.get("episode_id", "unknown")
    machine_id    = state.get("machine_id", "PM1")
    raw_candidates = state.get("raw_candidates", [])
    prediction_id  = state.get("prediction_id", "")

    log.info("safety_validation_agent.run",
             episode_id=episode_id, candidates=len(raw_candidates))
    t0 = time.time()

    # ── HARD RULE: any exception = fail closed ────────────────────────────────
    try:
        machine_limits = await _load_machine_limits(machine_id)
    except Exception as e:
        log.error("safety_validation_agent.limits_load_failed", error=str(e))
        return {"validated_candidates": [], "safety_rejected": True}

    if not raw_candidates:
        return {"validated_candidates": [], "safety_rejected": True}

    # ── Per-candidate validation ──────────────────────────────────────────────
    evaluated: list[dict] = []
    try:
        for candidate in raw_candidates:
            try:
                status, clamped = _validate_candidate(candidate, machine_limits)
                evaluated.append({
                    **candidate,
                    "safety_status": status,
                    "clamped_value": clamped if status == "clamped" else None,
                    "rejection_reason": clamped if status == "rejected" else None,
                })
            except Exception as e:
                log.error("safety_validation_agent.candidate_eval_error",
                          variable=candidate.get("variable_name"), error=str(e))
                evaluated.append({
                    **candidate,
                    "safety_status": "rejected",
                    "rejection_reason": f"Rule evaluation error: {e}",
                })
    except Exception as e:
        log.error("safety_validation_agent.fatal_error", error=str(e))
        return {"validated_candidates": [], "safety_rejected": True}

    # ── Interaction rule check ────────────────────────────────────────────────
    try:
        passable = [c for c in evaluated if c["safety_status"] in ("approved", "clamped")]
        interaction_ok, interaction_reason = _check_interaction_rules(passable)
        if not interaction_ok:
            log.warning("safety_validation_agent.interaction_rule_failed",
                        reason=interaction_reason)
            for c in evaluated:
                if c["safety_status"] in ("approved", "clamped"):
                    c["safety_status"] = "rejected"
                    c["rejection_reason"] = interaction_reason
    except Exception as e:
        log.error("safety_validation_agent.interaction_check_error", error=str(e))
        return {"validated_candidates": [], "safety_rejected": True}

    # ── Persist all candidates (approved/clamped/rejected) to Postgres ────────
    await _persist_candidates(episode_id, prediction_id, evaluated)
    await _write_audit(episode_id, evaluated)

    validated = [c for c in evaluated if c["safety_status"] in ("approved", "clamped")]
    safety_rejected = len(validated) == 0

    latency_ms = int((time.time() - t0) * 1000)
    log.info("safety_validation_agent.done",
             episode_id=episode_id,
             approved=len(validated),
             rejected=len(evaluated) - len(validated),
             safety_rejected=safety_rejected,
             latency_ms=latency_ms)

    return {
        "validated_candidates": validated,
        "safety_rejected":      safety_rejected,
    }
