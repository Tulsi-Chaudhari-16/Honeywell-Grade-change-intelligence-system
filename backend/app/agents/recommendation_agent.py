"""
GCIS — Recommendation Agent (LLM-backed via Groq)
Generates candidates ONLY from:
  (a) Variables flagged by Root Cause Agent
  (b) Actions observed in historical matches
LLM selects and narrates — cannot propose anything outside these lists.
All candidates sent to Safety Validation Agent before operator sees them.
"""
import time
import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.llm_factory import get_llm
from app.db.session import get_session

log = structlog.get_logger(__name__)

SYSTEM_PROMPT = """You are proposing corrective setpoint changes for a paper machine grade
transition. You are given: the top contributing factors to a predicted
off-spec event, the 3 most similar historical transitions and what
operators changed in each, and the hard recipe/machine limits for this
grade. Propose at most 3 candidate actions, each expressed as a variable,
a direction, and a bounded magnitude (percentage or absolute unit) that
a human operator could apply. Every magnitude you propose MUST fall
strictly within the provided recipe/machine limits — if you cannot justify
a magnitude from the historical matches or the root-cause factors, do not
propose one. For each candidate, state which historical transition(s)
support it and the expected improvement observed in that history. Do not
fabricate a historical match that wasn't provided to you.

Respond ONLY in this JSON format (array of objects):
[
  {
    "variable_name": "string",
    "direction": "increase" or "decrease",
    "proposed_value": number,
    "unit": "string",
    "historical_support": ["episode_id_1"],
    "expected_improvement": "string (factual, from history only)"
  }
]"""


async def _get_recipe_limits(to_recipe_id: str, machine_id: str) -> dict:
    """Fetch recipe targets and machine limits from Postgres."""
    from app.db.models import Recipe, MachineLimit
    from sqlalchemy import select

    limits = {}
    async with get_session() as session:
        # Recipe targets
        recipe = await session.get(Recipe, to_recipe_id)
        if recipe:
            limits["recipe"] = {
                "grade_code":          recipe.grade_code,
                "target_basis_weight": float(recipe.target_basis_weight or 0),
                "target_moisture":     float(recipe.target_moisture or 0),
            }

        # Machine hard limits
        result = await session.execute(
            select(MachineLimit).where(MachineLimit.machine_id == machine_id)
        )
        ml_rows = result.scalars().all()
        limits["machine_limits"] = {
            row.variable_name: {
                "min": float(row.min_value or 0),
                "max": float(row.max_value or 0),
                "max_rate": float(row.max_rate_of_change or 0),
                "unit": row.unit,
            }
            for row in ml_rows
        }
    return limits


def _extract_candidate_variables(root_cause_report: dict, historical_matches: list) -> list[str]:
    """
    Build the allowed candidate variable list:
    (a) Top factors from RCA
    (b) Variables mentioned in historical matches
    LLM is constrained to ONLY this list.
    """
    variables = set()

    # From RCA
    for f in root_cause_report.get("ranked_factors", []):
        variables.add(f["feature_name"])

    # From historical matches
    for match in historical_matches:
        actions = match.get("payload", {}).get("operator_actions", "")
        if isinstance(actions, list):
            for a in actions:
                if isinstance(a, dict) and "variable_name" in a:
                    variables.add(a["variable_name"])

    return list(variables)


def _build_prompt(state: dict, limits: dict, allowed_variables: list) -> str:
    """Build fully grounded recommendation prompt."""
    root_cause = state.get("root_cause_report", {})
    historical  = state.get("historical_matches", [])

    lines = [
        f"Machine: {state.get('machine_id', 'PM1')}",
        f"Grade transition: {state.get('from_recipe_id', '?')} → {state.get('to_recipe_id', '?')}",
        f"p_offspec: {state.get('p_offspec', 'N/A')}",
        "",
        "Root cause factors (ONLY propose changes to these variables):",
    ]
    for f in root_cause.get("ranked_factors", [])[:3]:
        lines.append(
            f"  - {f['feature_name']}: {f['feature_value']} ({f['direction']}), "
            f"shap={f['shap_value']}"
        )

    lines += ["", "Historical match operator actions:"]
    for i, m in enumerate(historical[:3], 1):
        payload = m.get("payload", {})
        lines.append(
            f"  Match {i}: outcome={payload.get('outcome','?')}, "
            f"actions={payload.get('operator_actions','none recorded')}"
        )

    lines += ["", "Hard machine limits (MUST NOT be violated):"]
    for var, lim in limits.get("machine_limits", {}).items():
        if var in allowed_variables:
            lines.append(
                f"  {var}: min={lim['min']}, max={lim['max']}, "
                f"max_rate={lim['max_rate']} {lim['unit']}"
            )

    lines += [
        "",
        f"Allowed variables (ONLY these): {', '.join(allowed_variables)}",
        "Respond with valid JSON array only.",
    ]
    return "\n".join(lines)


async def run(state: dict) -> dict:
    """
    LangGraph node entry point.
    Generates constrained recommendation candidates via Groq/Llama.
    Returns: raw_candidates list (not yet safety-validated).
    """
    episode_id = state.get("episode_id", "unknown")
    log.info("recommendation_agent.run", episode_id=episode_id)
    t0 = time.time()

    root_cause_report  = state.get("root_cause_report", {})
    historical_matches = state.get("historical_matches", [])
    to_recipe_id       = state.get("to_recipe_id", "")
    machine_id         = state.get("machine_id", "PM1")

    # Get limits and allowed variables
    limits            = await _get_recipe_limits(to_recipe_id, machine_id)
    allowed_variables = _extract_candidate_variables(root_cause_report, historical_matches)

    if not allowed_variables:
        log.warning("recommendation_agent.no_allowed_variables", episode_id=episode_id)
        return {"raw_candidates": []}

    raw_candidates = []
    try:
        llm = get_llm(temperature=0.1)
        prompt = _build_prompt(state, limits, allowed_variables)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        response = await llm.ainvoke(messages)
        content = response.content.strip()

        # Parse JSON response
        import json, re
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            raw_candidates = json.loads(json_match.group())
            # Enforce: only allowed variables
            raw_candidates = [
                c for c in raw_candidates
                if c.get("variable_name") in allowed_variables
            ]
    except Exception as e:
        log.error("recommendation_agent.llm_failed", error=str(e))

    latency_ms = int((time.time() - t0) * 1000)
    log.info("recommendation_agent.done",
             episode_id=episode_id, candidates=len(raw_candidates), latency_ms=latency_ms)

    return {
        "raw_candidates": raw_candidates,
        "recommendation_limits": limits,
    }
