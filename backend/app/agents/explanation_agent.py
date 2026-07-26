"""
GCIS — Explanation Agent (LLM-backed via Groq)
Synthesises ExplanationCard from RCA + Recommendations + Historical Matches.
All 7 mandatory fields always present. Missing data → "Not available", never fabricated.
"""
import time
import json
import re
import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.llm_factory import get_llm

log = structlog.get_logger(__name__)

MANDATORY_FIELDS = [
    "prediction_summary",
    "reason",
    "evidence",
    "historical_match",
    "confidence_statement",
    "expected_improvement",
    "safety_check_status",
]

SYSTEM_PROMPT = """You are synthesising an explanation card for a paper machine operator.
You will receive structured data: prediction result, root cause analysis, validated recommendations,
and historical matches. Produce a JSON object with EXACTLY these 7 fields:

{
  "prediction_summary":    "1-2 sentences on the prediction (p_offspec, trajectory direction)",
  "reason":                "1-2 sentences on the top root cause factor(s)",
  "evidence":              "Which SHAP factors support this, citing values from input only",
  "historical_match":      "Describe the most similar historical case and its outcome",
  "confidence_statement":  "State confidence level and attribution method factually",
  "expected_improvement":  "What improvement is expected from the top recommendation, per history",
  "safety_check_status":   "approved | clamped | rejected — with one-sentence explanation"
}

Rules:
- Use ONLY data present in the input. Never invent numbers, outcomes, or variables.
- If any input field is missing, write "Not available" for that output field.
- Respond with valid JSON only, no markdown fences."""


def _build_prompt(state: dict) -> str:
    root_cause     = state.get("root_cause_report", {})
    validated      = state.get("validated_candidates", [])
    historical     = state.get("historical_matches", [])
    p_offspec      = state.get("p_offspec", "N/A")
    confidence     = state.get("confidence", "N/A")
    trajectory     = state.get("trajectory", [])
    eta            = state.get("eta_stabilize_min", "N/A")
    degraded       = state.get("degraded_mode", False)

    top_candidate  = validated[0] if validated else {}
    top_match      = historical[0] if historical else {}
    safety_status  = top_candidate.get("safety_status", "rejected") if validated else "rejected"

    lines = [
        f"p_offspec: {p_offspec}",
        f"confidence: {confidence}",
        f"trajectory (next 5 steps): {trajectory[:5]}",
        f"eta_stabilize_min: {eta}",
        f"degraded_mode: {degraded}",
        f"attribution_method: {root_cause.get('attribution_method', 'N/A')}",
        f"root_cause_narrative: {root_cause.get('narrative', 'N/A')}",
        "",
        "Top SHAP factors:",
    ]
    for f in root_cause.get("ranked_factors", [])[:3]:
        lines.append(f"  {f['feature_name']}: {f['feature_value']} ({f['direction']})")

    lines += ["", "Top validated recommendation:"]
    if top_candidate:
        lines += [
            f"  variable: {top_candidate.get('variable_name', 'N/A')}",
            f"  direction: {top_candidate.get('direction', 'N/A')}",
            f"  proposed_value: {top_candidate.get('proposed_value', 'N/A')}",
            f"  safety_status: {safety_status}",
            f"  expected_improvement: {top_candidate.get('expected_improvement', 'N/A')}",
            f"  historical_support: {top_candidate.get('historical_support', [])}",
        ]
    else:
        lines.append("  None (all candidates rejected by safety gate)")

    lines += ["", "Most similar historical match:"]
    if top_match:
        payload = top_match.get("payload", {})
        lines += [
            f"  outcome: {payload.get('outcome', 'N/A')}",
            f"  recovery_time_min: {payload.get('recovery_time_min', 'N/A')}",
            f"  score: {round(top_match.get('score', 0), 3)}",
        ]
    else:
        lines.append("  No historical match available")

    return "\n".join(lines)


def _ensure_mandatory_fields(card: dict) -> dict:
    """Guarantee all 7 fields are present. Missing → 'Not available'."""
    for field in MANDATORY_FIELDS:
        if field not in card or not card[field]:
            card[field] = "Not available"
    return card


async def run(state: dict) -> dict:
    """
    LangGraph node entry point.
    Synthesises ExplanationCard with all 7 mandatory fields.
    Any LLM failure → returns safe fallback card with 'Not available' fields.
    """
    episode_id = state.get("episode_id", "unknown")
    log.info("explanation_agent.run", episode_id=episode_id)
    t0 = time.time()

    explanation_card: dict = {}

    try:
        llm = get_llm(temperature=0.1)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=_build_prompt(state)),
        ]
        response = await llm.ainvoke(messages)
        content = response.content.strip()

        # Parse JSON response
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            explanation_card = json.loads(json_match.group())
        else:
            log.warning("explanation_agent.json_parse_failed", raw=content[:200])

    except Exception as e:
        log.error("explanation_agent.llm_failed", error=str(e))

    # Guarantee all 7 mandatory fields
    explanation_card = _ensure_mandatory_fields(explanation_card)

    latency_ms = int((time.time() - t0) * 1000)
    log.info("explanation_agent.done", episode_id=episode_id, latency_ms=latency_ms)

    return {"explanation_card": explanation_card}
