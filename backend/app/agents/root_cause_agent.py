"""
GCIS — Root Cause Agent (LLM-backed via Groq)
Receives Prediction + SHAP RootCauseAttribution from ML partner.
LLM narrates — does NOT compute SHAP values.
"""
import time
from dataclasses import dataclass

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.llm_factory import get_llm

log = structlog.get_logger(__name__)

SYSTEM_PROMPT = """You are a paper-mill process engineer. You are given a ranked list of
SHAP feature attributions for a predicted Basis Weight excursion, plus
the current values and rates of change of the top factors, plus how far
each deviates from this grade transition's historical successful envelope.
Do not invent any numeric value not present in the input. Write a 2-3
sentence causal explanation in the style of an experienced operator's
shift note. Name the top 2 factors only. State direction (too high/too
low/too fast) and compare to the historical envelope in relative terms
only (e.g. "faster than typical successful transitions"), never inventing
a percentage that isn't in the input data."""


def _call_ml_shap(feature_vector: dict, prediction: dict) -> dict:
    """
    Call ML partner's SHAP output wrapper.
    Interface: ml/infer/explain(feature_vector, prediction) -> RootCauseAttribution

    Falls back to permutation importance stub if unavailable.
    """
    try:
        from ml.infer import explain  # type: ignore
        return explain(feature_vector, prediction)
    except ImportError:
        log.warning("root_cause_agent.shap_stub_used")
        return _shap_stub(feature_vector)
    except Exception as e:
        log.warning("root_cause_agent.shap_timeout", error=str(e))
        return _shap_stub(feature_vector, fallback=True)


def _shap_stub(feature_vector: dict, fallback: bool = False) -> dict:
    """
    Stub SHAP output using feature magnitude as proxy importance.
    Used until ML partner delivers explain() or on timeout.
    """
    features = feature_vector.get("features", feature_vector)
    if not features:
        features = {
            "steam_pressure_p1": 5.8,
            "machine_speed":     650,
            "headbox_pressure":  2.1,
        }

    sorted_features = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)
    ranked_factors = []
    for rank, (name, val) in enumerate(sorted_features[:5], start=1):
        ranked_factors.append({
            "feature_name":  name,
            "shap_value":    round(float(val) * 0.01, 4),
            "feature_value": float(val),
            "direction":     "too_high" if float(val) > 0 else "too_low",
            "rank":          rank,
        })

    return {
        "ranked_factors":     ranked_factors,
        "attribution_method": "permutation_fallback" if fallback else "shap_stub",
    }


def _build_user_prompt(root_cause_attr: dict, state: dict) -> str:
    """Build the grounded user prompt from SHAP data. No numbers invented here."""
    factors = root_cause_attr.get("ranked_factors", [])
    top2 = factors[:2]
    method = root_cause_attr.get("attribution_method", "shap")

    lines = [
        f"Grade transition: {state.get('from_recipe_id', 'unknown')} → {state.get('to_recipe_id', 'unknown')}",
        f"p_offspec: {state.get('p_offspec', 'N/A')}",
        f"Attribution method: {method}",
        "",
        "Top contributing factors:",
    ]
    for f in top2:
        lines.append(
            f"  - {f['feature_name']}: value={f['feature_value']}, "
            f"shap={f['shap_value']}, direction={f['direction']}, rank={f['rank']}"
        )

    lines += [
        "",
        "Write a 2-3 sentence causal explanation using ONLY the numbers above.",
        "Do not add any figures, percentages, or comparisons not present in the input.",
    ]
    return "\n".join(lines)


async def run(state: dict) -> dict:
    """
    LangGraph node entry point.
    1. Calls ML SHAP output wrapper (or fallback)
    2. Narrates with Groq/Llama — grounded strictly in provided numbers
    Returns: root_cause_report dict
    """
    episode_id     = state.get("episode_id", "unknown")
    feature_vector = state.get("feature_vector", {})

    log.info("root_cause_agent.run", episode_id=episode_id)
    t0 = time.time()

    # Step 1: Get SHAP attribution (ML partner or fallback)
    root_cause_attr = _call_ml_shap(feature_vector, state)

    # Step 2: Build prompt and call LLM
    narrative = "Not available"
    try:
        llm = get_llm(temperature=0.1)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=_build_user_prompt(root_cause_attr, state)),
        ]
        response = await llm.ainvoke(messages)
        narrative = response.content.strip()
    except Exception as e:
        log.error("root_cause_agent.llm_failed", error=str(e))
        narrative = "Root cause narration unavailable due to LLM error."

    latency_ms = int((time.time() - t0) * 1000)
    log.info("root_cause_agent.done",
             episode_id=episode_id,
             method=root_cause_attr.get("attribution_method"),
             latency_ms=latency_ms)

    root_cause_report = {
        "ranked_factors":     root_cause_attr.get("ranked_factors", []),
        "attribution_method": root_cause_attr.get("attribution_method", "unknown"),
        "narrative":          narrative,
        "latency_ms":         latency_ms,
    }

    return {"root_cause_report": root_cause_report}
