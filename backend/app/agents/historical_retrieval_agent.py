"""
GCIS — Historical Retrieval Agent (LLM-backed narration only)
Embeds current episode FeatureVector → Qdrant ANN search → narrates top matches.
Fallback: Postgres SQL Euclidean-distance query if Qdrant unavailable.
"""
import time
import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.llm_factory import get_llm
from app.vectorstore.qdrant_client import search_similar

log = structlog.get_logger(__name__)

SYSTEM_PROMPT = """You are a paper-mill process historian assistant.
You are given structured records of the 3 most similar historical grade transitions.
Each record contains: what operators changed, whether the transition succeeded,
and how long recovery took.
Do not invent any outcome or action not present in the input records.
Write 2-3 sentences summarising the most relevant patterns for the current situation.
Name the specific variables operators changed. State outcomes factually."""


def _call_ml_embed(feature_vector: dict) -> list[float] | None:
    """
    Call ML partner's embedding model.
    Interface: ml/infer/embed_transition(feature_vector_sequence) -> np.ndarray

    Returns None on failure (triggers SQL fallback in search_similar).
    """
    try:
        import numpy as np
        from ml.infer import embed_transition  # type: ignore
        vec = embed_transition(feature_vector)
        return vec.tolist() if hasattr(vec, "tolist") else list(vec)
    except ImportError:
        log.warning("historical_retrieval_agent.embed_stub_used")
        return _embed_stub(feature_vector)
    except Exception as e:
        log.error("historical_retrieval_agent.embed_failed", error=str(e))
        return None


def _embed_stub(feature_vector: dict) -> list[float]:
    """Stub embedding: L2-normalize feature values into a fixed-dim vector."""
    import math
    from app.core.config import settings

    features = feature_vector.get("features", feature_vector)
    values = list(features.values())[:settings.QDRANT_VECTOR_SIZE]

    # Pad or trim to QDRANT_VECTOR_SIZE
    values = (values + [0.0] * settings.QDRANT_VECTOR_SIZE)[:settings.QDRANT_VECTOR_SIZE]
    norm = math.sqrt(sum(v ** 2 for v in values)) or 1.0
    return [v / norm for v in values]


def _build_narration_prompt(matches: list[dict], state: dict) -> str:
    """Build grounded narration prompt — only facts from retrieved records."""
    grade_pair = f"{state.get('from_recipe_id', '?')} → {state.get('to_recipe_id', '?')}"
    lines = [
        f"Current grade transition: {grade_pair}",
        f"Current p_offspec: {state.get('p_offspec', 'N/A')}",
        "",
        "Top 3 similar historical transitions:",
    ]
    for i, m in enumerate(matches[:3], start=1):
        payload = m.get("payload", {})
        lines += [
            f"  Match {i} (similarity score: {round(m.get('score', 0), 3)}):",
            f"    Episode: {payload.get('episode_id', 'unknown')}",
            f"    Outcome: {payload.get('outcome', 'unknown')}",
            f"    Recovery time: {payload.get('recovery_time_min', 'N/A')} min",
            f"    Operator actions: {payload.get('operator_actions', 'not recorded')}",
        ]

    lines += [
        "",
        "Summarise patterns in 2-3 sentences using only the data above.",
        "Do not invent actions or outcomes not listed.",
    ]
    return "\n".join(lines)


async def run(state: dict) -> dict:
    """
    LangGraph node entry point.
    1. Embeds current feature vector
    2. Searches Qdrant (or SQL fallback)
    3. Narrates top matches with Groq/Llama
    Returns: historical_matches list + narration
    """
    episode_id     = state.get("episode_id", "unknown")
    feature_vector = state.get("feature_vector", {})
    grade_pair     = f"{state.get('from_recipe_id','')}_{state.get('to_recipe_id','')}"

    log.info("historical_retrieval_agent.run", episode_id=episode_id)
    t0 = time.time()

    # Step 1: Embed
    vector = _call_ml_embed(feature_vector)
    if vector is None:
        vector = _embed_stub(feature_vector)

    # Step 2: ANN search (Qdrant with grade_pair filter; SQL fallback inside)
    matches = await search_similar(vector, grade_pair=grade_pair, top_k=5)
    top3 = matches[:3]

    # Step 3: Narrate
    narration = "Not available"
    try:
        if top3:
            llm = get_llm(temperature=0.1)
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=_build_narration_prompt(top3, state)),
            ]
            response = await llm.ainvoke(messages)
            narration = response.content.strip()
        else:
            narration = "No similar historical transitions found."
    except Exception as e:
        log.error("historical_retrieval_agent.llm_failed", error=str(e))
        narration = "Historical narration unavailable."

    latency_ms = int((time.time() - t0) * 1000)
    log.info("historical_retrieval_agent.done",
             episode_id=episode_id, matches_found=len(top3), latency_ms=latency_ms)

    return {
        "historical_matches": [
            {**m, "narration": narration if i == 0 else None}
            for i, m in enumerate(top3)
        ],
    }
