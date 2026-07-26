"""
GCIS — Feature Engineering Agent (Deterministic)
Calls ml/features.py, publishes FeatureVector to Redis stream,
stores snapshot to Postgres feature_snapshots.
"""
import json
import time
import uuid
from datetime import datetime

import structlog

from app.core.config import settings
from app.core.redis_client import get_redis
from app.db.session import get_session
from app.db.models import FeatureSnapshot

log = structlog.get_logger(__name__)

REDIS_STREAM_KEY = "stream:feature_vectors"


def _call_ml_features(tag_window: dict) -> dict:
    """
    Call the ML partner's feature engineering module.
    Interface: ml/features.py::build_features(tag_window: dict) -> dict

    Falls back to a passthrough if ML module not yet available.
    """
    try:
        from ml.features import build_features  # type: ignore
        return build_features(tag_window)
    except ImportError:
        log.warning("feature_engineering_agent.ml_stub_used")
        # Stub: return raw tag values as features until ML partner delivers features.py
        return {
            "features": tag_window,
            "window_size": len(tag_window),
            "computed_at": time.time(),
            "stub": True,
        }
    except Exception as e:
        log.error("feature_engineering_agent.ml_error", error=str(e))
        raise


async def run(state: dict) -> dict:
    """
    LangGraph node entry point.
    Builds feature vector from current tag window, publishes to Redis stream,
    and persists snapshot to Postgres.

    Expects state to contain recent tag readings (collected by Data Agent).
    Returns: updated state with feature_vector key.
    """
    episode_id = state.get("episode_id", "unknown")
    log.info("feature_engineering_agent.run", episode_id=episode_id)

    # Collect recent tag values from Redis LKG store
    tag_window = await _collect_tag_window()

    if not tag_window:
        log.warning("feature_engineering_agent.no_tag_data", episode_id=episode_id)
        return {"feature_vector": {}, "feature_snapshot_id": None}

    # Call ML features module
    feature_vector = _call_ml_features(tag_window)

    # Publish to Redis stream
    r = await get_redis()
    await r.xadd(
        REDIS_STREAM_KEY,
        {
            "episode_id":     episode_id,
            "feature_vector": json.dumps(feature_vector),
            "ts":             str(time.time()),
        },
        maxlen=1000,   # rolling window
    )

    # Persist snapshot to Postgres
    snapshot_id = str(uuid.uuid4())
    now = datetime.utcnow()
    async with get_session() as session:
        session.add(FeatureSnapshot(
            snapshot_id=snapshot_id,
            episode_id=episode_id if episode_id != "unknown" else None,
            feature_vector=feature_vector,
            window_start=now,
            window_end=now,
        ))

    log.info("feature_engineering_agent.published",
             episode_id=episode_id, snapshot_id=snapshot_id,
             feature_count=len(feature_vector.get("features", {})))

    return {
        "feature_vector":      feature_vector,
        "feature_snapshot_id": snapshot_id,
    }


async def _collect_tag_window(tag_names: list[str] | None = None) -> dict:
    """
    Read last-known-good values for all monitored tags from Redis.
    Returns dict: {tag_name: value}
    """
    from app.core.redis_client import get_lkg
    from app.agents.data_agent import EXPECTED_UNITS

    all_tags = tag_names or list(EXPECTED_UNITS.keys())
    window: dict[str, float] = {}

    for tag in all_tags:
        lkg = await get_lkg(tag)
        if lkg:
            window[tag] = lkg["value"]

    return window
