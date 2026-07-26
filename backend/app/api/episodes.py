"""
GCIS Backend — Episodes API
Endpoints for episode state, predictions, root cause, and historical matches.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
import uuid

from app.db.session import get_db
from app.db.models import Transition, Prediction, Recommendation
from app.agents.supervisor_graph import start_episode

router = APIRouter()


@router.post("")
async def create_episode(
    machine_id: str,
    from_recipe_id: str,
    to_recipe_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Trigger a new episode manually (for testing/demo)."""
    episode_id = await start_episode(machine_id, from_recipe_id, to_recipe_id)
    return {"episode_id": episode_id, "status": "started"}


@router.get("/active")
async def get_active_episodes(db: AsyncSession = Depends(get_db)):
    """Return all currently active episodes (not closed)."""
    result = await db.execute(
        select(Transition)
        .where(Transition.current_state != "EpisodeClosed")
        .where(Transition.current_state != "Learning")
        .order_by(desc(Transition.started_at))
    )
    episodes = result.scalars().all()
    return [{"episode_id": str(e.episode_id), "machine": e.machine_id, "state": e.current_state} for e in episodes]


@router.get("/{episode_id}")
async def get_episode(episode_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Get basic transition details for an episode."""
    episode = await db.get(Transition, episode_id)
    if not episode:
        raise HTTPException(status_code=404, detail="Episode not found")
    return {
        "episode_id": str(episode.episode_id),
        "machine_id": episode.machine_id,
        "from_recipe_id": str(episode.from_recipe_id) if episode.from_recipe_id else None,
        "to_recipe_id": str(episode.to_recipe_id) if episode.to_recipe_id else None,
        "current_state": episode.current_state,
        "started_at": episode.started_at,
        "outcome": episode.outcome,
    }


@router.get("/{episode_id}/predictions")
async def get_predictions(episode_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Get the prediction history for an episode."""
    result = await db.execute(
        select(Prediction)
        .where(Prediction.episode_id == episode_id)
        .order_by(desc(Prediction.created_at))
    )
    predictions = result.scalars().all()
    return [
        {
            "p_offspec": float(p.p_offspec),
            "trajectory": p.trajectory,
            "eta_stabilize_min": float(p.eta_stabilize_min) if p.eta_stabilize_min else None,
            "confidence": float(p.confidence) if p.confidence else None,
            "created_at": p.created_at
        }
        for p in predictions
    ]


@router.get("/{episode_id}/recommendations")
async def get_recommendations(episode_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Get recommendations for an episode."""
    result = await db.execute(
        select(Recommendation)
        .where(Recommendation.episode_id == episode_id)
        .order_by(desc(Recommendation.created_at))
    )
    recs = result.scalars().all()
    return [
        {
            "id": str(r.recommendation_id),
            "variable_name": r.variable_name,
            "direction": r.direction,
            "proposed_value": float(r.proposed_value) if r.proposed_value else None,
            "safety_status": r.safety_status,
            "rejection_reason": r.rejection_reason
        }
        for r in recs
    ]


@router.get("/{episode_id}/historical-matches")
async def get_historical_matches(episode_id: uuid.UUID):
    """
    Mock endpoint: in a real system this might read from Qdrant directly or a cache.
    The dashboard agent pushes this live, so polling is rarely needed.
    """
    return {"matches": []}


@router.get("/{episode_id}/root-cause")
async def get_root_cause(episode_id: uuid.UUID):
    """
    Mock endpoint: The dashboard agent pushes this live via websocket.
    """
    return {"root_cause": None}
