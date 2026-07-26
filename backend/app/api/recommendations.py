"""
GCIS Backend — Recommendations API
Captures operator feedback (accept/reject/modify).
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
import uuid

from app.db.session import get_db
from app.db.models import Recommendation
from app.agents.operator_feedback_agent import capture_from_api

router = APIRouter()


class FeedbackRequest(BaseModel):
    operator_id: str
    action: str  # "accepted", "rejected", "modified"
    modified_value: float | None = None
    note: str | None = None


@router.post("/{recommendation_id}/feedback")
async def submit_feedback(
    recommendation_id: uuid.UUID,
    payload: FeedbackRequest,
    db: AsyncSession = Depends(get_db)
):
    """Submit operator feedback for a recommendation."""
    rec = await db.get(Recommendation, recommendation_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Recommendation not found")

    if payload.action not in ("accepted", "rejected", "modified"):
        raise HTTPException(status_code=400, detail="Invalid action")

    episode_id = str(rec.episode_id)

    # Capture feedback (writes to DB and signals supervisor via Redis)
    feedback_id = await capture_from_api(
        episode_id=episode_id,
        recommendation_id=str(recommendation_id),
        operator_id=payload.operator_id,
        action=payload.action,
        modified_value=payload.modified_value,
        note=payload.note,
    )

    return {"feedback_id": feedback_id, "status": "recorded"}
