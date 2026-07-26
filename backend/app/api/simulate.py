"""
GCIS Backend — Simulation API (Digital Twin What-If)
Re-runs Feature Engineering + Prediction Agent pipeline with overridden feature values.
Also runs Safety Validation Agent on simulated proposals.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Dict
from sqlalchemy.ext.asyncio import AsyncSession
import uuid

from app.db.session import get_db
from app.db.models import Transition
from app.agents.feature_engineering_agent import _call_ml_features, _collect_tag_window
from app.agents.prediction_agent import _call_ml_inference
from app.agents.safety_validation_agent import _validate_candidate, _load_machine_limits

router = APIRouter()


class SimulateRequest(BaseModel):
    episode_id: str
    overrides: Dict[str, float]


@router.post("")
async def simulate(
    payload: SimulateRequest,
    db: AsyncSession = Depends(get_db)
):
    """Run a what-if simulation by overriding specific tag values."""
    episode_id = payload.episode_id
    
    # 1. Validate episode exists
    if episode_id != "test":
        try:
            episode = await db.get(Transition, uuid.UUID(episode_id))
            if not episode:
                raise HTTPException(status_code=404, detail="Episode not found")
            machine_id = episode.machine_id
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid episode_id format")
    else:
        machine_id = "PM1"
        
    # 2. Get current tag window and apply overrides
    tag_window = await _collect_tag_window()
    for k, v in payload.overrides.items():
        tag_window[k] = v
        
    # 3. Generate feature vector (re-run FE agent logic)
    feature_vector = _call_ml_features(tag_window)
    
    # 4. Generate prediction (re-run Prediction agent logic)
    prediction = _call_ml_inference(feature_vector)
    
    # 5. Run safety validation on the proposed overrides
    safety_flags = []
    try:
        machine_limits = await _load_machine_limits(machine_id)
        for k, v in payload.overrides.items():
            # Mock a candidate structure for the validator
            candidate = {"variable_name": k, "proposed_value": v, "direction": "override"}
            status, clamped = _validate_candidate(candidate, machine_limits)
            safety_flags.append({
                "variable": k,
                "status": status,
                "clamped_value": clamped,
            })
    except Exception as e:
        safety_flags.append({"error": str(e)})

    return {
        "trajectory": prediction.get("trajectory", []),
        "p_offspec": prediction.get("p_offspec", 0.0),
        "safety_flags": safety_flags
    }
