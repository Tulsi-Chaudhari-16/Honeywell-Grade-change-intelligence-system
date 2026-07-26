"""
GCIS Backend — Alerts API
Endpoints for fetching and acknowledging alerts.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from datetime import datetime
import uuid

from app.db.session import get_db
from app.db.models import Alert

router = APIRouter()


@router.get("")
async def get_alerts(
    severity: str | None = None,
    acknowledged: bool | None = None,
    db: AsyncSession = Depends(get_db)
):
    """Get active alerts."""
    query = select(Alert).order_by(desc(Alert.ts))
    
    if severity:
        query = query.where(Alert.severity == severity)
    if acknowledged is not None:
        if acknowledged:
            query = query.where(Alert.acknowledged_at.isnot(None))
        else:
            query = query.where(Alert.acknowledged_at.is_(None))

    result = await db.execute(query.limit(100))
    alerts = result.scalars().all()
    
    return [
        {
            "id": str(a.alert_id),
            "severity": a.severity,
            "type": a.alert_type,
            "message": a.message,
            "ts": a.ts,
            "acknowledged": a.acknowledged_at is not None
        }
        for a in alerts
    ]


@router.post("/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: uuid.UUID,
    operator_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
):
    """Acknowledge an alert."""
    alert = await db.get(Alert, alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
        
    if alert.acknowledged_at:
        return {"status": "already_acknowledged"}
        
    alert.acknowledged_at = datetime.utcnow()
    alert.acknowledged_by = operator_id
    
    await db.commit()
    return {"status": "acknowledged"}
