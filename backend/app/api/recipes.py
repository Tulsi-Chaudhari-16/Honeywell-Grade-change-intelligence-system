"""
GCIS Backend — Recipes API
Endpoints to fetch grade targets and machine limits.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uuid

from app.db.session import get_db
from app.db.models import Recipe, MachineLimit

router = APIRouter()


@router.get("")
async def list_recipes(db: AsyncSession = Depends(get_db)):
    """List all available grade recipes."""
    result = await db.execute(select(Recipe))
    recipes = result.scalars().all()
    return [
        {
            "id": str(r.recipe_id),
            "code": r.grade_code,
            "name": r.grade_name,
            "targets": {
                "basis_weight": float(r.target_basis_weight) if r.target_basis_weight else None,
                "moisture": float(r.target_moisture) if r.target_moisture else None,
                "ash": float(r.target_ash) if r.target_ash else None,
            }
        }
        for r in recipes
    ]


@router.get("/{recipe_id}/limits")
async def get_recipe_limits(recipe_id: uuid.UUID, machine_id: str, db: AsyncSession = Depends(get_db)):
    """Get machine hard limits for a specific recipe context."""
    recipe = await db.get(Recipe, recipe_id)
    if not recipe:
        raise HTTPException(status_code=404, detail="Recipe not found")

    result = await db.execute(
        select(MachineLimit).where(MachineLimit.machine_id == machine_id)
    )
    limits = result.scalars().all()
    
    return {
        "recipe_code": recipe.grade_code,
        "machine_limits": {
            l.variable_name: {
                "min": float(l.min_value) if l.min_value else None,
                "max": float(l.max_value) if l.max_value else None,
                "unit": l.unit
            }
            for l in limits
        }
    }
