from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.core.state import AppState
from app.dependencies import get_app_state
from backend.app.models.system import SystemStatus

router = APIRouter()


@router.get("/{series}", response_model=dict)
async def get_models_and_status(
    series: str,
    app_state: Annotated[AppState, Depends(get_app_state)],
):
    """Get available models and system status for series."""
    if series not in app_state.models:
        raise HTTPException(status_code=404, detail=f"Series {series} not found")
    return {
        "models": list(app_state.models[series].keys()),
        "system_status": SystemStatus(**app_state.system_status),
    }
