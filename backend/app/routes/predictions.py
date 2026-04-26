from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.config import LOGGER
from app.core.state import AppState
from app.dependencies import get_app_state, get_data_service
from app.models.predictions import PredictionsResponse
from app.services.data_service import DataService
from app.services.prediction_service import PredictionService

router = APIRouter()


@router.get("/{series}", response_model=PredictionsResponse)
async def get_predictions(
    series: str,
    app_state: Annotated[AppState, Depends(get_app_state)],
    data_service: Annotated[DataService, Depends(get_data_service)],
):
    """Get predictions from all models."""
    try:
        prediction_service = PredictionService(app_state, series, data_service)
        return await prediction_service.get_predictions()
    except Exception as e:
        LOGGER.error(f"Error in get_predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))
