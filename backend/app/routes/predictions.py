import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.core.state import AppState
from app.dependencies import get_app_state, get_data_service
from app.models.predictions import ModelResults
from app.services.data_service import DataService
from app.services.prediction_service import PredictionService

log = logging.getLogger(__name__)
router = APIRouter()


@router.get("/{series}/{model}", response_model=ModelResults)
async def get_prediction(
    series: str,
    model: str,
    app_state: Annotated[AppState, Depends(get_app_state)],
    data_service: Annotated[DataService, Depends(get_data_service)],
):
    """Get predictions for specific series and model."""
    try:
        prediction_service = PredictionService(app_state, series, data_service)
        return await prediction_service.get_prediction_for_model(model)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        log.exception("Error in get_prediction")
        raise HTTPException(status_code=500, detail=str(e)) from e
