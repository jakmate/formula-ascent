from fastapi import APIRouter

from app.routes import health, models, predictions, schedule, system

api_router = APIRouter()
api_router.include_router(health.router, prefix="", tags=["Health"])
api_router.include_router(models.router, prefix="/models", tags=["Models"])
api_router.include_router(
    predictions.router, prefix="/predictions", tags=["Predictions"]
)
api_router.include_router(schedule.router, prefix="/races", tags=["Schedule"])
api_router.include_router(system.router, prefix="/system", tags=["System"])
