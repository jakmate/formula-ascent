from datetime import datetime

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    models_loaded: dict[str, int]
    last_training: datetime | None


class SystemStatus(BaseModel):
    last_scrape_full: datetime | None = None
    last_scrape_predictions: datetime | None = None
    last_scrape_schedule: datetime | None = None
    last_training: datetime | None = None
    last_trained_season: int | None = None
    models_available: dict[str, list[str]]
    data_health: dict[str, dict[str, int]] = {}


class RefreshResponse(BaseModel):
    message: str
    task_id: str | None = None
    estimated_completion: datetime | None = None
