from datetime import date

from pydantic import BaseModel

from app.models.system import SystemStatus


class PredictionResponse(BaseModel):
    driver: str
    nationality: str | None = None
    position: int
    points: float
    wins: int
    podiums: int
    win_rate: float
    dnf_rate: float
    experience: int
    age: float | None = None
    dob: date | None = None
    participation_rate: float
    teammate_h2h: float
    team: str
    team_pos: int
    team_points: float
    empirical_percentage: float | None = None


class ModelResults(BaseModel):
    model_name: str
    predictions: list[PredictionResponse]
    accuracy_metrics: dict[str, float]


class PredictionsResponse(BaseModel):
    models: list[str]
    predictions: dict[str, ModelResults]
    system_status: SystemStatus
