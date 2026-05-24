from fastapi import Request

from backend.app.core.state import AppState
from backend.app.services.cronjobs_service import CronjobService
from backend.app.services.data_service import DataService
from backend.app.services.model_service import ModelService


def get_app_state(request: Request) -> AppState:
    return request.app.state.app_state


def get_model_service(request: Request) -> ModelService:
    return request.app.state.model_service


def get_data_service(request: Request) -> DataService:
    return request.app.state.data_service


def get_cronjob_service(request: Request) -> CronjobService:
    return request.app.state.cronjob_service
