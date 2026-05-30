import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import LOGGER
from app.core.state import AppState
from app.routes.router import api_router
from app.services.cronjobs_service import CronjobService
from app.services.data_service import DataService
from app.services.init_service import InitService
from app.services.model_service import ModelService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown handling."""
    LOGGER.info("Starting application...")

    app.state.app_state = AppState()
    app.state.app_state.load_state()

    app.state.model_service = ModelService(app.state.app_state)
    app.state.data_service = DataService(app.state.app_state, {})
    app.state.init_service = InitService(
        app.state.app_state,
        app.state.data_service,
    )
    app.state.cronjob_service = CronjobService(
        app.state.app_state, app.state.data_service, app.state.init_service
    )

    if not await app.state.model_service.load_models():
        LOGGER.info("No models found. Initializing system...")
        await app.state.init_service.initialize_system()

    await app.state.cronjob_service.start()

    yield

    LOGGER.info("Shutting down application...")
    await app.state.cronjob_service.stop()
    app.state.app_state.save_state()
    LOGGER.info("Application shutdown complete")


def create_app() -> FastAPI:
    """Create FastAPI application with all configurations."""
    app = FastAPI(
        title="Formula Predictions API",
        version="1.0.0",
        description="API for predicting Formula 2 and 3 career promotions",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://localhost:3000",
            os.getenv("CORS_ORIGINS", ""),
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(api_router, prefix="/api")

    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "127.0.0.1")
    uvicorn.run(app, host=host, port=port, log_level="debug", reload=False)
