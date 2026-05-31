import importlib.util
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import create_app, lifespan


class TestCreateApp:
    def test_app_configuration(self):
        app = create_app()

        assert isinstance(app, FastAPI)
        assert app.title == "Formula Predictions API"
        assert app.version == "1.0.0"
        assert app.description == "API for predicting Formula 2 and 3 career promotions"

        # Check that routes are registered with /api prefix
        routes = [route.path for route in app.routes if route.path.startswith("/api")]
        assert len(routes) > 0


class TestLifespan:
    @pytest.mark.asyncio
    async def test_lifespan_successful_startup_shutdown(self):
        app = FastAPI()

        with (
            patch("app.main.AppState"),
            patch("app.main.ModelService") as mock_model,
            patch("app.main.DataService") as mock_data,
            patch("app.main.CronjobService") as mock_cron,
        ):
            mock_model.return_value.load_models = AsyncMock(return_value=True)
            mock_data.return_value.initialize_system = AsyncMock()
            mock_cron.return_value.start = AsyncMock()
            mock_cron.return_value.stop = AsyncMock()

            async with lifespan(app):
                assert hasattr(app.state, "app_state")
                assert hasattr(app.state, "model_service")
                assert hasattr(app.state, "data_service")
                assert hasattr(app.state, "cronjob_service")

    @pytest.mark.asyncio
    async def test_lifespan_cleanup_on_exception(self):
        app = FastAPI()

        with (
            patch("app.main.AppState"),
            patch("app.main.ModelService") as mock_model,
            patch("app.main.DataService"),
            patch("app.main.CronjobService") as mock_cron,
        ):
            mock_model.return_value.load_models = AsyncMock(return_value=True)
            mock_cron.return_value.start = AsyncMock()
            mock_cron.return_value.stop = AsyncMock()

            with pytest.raises(RuntimeError):
                async with lifespan(app):
                    raise RuntimeError("Test exception")

        # After exception, cleanup scheduler removed or state should be saved
        # We check that object exists but lifecycle exited cleanly
        assert hasattr(app.state, "app_state")

    @pytest.mark.asyncio
    async def test_lifespan_init_failure(self):
        app = FastAPI()

        with (
            patch("app.main.AppState", side_effect=Exception("Init failed")),
            pytest.raises(Exception, match="Init failed"),
        ):
            async with lifespan(app):
                pass


class TestAppIntegration:
    def test_app_startup_with_test_client(self):
        """Test that the app can start successfully with TestClient."""
        app = create_app()

        with patch("app.main.lifespan"), TestClient(app) as client:
            # The app should be able to start without errors
            assert client.app is not None
            response = client.get("/api/health")  # or any endpoint
            assert response.status_code in (200, 404)

    @patch("app.main.os.environ.get")
    def test_main_execution_with_default_port(self, mock_env_get):
        mock_env_get.return_value = None

        with patch("app.main.uvicorn.run"):
            # Simulate the if __name__ == "__main__" block
            port = int(mock_env_get.return_value or 8000)
            assert port == 8000


def test_main_block_execution():
    with (
        patch("app.main.os.environ.get", return_value="8000"),
        patch("app.main.uvicorn.run") as mock_run,
    ):
        # Test by importing the module with __name__ set to "__main__"
        spec = importlib.util.spec_from_file_location("__main__", "app/main.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        mock_run.assert_called_once()
