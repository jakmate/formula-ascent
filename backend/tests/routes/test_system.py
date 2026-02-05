from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import BackgroundTasks

from app.models.system import RefreshResponse
from app.routes.system import refresh_data, refresh_predictions, refresh_schedule


@pytest.fixture
def background_tasks():
    """Mock background tasks."""
    return Mock(spec=BackgroundTasks)


class TestRefreshData:
    @pytest.fixture
    def mock_cronjob_service(self):
        """Mock cronjob service."""
        mock_service = Mock()
        mock_service.scrape_and_train_task = AsyncMock()
        return mock_service

    @pytest.mark.asyncio
    async def test_refresh_data_success(self, mock_cronjob_service, background_tasks):
        """Test successful data refresh trigger."""
        with patch("app.routes.system.datetime") as mock_datetime:
            fixed_time = datetime(2024, 1, 1, 12, 0, 0)
            mock_datetime.now.return_value = fixed_time

            result = await refresh_data(
                background_tasks=background_tasks, cronjob_service=mock_cronjob_service
            )

            # Verify background task was added
            background_tasks.add_task.assert_called_once_with(
                mock_cronjob_service.scrape_and_train_task
            )

            # Verify response
            assert isinstance(result, RefreshResponse)
            assert result.message == "Data refresh and training started in background"


class TestRefreshPredictions:
    @pytest.fixture
    def mock_cronjob_service(self):
        svc = Mock()
        svc.scrape_predictions = AsyncMock()
        return svc

    @pytest.mark.asyncio
    async def test_refresh_predictions_schedules_task(
        self, mock_cronjob_service, background_tasks
    ):
        fixed_time = datetime(2025, 1, 1, 12, 0, 0)
        with patch("app.routes.system.datetime") as mock_datetime:
            mock_datetime.now.return_value = fixed_time

            result = await refresh_predictions(
                background_tasks=background_tasks, cronjob_service=mock_cronjob_service
            )

            # background task scheduled with the expected service method
            background_tasks.add_task.assert_called_once_with(
                mock_cronjob_service.scrape_predictions
            )

            # correct response shape and contents
            assert isinstance(result, RefreshResponse)
            assert (
                result.message
                == "Predictions refresh and training started in background"
            )


class TestRefreshSchedule:
    @pytest.fixture
    def mock_cronjob_service(self):
        svc = Mock()
        svc.scrape_schedule = AsyncMock()
        return svc

    @pytest.mark.asyncio
    async def test_refresh_schedule_schedules_task(
        self, mock_cronjob_service, background_tasks
    ):
        fixed_time = datetime(2025, 6, 1, 9, 30, 0)
        with patch("app.routes.system.datetime") as mock_datetime:
            mock_datetime.now.return_value = fixed_time

            result = await refresh_schedule(
                background_tasks=background_tasks, cronjob_service=mock_cronjob_service
            )

            background_tasks.add_task.assert_called_once_with(
                mock_cronjob_service.scrape_schedule
            )

            assert isinstance(result, RefreshResponse)
            assert result.message == "Schedule refresh started in background"
