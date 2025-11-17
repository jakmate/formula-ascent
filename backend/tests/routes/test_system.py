import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from fastapi import BackgroundTasks

from app.routes.system import refresh_data, refresh_predictions, refresh_schedule
from app.models.system import RefreshResponse


class TestRefreshData:
    @pytest.fixture
    def mock_cronjob_service(self):
        """Mock cronjob service."""
        mock_service = Mock()
        mock_service.scrape_and_train_task = AsyncMock()
        return mock_service

    @pytest.fixture
    def background_tasks(self):
        """Mock background tasks."""
        return Mock(spec=BackgroundTasks)

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
            assert result.estimated_completion == fixed_time + timedelta(minutes=2)

    @pytest.mark.asyncio
    async def test_refresh_data_response_model(self, mock_cronjob_service, background_tasks):
        """Test response follows RefreshResponse model."""
        result = await refresh_data(
            background_tasks=background_tasks, cronjob_service=mock_cronjob_service
        )

        # Test response structure
        assert hasattr(result, "message")
        assert hasattr(result, "estimated_completion")
        assert isinstance(result.message, str)
        assert isinstance(result.estimated_completion, datetime)

    @pytest.mark.asyncio
    async def test_estimated_completion_timing(self, mock_cronjob_service, background_tasks):
        """Test estimated completion is 2 minutes from now."""
        before_call = datetime.now()

        result = await refresh_data(
            background_tasks=background_tasks, cronjob_service=mock_cronjob_service
        )

        after_call = datetime.now()

        # Check completion time is roughly 2 minutes from call time
        expected_min = before_call + timedelta(minutes=2)
        expected_max = after_call + timedelta(minutes=2)

        assert expected_min <= result.estimated_completion <= expected_max


class TestRefreshPredictions:
    @pytest.fixture
    def mock_cronjob_service(self):
        svc = Mock()
        svc.scrape_predictions = AsyncMock()
        return svc

    @pytest.fixture
    def background_tasks(self):
        return Mock(spec=BackgroundTasks)

    @pytest.mark.asyncio
    async def test_refresh_predictions_schedules_task(self, mock_cronjob_service, background_tasks):
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
            assert result.message == "Predictions refresh and training started in background"
            assert result.estimated_completion == fixed_time + timedelta(minutes=1)


class TestRefreshSchedule:
    @pytest.fixture
    def mock_cronjob_service(self):
        svc = Mock()
        svc.scrape_schedule = AsyncMock()
        return svc

    @pytest.fixture
    def background_tasks(self):
        return Mock(spec=BackgroundTasks)

    @pytest.mark.asyncio
    async def test_refresh_schedule_schedules_task(self, mock_cronjob_service, background_tasks):
        fixed_time = datetime(2025, 6, 1, 9, 30, 0)
        with patch("app.routes.system.datetime") as mock_datetime:
            mock_datetime.now.return_value = fixed_time

            result = await refresh_schedule(
                background_tasks=background_tasks, cronjob_service=mock_cronjob_service
            )

            background_tasks.add_task.assert_called_once_with(mock_cronjob_service.scrape_schedule)

            assert isinstance(result, RefreshResponse)
            assert result.message == "Schedule refresh started in background"
            assert result.estimated_completion == fixed_time + timedelta(minutes=1)
