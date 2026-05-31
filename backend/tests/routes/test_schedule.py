from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from app.routes.schedule import get_next_race, get_series_schedule
from app.services.schedule_service import ScheduleRequest


@pytest.fixture
def mock_schedule_service():
    """Mock schedule service."""
    return AsyncMock()


class TestGetSeriesSchedule:
    @pytest.mark.asyncio
    async def test_get_series_schedule_success(self, mock_schedule_service):
        """Test successful schedule retrieval."""
        expected_data = {"schedule": "test_data"}
        mock_schedule_service.get_series_schedule.return_value = expected_data

        with patch(
            "app.routes.schedule.ScheduleService",
            return_value=mock_schedule_service,
        ):
            result = await get_series_schedule("f1", "UTC", "America/New_York")

            assert result == expected_data
            # Verify the ScheduleRequest was created and passed correctly
            call_args = mock_schedule_service.get_series_schedule.call_args[0][0]
            assert isinstance(call_args, ScheduleRequest)
            assert call_args.series == "f1"
            assert call_args.timezone == "UTC"
            assert call_args.x_timezone == "America/New_York"

    @pytest.mark.asyncio
    async def test_get_series_schedule_exception(self, mock_schedule_service):
        mock_schedule_service.get_series_schedule.side_effect = Exception(
            "Service error",
        )

        with patch(
            "app.routes.schedule.ScheduleService",
            return_value=mock_schedule_service,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_series_schedule("f1", None, None)

            assert exc_info.value.status_code == 500
            assert exc_info.value.detail == "Service error"


class TestGetNextRace:
    @pytest.mark.asyncio
    async def test_get_next_race_success(self, mock_schedule_service):
        expected_data = {"next_race": "test_data"}
        mock_schedule_service.get_next_race.return_value = expected_data

        with patch(
            "app.routes.schedule.ScheduleService",
            return_value=mock_schedule_service,
        ):
            result = await get_next_race("f1", "UTC", "America/New_York")

            assert result == expected_data
            call_args = mock_schedule_service.get_next_race.call_args[0][0]
            assert isinstance(call_args, ScheduleRequest)
            assert call_args.series == "f1"
            assert call_args.timezone == "UTC"
            assert call_args.x_timezone == "America/New_York"

    @pytest.mark.asyncio
    async def test_get_next_race_exception(self, mock_schedule_service):
        mock_schedule_service.get_next_race.side_effect = Exception("Service error")

        with patch(
            "app.routes.schedule.ScheduleService",
            return_value=mock_schedule_service,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_next_race("f1", None, None)

            assert exc_info.value.status_code == 500
            assert exc_info.value.detail == "Service error"
