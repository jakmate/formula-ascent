import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app.models.schedule import ScheduleRequest
from app.services.schedule_service import ScheduleService


@pytest.fixture
def schedule_service():
    return ScheduleService()


@pytest.fixture
def sample_schedule():
    return [
        {
            "round": 1,
            "name": "Melbourne",
            "location": "Australia",
            "sessions": {
                "practice": {
                    "start": "2025-03-14T01:00:00",
                    "end": "2025-03-14T01:45:00",
                },
                "race": {
                    "start": "2025-03-15T22:55:00",
                    "end": "2025-03-15T23:40:00",
                },
            },
        },
        {
            "round": 2,
            "name": "Sakhir",
            "location": "Bahrain",
            "sessions": {
                "practice": {
                    "start": "2025-04-11T11:00:00",
                    "end": "2025-04-11T11:45:00",
                },
                "race": {
                    "start": "2025-04-13T07:40:00",
                    "end": "2025-04-13T08:25:00",
                },
            },
        },
    ]


@pytest.fixture
def mock_aiofiles_open():
    def _mock(data):
        mock_file = AsyncMock()
        mock_file.__aenter__.return_value.read = AsyncMock(
            return_value=json.dumps(data),
        )
        return patch("aiofiles.open", return_value=mock_file)

    return _mock


class TestGetSeriesSchedule:
    @pytest.mark.asyncio
    async def test_returns_schedule_in_utc(
        self,
        schedule_service,
        sample_schedule,
        mock_aiofiles_open,
    ):
        with (
            patch("os.path.exists", return_value=True),
            mock_aiofiles_open(sample_schedule),
        ):
            result = await schedule_service.get_series_schedule(
                ScheduleRequest(series="f1"),
            )

            assert result == sample_schedule
            assert result[0]["sessions"]["practice"]["start"] == "2025-03-14T01:00:00"

    @pytest.mark.asyncio
    async def test_file_not_found(self, schedule_service):
        with patch("os.path.exists", return_value=False):
            with pytest.raises(HTTPException) as exc_info:
                await schedule_service.get_series_schedule(ScheduleRequest(series="f1"))

            assert exc_info.value.status_code == 404
            assert exc_info.value.detail == "Schedule data not found"

    @pytest.mark.asyncio
    async def test_converts_to_target_timezone(
        self,
        schedule_service,
        sample_schedule,
        mock_aiofiles_open,
    ):
        with (
            patch("os.path.exists", return_value=True),
            mock_aiofiles_open(sample_schedule),
        ):
            result = await schedule_service.get_series_schedule(
                ScheduleRequest(series="f1", timezone="America/New_York"),
            )

            # Verify conversion happened
            assert result[0]["sessions"]["practice"]["start"] != "2025-03-14T01:00:00"

    @pytest.mark.asyncio
    async def test_skips_tbc_sessions_during_conversion(
        self,
        schedule_service,
        mock_aiofiles_open,
    ):
        schedule = [
            {
                "round": 1,
                "name": "Test",
                "sessions": {
                    "practice": {"start": "2025-08-01", "time": "TBC"},
                    "race": {
                        "start": "2025-08-02T15:00:00",
                        "end": "2025-08-02T16:00:00",
                    },
                },
            },
        ]

        with patch("os.path.exists", return_value=True), mock_aiofiles_open(schedule):
            result = await schedule_service.get_series_schedule(
                ScheduleRequest(series="f1", timezone="America/New_York"),
            )
            # TBC sessions should not be converted
            assert result[0]["sessions"]["practice"]["start"] == "2025-08-01"
            assert result[0]["sessions"]["practice"]["time"] == "TBC"
            # Non-TBC should be converted
            assert result[0]["sessions"]["race"]["start"] != "2025-08-02T15:00:00"

    @pytest.mark.asyncio
    async def test_handles_missing_time_fields(
        self,
        schedule_service,
        mock_aiofiles_open,
    ):
        schedule = [
            {
                "round": 1,
                "name": "Test",
                "sessions": {
                    "practice": {},  # No start or end
                },
            },
        ]

        with patch("os.path.exists", return_value=True), mock_aiofiles_open(schedule):
            result = await schedule_service.get_series_schedule(
                ScheduleRequest(series="f1", timezone="America/New_York"),
            )
            assert len(result) == 1


class TestGetNextRace:
    @pytest.mark.asyncio
    async def test_file_not_found_raises_404(self, schedule_service):
        with patch("os.path.exists", return_value=False):
            with pytest.raises(HTTPException) as exc:
                await schedule_service.get_next_race(ScheduleRequest(series="f1"))
            assert exc.value.status_code == 404
            assert exc.value.detail == "Schedule data not found"

    @pytest.mark.asyncio
    async def test_returns_next_race_with_future_sessions(
        self,
        schedule_service,
        sample_schedule,
        mock_aiofiles_open,
    ):
        sample_schedule[1]["sessions"]["practice"]["start"] = "2030-04-11T11:00:00"
        sample_schedule[1]["sessions"]["race"]["start"] = "2030-04-13T07:40:00"

        with (
            patch("os.path.exists", return_value=True),
            mock_aiofiles_open(sample_schedule),
        ):
            result = await schedule_service.get_next_race(
                ScheduleRequest(series="f1"),
            )

            assert result["round"] == 2
            assert result["totalRounds"] == 2
            assert result["seasonCompleted"] is False
            assert result["nextSession"]["name"] == "practice"
            assert result["nextSession"]["date"] == "2030-04-11T11:00:00"
            assert result["nextSession"]["isTBC"] is False

    @pytest.mark.asyncio
    async def test_selects_earliest_future_session(
        self,
        schedule_service,
        mock_aiofiles_open,
    ):
        schedule = [
            {
                "round": 1,
                "name": "Test",
                "sessions": {
                    "qualifying": {"start": "2030-03-15T14:00:00"},
                    "practice": {"start": "2030-03-15T10:00:00"},
                    "race": {"start": "2030-03-16T15:00:00"},
                },
            },
        ]

        with patch("os.path.exists", return_value=True), mock_aiofiles_open(schedule):
            result = await schedule_service.get_next_race(
                ScheduleRequest(series="f1"),
            )

            assert result["nextSession"]["name"] == "practice"
            assert result["nextSession"]["date"] == "2030-03-15T10:00:00"

    @pytest.mark.asyncio
    async def test_returns_last_race_when_season_complete(
        self,
        schedule_service,
        sample_schedule,
        mock_aiofiles_open,
    ):
        # All sessions in the past
        with (
            patch("os.path.exists", return_value=True),
            mock_aiofiles_open(sample_schedule),
        ):
            result = await schedule_service.get_next_race(
                ScheduleRequest(series="f1"),
            )

            assert result["round"] == 2
            assert result["totalRounds"] == 2
            assert result["seasonCompleted"] is True
            assert "nextSession" not in result

    @pytest.mark.asyncio
    async def test_empty_schedule_returns_none(
        self,
        schedule_service,
        mock_aiofiles_open,
    ):
        with patch("os.path.exists", return_value=True), mock_aiofiles_open([]):
            result = await schedule_service.get_next_race(
                ScheduleRequest(series="f1"),
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_handles_date_only_format(self, schedule_service, mock_aiofiles_open):
        schedule = [
            {
                "round": 1,
                "name": "Test",
                "sessions": {
                    "practice": {"start": "2030-08-01", "time": "TBC"},
                    "race": {"start": "2030-08-02T15:00:00"},
                },
            },
        ]

        with patch("os.path.exists", return_value=True), mock_aiofiles_open(schedule):
            result = await schedule_service.get_next_race(
                ScheduleRequest(series="f1"),
            )

            assert result["nextSession"]["name"] == "practice"
            assert result["nextSession"]["date"] == "2030-08-01"
            assert result["nextSession"]["isTBC"] is True

    @pytest.mark.asyncio
    async def test_skips_invalid_datetime_strings(
        self,
        schedule_service,
        mock_aiofiles_open,
    ):
        schedule = [
            {
                "round": 1,
                "name": "Test",
                "sessions": {
                    "practice": {"start": "invalid-datetime"},
                    "race": {"start": "2030-03-16T15:00:00"},
                },
            },
        ]

        with patch("os.path.exists", return_value=True), mock_aiofiles_open(schedule):
            result = await schedule_service.get_next_race(
                ScheduleRequest(series="f1"),
            )

            assert result["nextSession"]["name"] == "race"

    @pytest.mark.asyncio
    async def test_converts_to_target_timezone(
        self,
        schedule_service,
        sample_schedule,
        mock_aiofiles_open,
    ):
        sample_schedule[0]["sessions"]["practice"]["start"] = "2030-03-14T01:00:00"
        sample_schedule[0]["sessions"]["practice"]["end"] = "2030-03-14T01:45:00"

        with (
            patch("os.path.exists", return_value=True),
            mock_aiofiles_open(sample_schedule),
        ):
            result = await schedule_service.get_next_race(
                ScheduleRequest(series="f1", timezone="America/New_York"),
            )

            # Times should be converted
            assert result["sessions"]["practice"]["start"] != "2030-03-14T01:00:00"
            assert (
                "-05:00" in result["sessions"]["practice"]["start"]
                or "-04:00" in result["sessions"]["practice"]["start"]
            )

    @pytest.mark.asyncio
    async def test_does_not_convert_date_only_strings(
        self,
        schedule_service,
        mock_aiofiles_open,
    ):
        schedule = [
            {
                "round": 1,
                "name": "Test",
                "sessions": {
                    "practice": {"start": "2030-08-01", "time": "TBC"},
                },
            },
        ]

        with patch("os.path.exists", return_value=True), mock_aiofiles_open(schedule):
            result = await schedule_service.get_next_race(
                ScheduleRequest(series="f1", timezone="America/New_York"),
            )

            # Date-only strings should not be converted
            assert result["sessions"]["practice"]["start"] == "2030-08-01"


class TestGetScheduleDir:
    def test_returns_current_year_when_has_json_files(self, schedule_service):
        """Should return current year directory when it exists and has JSON files."""
        mock_schedule_dir = MagicMock(spec=Path)
        mock_schedule_dir.exists.return_value = True
        mock_schedule_dir.glob.return_value = [Path("f1.json"), Path("f2.json")]
        mock_schedule_dir.name = "2025"

        with patch("app.services.schedule_service.SCHEDULE_DIR", mock_schedule_dir):
            result = schedule_service._get_schedule_dir()

            assert result == mock_schedule_dir
            mock_schedule_dir.glob.assert_called_once_with("*.json")

    def test_fallback_to_previous_year_when_current_empty(self, schedule_service):
        """Should fallback to previous year when current year has no JSON files."""
        mock_current_dir = MagicMock(spec=Path)
        mock_current_dir.exists.return_value = True
        mock_current_dir.glob.return_value = []  # No JSON files
        mock_current_dir.name = "2025"

        mock_parent = MagicMock(spec=Path)
        mock_current_dir.parent = mock_parent

        mock_prev_dir = MagicMock(spec=Path)
        mock_prev_dir.exists.return_value = True
        mock_parent.__truediv__.return_value = mock_prev_dir

        with patch("app.services.schedule_service.SCHEDULE_DIR", mock_current_dir):
            result = schedule_service._get_schedule_dir()

            assert result == mock_prev_dir
            mock_parent.__truediv__.assert_called_once_with("2024")

    def test_returns_current_when_previous_year_not_exists(self, schedule_service):
        """Should return current year dir when previous year doesn't exist."""
        mock_current_dir = MagicMock(spec=Path)
        mock_current_dir.exists.return_value = True
        mock_current_dir.glob.return_value = []
        mock_current_dir.name = "2025"

        mock_parent = MagicMock(spec=Path)
        mock_current_dir.parent = mock_parent

        mock_prev_dir = MagicMock(spec=Path)
        mock_prev_dir.exists.return_value = False
        mock_parent.__truediv__.return_value = mock_prev_dir

        with patch("app.services.schedule_service.SCHEDULE_DIR", mock_current_dir):
            result = schedule_service._get_schedule_dir()

            assert result == mock_current_dir

    def test_handles_non_numeric_directory_name(self, schedule_service):
        """Should handle non-numeric directory names gracefully."""
        mock_current_dir = MagicMock(spec=Path)
        mock_current_dir.exists.return_value = True
        mock_current_dir.glob.return_value = []
        mock_current_dir.name = "schedules"  # Non-numeric

        mock_parent = MagicMock(spec=Path)
        mock_current_dir.parent = mock_parent

        with patch("app.services.schedule_service.SCHEDULE_DIR", mock_current_dir):
            result = schedule_service._get_schedule_dir()

            # Should return current dir when name is not numeric
            assert result == mock_current_dir
