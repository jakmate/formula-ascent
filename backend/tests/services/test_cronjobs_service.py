from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.core.state import AppState
from app.services.cronjobs_service import CronjobService


@pytest.fixture
def mock_app_state():
    app_state = Mock(spec=AppState)
    app_state.system_status = {"last_scrape_full": None, "last_trained_season": 2022}
    app_state.save_state = Mock()
    return app_state


@pytest.fixture
def mock_services():
    model_service = Mock()
    data_service = Mock()
    data_service.initialize_system = AsyncMock()
    return model_service, data_service


@pytest.fixture
def cronjobs_service(mock_app_state, mock_services):
    model_service, data_service = mock_services
    with patch("app.services.cronjobs_service.PredictionService"):
        return CronjobService(mock_app_state, model_service, data_service)


class TestInit:
    def test_init(self, mock_app_state, mock_services):
        model_service, data_service = mock_services
        cronjobs = CronjobService(mock_app_state, model_service, data_service)

        assert cronjobs.app_state == mock_app_state
        assert cronjobs.model_service == model_service
        assert cronjobs.data_service == data_service
        assert cronjobs.scheduler is not None


class TestStart:
    @pytest.mark.asyncio
    @patch("app.services.cronjobs_service.LOGGER")
    async def test_start(self, mock_logger, cronjobs_service):
        with (
            patch.object(cronjobs_service.scheduler, "add_job") as mock_add_job,
            patch.object(cronjobs_service.scheduler, "start") as mock_start,
        ):
            await cronjobs_service.start()

            mock_add_job.assert_called_once()
            mock_start.assert_called_once()
            mock_logger.info.assert_called_with("Scheduler started")


class TestStop:
    @pytest.mark.asyncio
    async def test_stop(self, cronjobs_service):
        with patch.object(cronjobs_service.scheduler, "shutdown") as mock_shutdown:
            await cronjobs_service.stop()
            mock_shutdown.assert_called_once()


class TestScrapeAndTrainTask:
    @pytest.mark.asyncio
    @patch("app.services.cronjobs_service.scrape_current_year")
    @patch("app.services.cronjobs_service.datetime")
    async def test_with_no_training(self, mock_datetime, mock_scrape, cronjobs_service):
        # Mock datetime.now()
        mock_now = datetime(2023, 6, 15)
        mock_datetime.now.return_value = mock_now

        # Mock season not complete
        with patch.object(cronjobs_service, "_is_season_complete", return_value=False):
            await cronjobs_service.scrape_and_train_task()

        # Verify scraping happened
        mock_scrape.assert_called_once()
        cronjobs_service.app_state.save_state.assert_called_once()
        assert cronjobs_service.app_state.system_status["last_scrape_full"] == mock_now

    @pytest.mark.asyncio
    @patch("app.services.cronjobs_service.scrape_current_year")
    @patch("app.services.cronjobs_service.CURRENT_YEAR", 2024)
    async def test_with_training(self, mock_scrape, cronjobs_service):
        # Mock season complete and new season available
        with (
            patch.object(cronjobs_service, "_is_season_complete", return_value=True),
        ):
            await cronjobs_service.scrape_and_train_task()

        mock_scrape.assert_called_once()
        cronjobs_service.app_state.save_state.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.services.cronjobs_service.scrape_current_year")
    @patch("app.services.cronjobs_service.CURRENT_YEAR", 2024)
    @patch("app.services.cronjobs_service.LOGGER")
    async def test_training_failure(self, mock_logger, mock_scrape, cronjobs_service):
        # Mock season complete and new season available
        with patch.object(cronjobs_service, "_is_season_complete", return_value=True):
            # Make initialize_system raise exception
            cronjobs_service.data_service.initialize_system = AsyncMock(
                side_effect=Exception("Training failed"),
            )

            await cronjobs_service.scrape_and_train_task()

        # Verify scraping succeeded
        mock_scrape.assert_called_once()

        # Verify training error was logged
        mock_logger.error.assert_called_with("Training task failed: Training failed")

        # Verify state was still saved
        cronjobs_service.app_state.save_state.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.services.cronjobs_service.scrape_current_year")
    @patch("app.services.cronjobs_service.PredictionService")
    async def test_update_predictions(
        self,
        mock_prediction_service_class,
        mock_scrape,
        cronjobs_service,
    ):
        # Mock prediction service
        mock_prediction_service = Mock()
        mock_prediction_service.update_predictions = AsyncMock()
        mock_prediction_service_class.return_value = mock_prediction_service

        with patch.object(cronjobs_service, "_is_season_complete", return_value=False):
            await cronjobs_service.scrape_and_train_task()

        # Verify prediction service was called for each series
        assert mock_prediction_service_class.call_count == 2
        assert mock_prediction_service.update_predictions.call_count == 2

    @pytest.mark.asyncio
    @patch("app.services.cronjobs_service.scrape_current_year")
    @patch("app.services.cronjobs_service.LOGGER")
    async def test_scrape_exception_handling(
        self,
        mock_logger,
        mock_scrape,
        cronjobs_service,
    ):
        # Make scraping raise exception
        mock_scrape.side_effect = Exception("Scraping failed")

        await cronjobs_service.scrape_and_train_task()

        # Verify specific error message was logged
        mock_logger.error.assert_called_with(
            "Scrape and train task failed: Scraping failed",
        )
        cronjobs_service.app_state.save_state.assert_called_once()


class TestIsSeasonCompleteTrue:
    @patch("app.services.cronjobs_service.CURRENT_YEAR", 2023)
    @patch("app.services.cronjobs_service.SEASON_END_MONTH", 11)
    def test_true(self, cronjobs_service):
        with patch("app.services.cronjobs_service.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2023, 12, 15)

            result = cronjobs_service._is_season_complete()
            assert result is True

    @patch("app.services.cronjobs_service.CURRENT_YEAR", 2023)
    @patch("app.services.cronjobs_service.SEASON_END_MONTH", 11)
    def test_false_early_month(self, cronjobs_service):
        with patch("app.services.cronjobs_service.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2023, 9, 15)

            result = cronjobs_service._is_season_complete()
            assert result is False

    @patch("app.services.cronjobs_service.CURRENT_YEAR", 2023)
    @patch("app.services.cronjobs_service.SEASON_END_MONTH", 11)
    def test_false_wrong_year(self, cronjobs_service):
        with patch("app.services.cronjobs_service.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 12, 15)

            result = cronjobs_service._is_season_complete()
            assert result is False


class TestScrapePredictions:
    @pytest.mark.asyncio
    @patch("app.services.cronjobs_service.scrape_wiki")
    @patch("app.services.cronjobs_service.CURRENT_YEAR", 2024)
    @patch("app.services.cronjobs_service.datetime")
    async def test_success(self, mock_datetime, mock_scrape_wiki, cronjobs_service):
        mock_now = datetime(2024, 6, 15)
        mock_datetime.now.return_value = mock_now

        with patch("app.services.cronjobs_service.PredictionService") as mock_ps_class:
            mock_ps = Mock()
            mock_ps.update_predictions = AsyncMock()
            mock_ps_class.return_value = mock_ps

            await cronjobs_service.scrape_predictions()

        mock_scrape_wiki.assert_called_once_with(start_year=2024)
        assert (
            cronjobs_service.app_state.system_status["last_scrape_predictions"]
            == mock_now
        )
        assert mock_ps_class.call_count == 2
        assert mock_ps.update_predictions.call_count == 2
        cronjobs_service.app_state.save_state.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.services.cronjobs_service.scrape_wiki")
    @patch("app.services.cronjobs_service.LOGGER")
    async def test_exception(self, mock_logger, mock_scrape_wiki, cronjobs_service):
        mock_scrape_wiki.side_effect = Exception("Scrape failed")

        await cronjobs_service.scrape_predictions()

        mock_logger.error.assert_called_with(
            "Predictions scrape task failed: Scrape failed",
        )
        cronjobs_service.app_state.save_state.assert_called_once()


class TestScrapeSchedule:
    @pytest.mark.asyncio
    @patch("app.services.cronjobs_service.scrape_schedules")
    @patch("app.services.cronjobs_service.datetime")
    async def test_success(
        self,
        mock_datetime,
        mock_scrape_schedules,
        cronjobs_service,
    ):
        mock_now = datetime(2024, 6, 15)
        mock_datetime.now.return_value = mock_now

        await cronjobs_service.scrape_schedule()

        mock_scrape_schedules.assert_called_once()
        assert (
            cronjobs_service.app_state.system_status["last_scrape_schedule"] == mock_now
        )
        cronjobs_service.app_state.save_state.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.services.cronjobs_service.scrape_schedules")
    @patch("app.services.cronjobs_service.LOGGER")
    async def test_exception(
        self,
        mock_logger,
        mock_scrape_schedules,
        cronjobs_service,
    ):
        mock_scrape_schedules.side_effect = Exception("Schedule scrape failed")

        await cronjobs_service.scrape_schedule()

        mock_logger.error.assert_called_with(
            "Schedule scrape task failed: Schedule scrape failed",
        )
        cronjobs_service.app_state.save_state.assert_called_once()
