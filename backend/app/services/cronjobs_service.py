import asyncio
import logging
from datetime import UTC, datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.config import CURRENT_YEAR, SEASON_END_MONTH
from app.core.state import AppState
from app.scrapers.schedule_scraper import scrape_schedules
from app.scrapers.scrape import scrape_current_year, scrape_wiki
from app.services.data_service import DataService
from app.services.init_service import InitService
from app.services.prediction_service import PredictionService

log = logging.getLogger(__name__)


class CronjobService:
    def __init__(
        self, app_state: AppState, data_service: DataService, init_service: InitService
    ) -> None:
        self.app_state = app_state
        self.data_service = data_service
        self.init_service = init_service
        self.scheduler = AsyncIOScheduler()

    async def start(self):
        """Start scheduler."""
        self.scheduler.add_job(
            self.scrape_and_train_task,
            "cron",
            day_of_week="mon",
            hour=3,
            id="weekly_scrape_train",
        )
        self.scheduler.start()
        await asyncio.sleep(0)
        log.info("Scheduler started")

    async def stop(self):
        """Stop scheduler."""
        await asyncio.sleep(0)
        self.scheduler.shutdown()

    async def scrape_and_train_task(self):
        """Combined scraping and training task for new seasons."""
        try:
            log.info("Starting data scraping task...")
            await asyncio.get_event_loop().run_in_executor(None, scrape_current_year)
            self.app_state.system_status["last_scrape_full"] = datetime.now(UTC)
            self.app_state.system_status["last_scrape_predictions"] = datetime.now(UTC)
            self.app_state.system_status["last_scrape_schedule"] = datetime.now(UTC)
            log.info("Data scraping completed")

            if (
                self._is_season_complete()
                and self.app_state.system_status["last_trained_season"] < CURRENT_YEAR
            ):
                log.info("New season %s complete. Starting training...", CURRENT_YEAR)
                try:
                    await self.init_service.initialize_system()
                except Exception:
                    log.exception("Training task failed")
            else:
                log.info(
                    "No new complete season available. Updating predictions only.",
                )

                for series in ["f3_to_f2", "f2_to_f1"]:
                    prediction_service = PredictionService(
                        self.app_state,
                        series,
                        self.data_service,
                    )
                    await prediction_service.update_predictions()
        except Exception:
            log.exception("Scrape and train task failed")
        finally:
            self.app_state.save_state()

    def _is_season_complete(self):
        """Check if current season is complete based on date."""
        now = datetime.now(UTC)
        return now.month > SEASON_END_MONTH and now.year == CURRENT_YEAR

    async def scrape_predictions(self):
        """Scrape prediction-related data."""
        try:
            log.info("Starting predictions scraping task...")
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: scrape_wiki(start_year=CURRENT_YEAR),
            )
            self.app_state.system_status["last_scrape_predictions"] = datetime.now(UTC)
            log.info("Predictions scraping completed")

            # Update predictions without training
            for series in ["f3_to_f2", "f2_to_f1"]:
                prediction_service = PredictionService(
                    self.app_state,
                    series,
                    self.data_service,
                )
                await prediction_service.update_predictions()
        except Exception:
            log.exception("Predictions scrape task failed")
        finally:
            self.app_state.save_state()

    async def scrape_schedule(self):
        """Scrape schedule data."""
        try:
            log.info("Starting schedule scraping task...")
            await asyncio.get_event_loop().run_in_executor(None, scrape_schedules)
            self.app_state.system_status["last_scrape_schedule"] = datetime.now(UTC)
            log.info("Schedule scraping completed")
        except Exception:
            log.exception("Schedule scrape task failed")
        finally:
            self.app_state.save_state()
