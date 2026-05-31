import asyncio
import logging
import time

from fastapi import HTTPException

from app.config import CURRENT_YEAR
from app.core.feature_creator import calculate_qualifying_features, engineer_features
from app.core.loader import load_data, load_qualifying_data, load_standings_data
from app.core.predictor import create_target_variable
from app.core.state import AppState

log = logging.getLogger(__name__)


class DataService:
    def __init__(self, app_state: AppState, data_cache: dict | None = None) -> None:
        self.app_state = app_state
        self.data_cache = data_cache if data_cache is not None else {}

    async def load_current_data(self, series: str):
        """Load and process current racing data with caching."""
        start_time = time.time()
        cache_key = f"current_data_{series}"

        # Return cached data if available
        if cache_key in self.data_cache:
            log.info(
                "Cache HIT for %s - returned in %.2fs", series, time.time() - start_time
            )
            await asyncio.sleep(0)
            return self.data_cache[cache_key]

        log.info("Cache MISS for %s - processing data...", series)
        load_start = time.time()

        # Parse series to get feeder and parent series
        feeder_series, parent_series = self._parse_series(series)

        feeder_df = load_data(feeder_series)
        parent_df = load_standings_data(parent_series, "drivers")
        log.info("Data loading took %.2fs", time.time() - load_start)

        if feeder_df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No {feeder_series} data available",
            )

        processing_start = time.time()
        feeder_quali_df = load_qualifying_data(feeder_series)
        feeder_df = calculate_qualifying_features(feeder_df, feeder_quali_df)
        feeder_df = create_target_variable(feeder_df, parent_df, parent_series)
        features_df = engineer_features(feeder_df)
        features_df["promoted"] = feeder_df["promoted"]
        log.info("Feature processing took %.2fs", time.time() - processing_start)

        current_df = features_df[features_df["year"] == CURRENT_YEAR].copy()
        if current_df.empty:
            current_year = features_df["year"].max()
            current_df = features_df[features_df["year"] == current_year].copy()

        if current_df.empty:
            raise HTTPException(status_code=404, detail="No drivers data available")

        # Cache the processed data
        self.data_cache[cache_key] = current_df
        log.info(
            "Total processing for %s: %.2fs - cached %s records",
            series,
            time.time() - start_time,
            len(current_df),
        )

        return current_df

    def _parse_series(self, series: str):
        """Parse series string to get feeder and parent series."""
        if series == "f3_to_f2":
            return "F3", "F2"
        if series == "f2_to_f1":
            return "F2", "F1"
        raise ValueError(f"Unknown series: {series}")

    def clear_cache(self, series: str | None = None):
        """Clear cached data for specific series or all."""
        if series:
            keys_to_remove = [k for k in self.data_cache if series in k]
            for key in keys_to_remove:
                del self.data_cache[key]
            log.info("Cleared cache for %s", series)
        else:
            self.data_cache.clear()
            log.info("Cleared all cached data")
