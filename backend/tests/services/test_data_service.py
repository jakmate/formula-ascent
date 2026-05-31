from unittest.mock import Mock, patch

import pandas as pd
import pytest
from fastapi import HTTPException

from app.config import CURRENT_YEAR
from app.core.state import AppState
from app.services.data_service import DataService


@pytest.fixture
def mock_app_state():
    return Mock(spec=AppState)


@pytest.fixture
def data_service(mock_app_state):
    return DataService(mock_app_state)


@pytest.fixture(autouse=True)
def clear_cache(data_service):
    data_service.data_cache.clear()


class TestLoadCurrentData:
    @patch("app.services.data_service.load_data")
    @pytest.mark.asyncio
    async def test_empty_feeder_df(self, mock_load_data, data_service):
        mock_load_data.return_value = pd.DataFrame()

        with pytest.raises(HTTPException) as exc_info:
            await data_service.load_current_data("f2_to_f1")

        assert exc_info.value.status_code == 404
        assert "No F2 data available" in str(exc_info.value.detail)

    @patch("app.services.data_service.load_data")
    @patch("app.services.data_service.load_standings_data")
    @patch("app.services.data_service.load_qualifying_data")
    @patch("app.services.data_service.calculate_qualifying_features")
    @patch("app.services.data_service.create_target_variable")
    @patch("app.services.data_service.engineer_features")
    @pytest.mark.asyncio
    async def test_fallback_to_max_year(
        self,
        mock_engineer,
        mock_target,
        mock_quali_features,
        mock_load_quali,
        mock_load_standings,
        mock_load_data,
        data_service,
    ):
        # Setup mock data
        mock_load_data.return_value = pd.DataFrame({"driver": ["A"]})
        mock_load_standings.return_value = pd.DataFrame({"driver": ["A"]})
        mock_load_quali.return_value = pd.DataFrame({"driver": ["A"]})
        mock_quali_features.return_value = pd.DataFrame({"driver": ["A"]})
        mock_target.return_value = pd.DataFrame({"driver": ["A"], "promoted": [1]})

        # Features dataframe with no current year data but has historical data
        features_df = pd.DataFrame(
            {"driver": ["A", "B"], "year": [2022, 2023], "feature1": [1, 2]},
        )
        mock_engineer.return_value = features_df

        result = await data_service.load_current_data("f2_to_f1")

        # Should return data from max year (2023)
        assert len(result) == 1
        assert result["year"].iloc[0] == 2023

    @patch("app.services.data_service.load_data")
    @patch("app.services.data_service.load_standings_data")
    @patch("app.services.data_service.load_qualifying_data")
    @patch("app.services.data_service.calculate_qualifying_features")
    @patch("app.services.data_service.create_target_variable")
    @patch("app.services.data_service.engineer_features")
    @pytest.mark.asyncio
    async def test_no_drivers_available(
        self,
        mock_engineer,
        mock_target,
        mock_quali_features,
        mock_load_quali,
        mock_load_standings,
        mock_load_data,
        data_service,
    ):
        # Setup mock data
        mock_load_data.return_value = pd.DataFrame({"driver": ["A"]})
        mock_load_standings.return_value = pd.DataFrame({"driver": ["A"]})
        mock_load_quali.return_value = pd.DataFrame({"driver": ["A"]})
        mock_quali_features.return_value = pd.DataFrame({"driver": ["A"]})
        mock_target.return_value = pd.DataFrame({"driver": ["A"], "promoted": [1]})

        # Features dataframe with no current year data and empty after max year fallback
        features_df = pd.DataFrame(columns=["year", "feature1"])
        mock_engineer.return_value = features_df

        with pytest.raises(HTTPException) as exc_info:
            await data_service.load_current_data("f2_to_f1")

        assert exc_info.value.status_code == 404
        assert "No drivers data available" in str(exc_info.value.detail)

    @patch("app.services.data_service.time.time")
    @pytest.mark.asyncio
    async def test_cache_hit(self, mock_time, data_service):
        # Setup cached data
        cache_key = "current_data_f2_to_f1"
        cached_data = pd.DataFrame({"driver": ["A"], "year": [2023]})
        data_service.data_cache[cache_key] = cached_data

        # Mock time to verify performance logging
        mock_time.side_effect = [1000, 1000.5]  # start and end times

        result = await data_service.load_current_data("f2_to_f1")

        # Verify cached data was returned
        pd.testing.assert_frame_equal(result, cached_data)

    @patch("app.services.data_service.engineer_features")
    @patch("app.services.data_service.create_target_variable")
    @patch("app.services.data_service.calculate_qualifying_features")
    @patch("app.services.data_service.load_qualifying_data")
    @patch("app.services.data_service.load_standings_data")
    @patch("app.services.data_service.load_data")
    @patch("app.services.data_service.time.time")
    @pytest.mark.asyncio
    async def test_cache_miss(
        self,
        mock_time,
        mock_load_data,
        mock_load_standings,
        mock_load_quali,
        mock_quali_features,
        mock_target,
        mock_engineer,
        data_service,
    ):
        # Ensure cache is empty
        data_service.data_cache.clear()

        # Setup mock data
        mock_load_data.return_value = pd.DataFrame({"driver": ["A"]})
        mock_load_standings.return_value = pd.DataFrame({"driver": ["A"]})
        mock_load_quali.return_value = pd.DataFrame({"driver": ["A"]})
        mock_quali_features.return_value = pd.DataFrame({"driver": ["A"]})
        mock_target.return_value = pd.DataFrame({"driver": ["A"], "promoted": [1]})

        # Mock features with current year data
        features_df = pd.DataFrame(
            {"driver": ["A"], "year": [CURRENT_YEAR], "feature1": [1]},
        )
        mock_engineer.return_value = features_df

        # Use a counter to return incremental time values
        time_counter = 1000.0

        def time_side_effect():
            nonlocal time_counter
            current_time = time_counter
            time_counter += 0.1
            return current_time

        mock_time.side_effect = time_side_effect

        result = await data_service.load_current_data("f2_to_f1")

        # Verify data was cached
        cache_key = "current_data_f2_to_f1"
        assert cache_key in data_service.data_cache
        pd.testing.assert_frame_equal(data_service.data_cache[cache_key], result)


class TestParseSeries:
    def test_unknown_series(self, data_service):
        with pytest.raises(ValueError, match="unknown_series") as exc_info:
            data_service._parse_series("unknown_series")

        assert "Unknown series: unknown_series" in str(exc_info.value)


class TestClearCache:
    def test_specific_series(self, data_service):
        # Setup cache with multiple entries
        data_service.data_cache = {
            "current_data_f2_to_f1": pd.DataFrame({"a": [1]}),
            "current_data_f3_to_f2": pd.DataFrame({"b": [2]}),
            "full_data_f2_to_f1": pd.DataFrame({"c": [3]}),
            "other_data": pd.DataFrame({"d": [4]}),
        }

        data_service.clear_cache("f2_to_f1")

        # Verify only f2_to_f1 related entries were removed
        assert "current_data_f2_to_f1" not in data_service.data_cache
        assert "full_data_f2_to_f1" not in data_service.data_cache
        assert "current_data_f3_to_f2" in data_service.data_cache
        assert "other_data" in data_service.data_cache

    def test_clear_all(self, data_service):
        # Setup cache
        data_service.data_cache = {
            "current_data_f2_to_f1": pd.DataFrame({"a": [1]}),
            "current_data_f3_to_f2": pd.DataFrame({"b": [2]}),
        }

        data_service.clear_cache()

        # Verify cache is empty
        assert data_service.data_cache == {}

    def test_nonexistent_series(self, data_service):
        # Setup cache with different series
        data_service.data_cache = {"current_data_f3_to_f2": pd.DataFrame({"b": [2]})}

        data_service.clear_cache("f2_to_f1")

        # Verify cache unchanged
        assert "current_data_f3_to_f2" in data_service.data_cache
        assert len(data_service.data_cache) == 1
