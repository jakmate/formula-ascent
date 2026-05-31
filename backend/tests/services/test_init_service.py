from unittest.mock import AsyncMock, Mock, patch

import pandas as pd
import pytest

from app.config import CURRENT_YEAR
from app.core.state import AppState
from app.services.data_service import DataService
from app.services.init_service import InitService


@pytest.fixture
def mock_app_state():
    return Mock(spec=AppState)


@pytest.fixture
def data_service(mock_app_state):
    return DataService(mock_app_state)


@pytest.fixture
def init_service(mock_app_state, data_service):
    return InitService(mock_app_state, data_service)


class TestInitializeSystem:
    @patch("app.services.init_service.ModelService")
    @patch("app.services.init_service.load_data")
    @patch("app.services.init_service.load_standings_data")
    @patch("app.services.init_service.load_qualifying_data")
    @patch("app.services.init_service.calculate_qualifying_features")
    @patch("app.services.init_service.create_target_variable")
    @patch("app.services.init_service.engineer_features")
    @patch("app.services.init_service.PredictionService")
    @pytest.mark.asyncio
    async def test_success(
        self,
        mock_prediction_service_class,
        mock_engineer,
        mock_target,
        mock_quali_features,
        mock_load_quali,
        mock_load_standings,
        mock_load_data,
        mock_model_service_class,
        init_service,
    ):
        # Setup mock data
        mock_model_service_instance = AsyncMock()
        mock_model_service_class.return_value = mock_model_service_instance
        mock_load_data.return_value = pd.DataFrame({"driver": ["A", "B"]})
        mock_load_standings.return_value = pd.DataFrame({"driver": ["A", "B"]})
        mock_load_quali.return_value = pd.DataFrame({"driver": ["A", "B"]})
        mock_quali_features.return_value = pd.DataFrame({"driver": ["A", "B"]})
        mock_target.return_value = pd.DataFrame(
            {"driver": ["A", "B"], "promoted": [1, 0], "year": [2022, 2023]},
        )

        features_df = pd.DataFrame(
            {"driver": ["A", "B"], "year": [2022, 2023], "feature1": [1, 2]},
        )
        mock_engineer.return_value = features_df

        mock_prediction_service = AsyncMock()
        mock_prediction_service_class.return_value = mock_prediction_service

        init_service.app_state.models = {
            "f3_to_f2": {"RandomForest": Mock(), "LightGBM": Mock()},
            "f2_to_f1": {"RandomForest": Mock(), "LightGBM": Mock()},
        }

        init_service.app_state.save_state = Mock()

        await init_service.initialize_system()

        # Verify model training was called for both series
        assert mock_model_service_class.call_count == 2
        assert mock_model_service_instance.train_models.call_count == 2
        assert mock_model_service_instance.save_models.call_count == 2

        # Verify predictions were updated
        assert mock_prediction_service.get_prediction_for_model.call_count == 4

        # Verify state was saved
        init_service.app_state.save_state.assert_called_once()

    @patch("app.services.init_service.ModelService")
    @patch("app.services.init_service.load_data")
    @patch("app.services.init_service.load_standings_data")
    @patch("app.services.init_service.load_qualifying_data")
    @patch("app.services.init_service.calculate_qualifying_features")
    @patch("app.services.init_service.create_target_variable")
    @patch("app.services.init_service.engineer_features")
    @patch("app.services.init_service.PredictionService")
    @pytest.mark.asyncio
    async def test_no_historical_data(
        self,
        mock_prediction_service_class,
        mock_engineer,
        mock_target,
        mock_quali_features,
        mock_load_quali,
        mock_load_standings,
        mock_load_data,
        mock_model_service_class,
        init_service,
    ):
        # Setup mock data with only current year data
        mock_model_service_instance = AsyncMock()
        mock_model_service_class.return_value = mock_model_service_instance
        mock_load_data.return_value = pd.DataFrame({"driver": ["A"]})
        mock_load_standings.return_value = pd.DataFrame({"driver": ["A"]})
        mock_load_quali.return_value = pd.DataFrame({"driver": ["A"]})
        mock_quali_features.return_value = pd.DataFrame({"driver": ["A"]})
        mock_target.return_value = pd.DataFrame(
            {
                "driver": ["A"],
                "promoted": [1],
                "year": [CURRENT_YEAR],
            },
        )

        features_df = pd.DataFrame(
            {"driver": ["A"], "year": [CURRENT_YEAR], "feature1": [1]},
        )
        mock_engineer.return_value = features_df

        init_service.app_state.models = {
            "f3_to_f2": {"RandomForest": Mock(), "LightGBM": Mock()},
            "f2_to_f1": {"RandomForest": Mock(), "LightGBM": Mock()},
        }

        init_service.app_state.save_state = Mock()

        await init_service.initialize_system()

        # Verify ModelService was not called since no trainable data
        mock_model_service_class.assert_not_called()
        mock_model_service_instance.train_models.assert_not_called()
        mock_model_service_instance.save_models.assert_not_called()

    @patch("app.services.init_service.load_data")
    @pytest.mark.asyncio
    async def test_exception_handling(self, mock_load_data, init_service):
        # Make load_data raise an exception
        mock_load_data.side_effect = Exception("Test error")

        init_service.app_state.save_state = Mock()

        await init_service.initialize_system()

        # Verify state was still saved despite errors
        init_service.app_state.save_state.assert_called_once()
