from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import HTTPException

from app.routes.predictions import get_predictions


class TestGetPredictions:
    @pytest.fixture
    def mock_app_state(self):
        """Mock app state."""
        return Mock()

    @pytest.fixture
    def mock_data_service(self):
        """Mock data service."""
        return Mock()

    @pytest.mark.asyncio
    async def test_prediction_service_success(self, mock_app_state, mock_data_service):
        """Test successful prediction flow."""
        expected_response = {"predictions": ["p1", "p2"]}

        with patch("app.routes.predictions.PredictionService") as mock_service_class:
            mock_service = Mock()
            mock_service.get_predictions = AsyncMock(return_value=expected_response)
            mock_service_class.return_value = mock_service

            result = await get_predictions(
                "f2_to_f1",
                mock_app_state,
                mock_data_service,
            )

            # Route returns what the service returns
            assert result == expected_response

            # Service was constructed correctly
            mock_service_class.assert_called_once_with(
                mock_app_state,
                "f2_to_f1",
                mock_data_service,
            )

            # Async method was awaited
            mock_service.get_predictions.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_prediction_service_exception(
        self,
        mock_app_state,
        mock_data_service,
    ):
        """Test exceptions are handled properly."""
        with patch("app.routes.predictions.PredictionService") as mock_service_class:
            mock_service = Mock()
            mock_service.get_predictions = AsyncMock(side_effect=Exception("boom"))
            mock_service_class.return_value = mock_service

            with pytest.raises(HTTPException) as exc_info:
                await get_predictions("f2_to_f1", mock_app_state, mock_data_service)

            assert exc_info.value.status_code == 500
            assert exc_info.value.detail == "boom"
