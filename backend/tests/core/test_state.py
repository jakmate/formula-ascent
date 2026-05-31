import json
from datetime import UTC, datetime
from unittest.mock import MagicMock, mock_open, patch

import pytest

from app.core.state import AppState


@pytest.fixture
def mock_state_file(tmp_path):
    state_file = tmp_path / "test_state.json"
    with patch("app.core.state.STATE_FILE", state_file):
        yield state_file


@pytest.fixture
def sample_state_data():
    return {
        "last_scrape_full": "2024-01-01T12:00:00",
        "last_scrape_predictions": "2024-01-01T12:00:00",
        "last_scrape_schedule": "2024-01-01T12:00:00",
        "last_training": "2024-01-01T13:00:00",
        "last_trained_season": "2024",
        "models_available": {
            "f3_to_f2": ["f3_to_f2_model1"],
            "f2_to_f1": ["f2_to_f1_model2"],
        },
    }


class TestAppStateInit:
    def test_init_default_values(self):
        state = AppState()

        # Test series-specific structures
        assert state.models == {"f3_to_f2": {}, "f2_to_f1": {}}
        assert state.feature_cols == {"f3_to_f2": [], "f2_to_f1": []}
        assert state.scaler == {"f3_to_f2": None, "f2_to_f1": None}

        # Test other default values
        assert state.current_predictions == {}
        assert state.system_status["last_scrape_full"] is None
        assert state.system_status["last_scrape_predictions"] is None
        assert state.system_status["last_scrape_schedule"] is None
        assert state.system_status["last_training"] is None
        assert state.system_status["last_trained_season"] is None
        assert state.system_status["models_available"] == {
            "f3_to_f2": [],
            "f2_to_f1": [],
        }
        assert state.system_status["data_health"] == {}
        assert state.scheduler is not None


class TestSaveState:
    def test_save_state_with_datetime_values(self):
        state = AppState()
        test_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)

        state.system_status["last_scrape_full"] = test_time
        state.system_status["last_scrape_predictions"] = test_time
        state.system_status["last_scrape_schedule"] = test_time
        state.system_status["last_training"] = test_time
        state.system_status["last_trained_season"] = "2024"
        state.system_status["models_available"] = {
            "f3_to_f2": ["f3_to_f2_model1"],
            "f2_to_f1": [],
        }

        mock_file = mock_open()
        mock_path = MagicMock()
        mock_path.open = mock_file
        with patch("app.core.state.STATE_FILE", mock_path):
            state.save_state()

        handle = mock_file.return_value.__enter__.return_value
        written_data = "".join(call.args[0] for call in handle.write.call_args_list)
        saved_data = json.loads(written_data)

        assert saved_data["last_scrape_full"] == "2024-01-01T12:00:00+00:00"
        assert saved_data["last_trained_season"] == "2024"

    def test_save_state_with_none_values(self):
        state = AppState()
        mock_file = mock_open()
        mock_path = MagicMock()
        mock_path.open = mock_file
        with patch("app.core.state.STATE_FILE", mock_path):
            state.save_state()

        handle = mock_file.return_value.__enter__.return_value
        written_data = "".join(call.args[0] for call in handle.write.call_args_list)
        saved_data = json.loads(written_data)

        assert saved_data["last_scrape_full"] is None
        assert saved_data["models_available"] == {"f3_to_f2": [], "f2_to_f1": []}


class TestLoadState:
    def _make_mock_path(self, exists=True, read_data=None, open_side_effect=None):
        mock_path = MagicMock()
        mock_path.exists.return_value = exists
        if open_side_effect:
            mock_path.open.side_effect = open_side_effect
        else:
            mock_path.open = mock_open(read_data=read_data)
        return mock_path

    def test_load_state_success(self, sample_state_data):
        state = AppState()
        mock_path = self._make_mock_path(read_data=json.dumps(sample_state_data))
        with patch("app.core.state.STATE_FILE", mock_path):
            result = state.load_state()

        assert result is True
        assert state.system_status["last_scrape_full"] == datetime(
            2024, 1, 1, 12, 0, 0, tzinfo=UTC
        )
        assert state.system_status["last_training"] == datetime(
            2024, 1, 1, 13, 0, 0, tzinfo=UTC
        )
        assert state.system_status["models_available"] == {
            "f3_to_f2": ["f3_to_f2_model1"],
            "f2_to_f1": ["f2_to_f1_model2"],
        }

    def test_load_state_file_not_exists(self):
        state = AppState()
        mock_path = self._make_mock_path(exists=False)
        with patch("app.core.state.STATE_FILE", mock_path):
            result = state.load_state()

        assert result is False
        assert state.system_status["last_scrape_full"] is None

    def test_load_state_none_datetime_values(self):
        state = AppState()
        state_data = {
            "last_scrape_full": None,
            "last_training": None,
            "last_trained_season": "2024",
            "models_available": ["f3_to_f2_model1"],
        }
        mock_path = self._make_mock_path(read_data=json.dumps(state_data))
        with patch("app.core.state.STATE_FILE", mock_path):
            result = state.load_state()

        assert result is True
        assert state.system_status["last_scrape_full"] is None

    def test_load_state_json_decode_error(self):
        state = AppState()
        mock_path = self._make_mock_path(read_data="invalid json")
        with patch("app.core.state.STATE_FILE", mock_path):
            result = state.load_state()

        assert result is False
        mock_path.rename.assert_called_once()

    def test_load_state_general_exception(self):
        state = AppState()
        mock_path = self._make_mock_path(open_side_effect=OSError("File error"))
        with patch("app.core.state.STATE_FILE", mock_path):
            result = state.load_state()

        assert result is False

    def test_load_state_datetime_parsing(self):
        state = AppState()
        state_data = {
            "last_scrape_full": "2024-06-15T14:30:45+00:00",
            "last_training": "2024-06-15T15:45:30+00:00",
            "last_trained_season": "2024",
            "models_available": [],
        }
        mock_path = self._make_mock_path(read_data=json.dumps(state_data))
        with patch("app.core.state.STATE_FILE", mock_path):
            result = state.load_state()

        assert result is True
        assert state.system_status["last_scrape_full"] == datetime(
            2024, 6, 15, 14, 30, 45, tzinfo=UTC
        )

    def test_load_state_adds_missing_models_available_keys(self):
        state = AppState()
        state_data = {
            "last_scrape_full": "2024-01-01T12:00:00",
            "last_training": "2024-01-01T13:00:00",
            "last_trained_season": "2024",
            "models_available": {"f3_to_f2": ["f3_model_only"]},
        }
        mock_path = self._make_mock_path(read_data=json.dumps(state_data))
        with patch("app.core.state.STATE_FILE", mock_path):
            result = state.load_state()

        assert result is True
        assert state.system_status["models_available"] == {
            "f3_to_f2": ["f3_model_only"],
            "f2_to_f1": [],
        }


class TestStateIntegration:
    def _roundtrip(self, state1):
        """Save state1, return the JSON string."""
        mock_file = mock_open()
        mock_path = MagicMock()
        mock_path.open = mock_file
        with patch("app.core.state.STATE_FILE", mock_path):
            state1.save_state()
        handle = mock_file.return_value.__enter__.return_value
        return "".join(call.args[0] for call in handle.write.call_args_list)

    def test_save_load_roundtrip(self):
        state1 = AppState()
        test_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        state1.system_status["last_scrape_full"] = test_time
        state1.system_status["last_training"] = test_time
        state1.system_status["last_trained_season"] = "2024"
        state1.system_status["models_available"] = {
            "f3_to_f2": ["f3_to_f2_model1"],
            "f2_to_f1": ["f2_to_f1_model2"],
        }

        saved_data = self._roundtrip(state1)

        state2 = AppState()
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.open = mock_open(read_data=saved_data)
        with patch("app.core.state.STATE_FILE", mock_path):
            result = state2.load_state()

        assert result is True
        assert state2.system_status["last_scrape_full"] == test_time
        assert state2.system_status["models_available"] == {
            "f3_to_f2": ["f3_to_f2_model1"],
            "f2_to_f1": ["f2_to_f1_model2"],
        }

    def test_save_load_with_series_data(self):
        state1 = AppState()
        state1.models["f3_to_f2"] = {"RandomForest": "model1"}
        state1.system_status["models_available"] = {
            "f3_to_f2": ["RandomForest"],
            "f2_to_f1": ["LightGBM"],
        }

        saved_data = self._roundtrip(state1)

        state2 = AppState()
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.open = mock_open(read_data=saved_data)
        with patch("app.core.state.STATE_FILE", mock_path):
            result = state2.load_state()

        assert result is True
        assert state2.system_status["models_available"] == {
            "f3_to_f2": ["RandomForest"],
            "f2_to_f1": ["LightGBM"],
        }
        assert state2.models == {"f3_to_f2": {}, "f2_to_f1": {}}
