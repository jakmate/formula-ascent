from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from app.core.predictor import RacingPredictor
from app.core.state import AppState
from app.services.model_service import ModelService


@pytest.fixture
def mock_app_state():
    app_state = Mock(spec=AppState)
    app_state.models = {"f3_to_f2": {}, "f2_to_f1": {}}
    app_state.scaler = {"f3_to_f2": Mock(), "f2_to_f1": Mock()}
    app_state.feature_cols = {
        "f3_to_f2": ["col1", "col2"],
        "f2_to_f1": ["col1", "col2"],
    }
    app_state.system_status = {
        "models_available": {"f3_to_f2": [], "f2_to_f1": []},
        "last_training": None,
        "last_trained_season": None,
        "data_health": {},
    }
    return app_state


@pytest.fixture
def model_service(mock_app_state):
    return ModelService(mock_app_state, series="f3_to_f2")


class TestModelServiceInit:
    def test_init(self, mock_app_state):
        service = ModelService(mock_app_state, series="f3_to_f2")
        assert service.app_state == mock_app_state
        assert service.series == "f3_to_f2"


class TestSaveModels:
    test_models_dir = Path("/") / "test" / "models"

    @patch("app.services.model_service.MODELS_DIR", test_models_dir)
    @patch("pathlib.Path.mkdir")
    @patch("joblib.dump")
    @patch("app.services.model_service.LOGGER")
    @pytest.mark.asyncio
    async def test_with_series(
        self,
        mock_logger,
        mock_joblib_dump,
        mock_makedirs,
        model_service,
    ):
        # Setup mock models
        sklearn_model = Mock()
        model_service.app_state.models["f3_to_f2"] = {"RandomForest": sklearn_model}

        await model_service.save_models()

        # Verify directory creation
        mock_makedirs.assert_called_once()
        call_args = mock_makedirs.call_args[0][0]  # Get the first positional argument
        assert str(call_args).endswith(
            "f3_to_f2"
        )  # Verify it ends with the series name

        # Verify sklearn model save and preprocessor save
        assert mock_joblib_dump.call_count == 2  # ml model + preprocessor

        mock_logger.info.assert_called_with("Models saved successfully for f3_to_f2")

    @patch("app.services.model_service.MODELS_DIR", test_models_dir)
    @patch("pathlib.Path.mkdir")
    @patch("torch.save")
    @patch("joblib.dump")
    @patch("app.services.model_service.LOGGER")
    @pytest.mark.asyncio
    async def test_with_pytorch(
        self,
        mock_logger,
        mock_joblib_dump,
        mock_torch_save,
        mock_makedirs,
        model_service,
    ):
        # Setup mock models
        pytorch_model = Mock()
        pytorch_model.state_dict = Mock(return_value={"state": "dict"})
        model_service.app_state.models["f3_to_f2"] = {"PyTorch": pytorch_model}

        await model_service.save_models()

        # Verify directory creation
        mock_makedirs.assert_called_once()
        call_args = mock_makedirs.call_args[0][0]  # Get the first positional argument
        assert str(call_args).endswith(
            "f3_to_f2"
        )  # Verify it ends with the series name

        # Verify PyTorch model save
        expected_path = Path("/") / "test" / "models" / "f3_to_f2" / "PyTorch.pt"
        mock_torch_save.assert_called_once_with(
            {"state": "dict"},
            expected_path,
            _use_new_zipfile_serialization=True,
        )

        # Verify sklearn model save and preprocessor save
        assert mock_joblib_dump.call_count == 2  # preprocessor + calibrator

        mock_logger.info.assert_called_with("Models saved successfully for f3_to_f2")

    @patch("pathlib.Path.mkdir")
    @patch("joblib.dump")
    @patch("app.services.model_service.LOGGER")
    @pytest.mark.asyncio
    async def test_without_series(
        self, mock_logger, mock_joblib_dump, mock_makedirs, mock_app_state
    ):
        service = ModelService(mock_app_state, series=None)
        mock_app_state.models = {"RandomForest": Mock()}
        mock_app_state.scaler = Mock()
        mock_app_state.feature_cols = ["col1", "col2"]

        with patch("app.services.model_service.MODELS_DIR", self.test_models_dir):
            await service.save_models()

        mock_makedirs.assert_called_once()
        assert "models" in str(mock_makedirs.call_args[0][0])
        mock_logger.info.assert_called_with("Models saved successfully for all series")

    @patch("pathlib.Path.mkdir")
    @patch("app.services.model_service.LOGGER")
    @pytest.mark.asyncio
    async def test_exception(self, mock_logger, mock_makedirs, model_service):
        mock_makedirs.side_effect = Exception("Directory error")

        await model_service.save_models()

        mock_logger.error.assert_called_with("Error saving models: Directory error")


class TestLoadModels:
    def _make_series_dir(self, files=None, dir_exists=True, iterdir_exc=None):
        mock_models_dir = MagicMock()
        mock_series_dir = MagicMock()
        mock_series_dir.exists.return_value = dir_exists
        if iterdir_exc:
            mock_series_dir.iterdir.side_effect = iterdir_exc
        else:
            mock_series_dir.iterdir.return_value = [
                self._make_file(f) for f in (files or [])
            ]
        mock_preprocessor = MagicMock()
        mock_preprocessor.exists.return_value = True
        mock_calibrator = MagicMock()
        mock_calibrator.exists.return_value = True

        def series_truediv(key):
            if str(key) == "preprocessor.joblib":
                return mock_preprocessor
            if "_calibrator" in str(key):
                return mock_calibrator
            return MagicMock()

        mock_series_dir.__truediv__ = MagicMock(side_effect=series_truediv)
        mock_models_dir.__truediv__ = MagicMock(return_value=mock_series_dir)
        return mock_models_dir, mock_series_dir

    def _make_file(self, name):
        f = MagicMock()
        f.__str__ = lambda _: name
        f.__eq__ = lambda _, o: str(o) == name
        f.endswith = lambda suffix: name.endswith(suffix)
        f.__contains__ = lambda _, x: x in name
        f.stem = Path(name).stem
        f.name = name
        return f

    @pytest.mark.asyncio
    async def test_success(self, model_service):
        mock_dir, _ = self._make_series_dir(
            ["RandomForest.joblib", "preprocessor.joblib"]
        )
        with (
            patch("app.services.model_service.MODELS_DIR", mock_dir),
            patch(
                "joblib.load",
                side_effect=[
                    {"scaler": Mock(), "feature_cols": ["col1", "col2"]},
                    Mock(),
                ],
            ),
            patch("app.services.model_service.LOGGER") as mock_logger,
        ):
            result = await model_service.load_models()
        assert result is True
        mock_logger.info.assert_called()

    @pytest.mark.asyncio
    async def test_no_directory(self, model_service):
        mock_dir, _ = self._make_series_dir(dir_exists=False)
        with patch("app.services.model_service.MODELS_DIR", mock_dir):
            result = await model_service.load_models()
        assert result is False

    @pytest.mark.asyncio
    async def test_without_series(self, mock_app_state):
        service = ModelService(mock_app_state, series=None)
        mock_dir, _ = self._make_series_dir(
            ["RandomForest.joblib", "preprocessor.joblib"]
        )
        # without_series loads f3_to_f2 and f2_to_f1 — both hit same mock_dir
        with (
            patch("app.services.model_service.MODELS_DIR", mock_dir),
            patch(
                "joblib.load",
                side_effect=[
                    {"scaler": Mock(), "feature_cols": ["col1", "col2"]},
                    Mock(),
                    {"scaler": Mock(), "feature_cols": ["col1", "col2"]},
                    Mock(),
                ],
            ),
        ):
            result = await service.load_models()
        assert result is True

    @pytest.mark.asyncio
    async def test_exception(self, model_service):
        mock_dir, _ = self._make_series_dir(
            iterdir_exc=Exception("Directory read error")
        )
        with (
            patch("app.services.model_service.MODELS_DIR", mock_dir),
            patch("joblib.load", return_value={"scaler": Mock(), "feature_cols": []}),
            patch("app.services.model_service.LOGGER") as mock_logger,
        ):
            result = await model_service.load_models()
        assert result is False
        mock_logger.error.assert_called_with(
            "Error loading models: Directory read error"
        )

    @pytest.mark.asyncio
    async def test_pytorch_loading(self, model_service):
        mock_dir, _ = self._make_series_dir(
            ["PyTorch.pt", "PyTorch_calibrator.joblib", "preprocessor.joblib"]
        )
        mock_model = Mock(spec=RacingPredictor)
        mock_model.to.return_value = mock_model
        with (
            patch("app.services.model_service.MODELS_DIR", mock_dir),
            patch(
                "joblib.load",
                side_effect=[
                    {"scaler": Mock(), "feature_cols": ["col1", "col2"]},
                    Mock(),  # calibrator
                ],
            ),
            patch("torch.load", return_value={"param": "value"}),
            patch("torch.cuda.is_available", return_value=False),
            patch(
                "app.services.model_service.RacingPredictor", return_value=mock_model
            ),
        ):
            result = await model_service.load_models()
        assert result is True
        mock_model.load_state_dict.assert_called_once_with({"param": "value"})
        mock_model.eval.assert_called_once()

    @pytest.mark.asyncio
    async def test_unknown_file_extension(self, model_service):
        mock_dir, _ = self._make_series_dir(
            ["preprocessor.joblib", "model.txt", "RandomForest.pkl"]
        )
        with (
            patch("app.services.model_service.MODELS_DIR", mock_dir),
            patch(
                "joblib.load", return_value={"scaler": Mock(), "feature_cols": ["col1"]}
            ),
        ):
            result = await model_service.load_models()
        assert result is False
