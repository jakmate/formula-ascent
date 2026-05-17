import asyncio
from datetime import datetime
from pathlib import Path

import joblib
import torch

from app.config import LOGGER, MODELS_DIR
from app.core.predictor import RacingPredictor
from app.core.state import AppState


class ModelService:
    def __init__(self, app_state: AppState, series: str | None = None) -> None:
        self.app_state = app_state
        self.series = series

    async def save_models(self):
        """Save models to disk."""
        try:
            # Create series-specific directory
            series_dir = MODELS_DIR / self.series if self.series else MODELS_DIR
            await asyncio.to_thread(Path.mkdir, series_dir, exist_ok=True)

            models_to_save = (
                self.app_state.models[self.series]
                if self.series
                else self.app_state.models
            )

            for name, model in models_to_save.items():
                if name == "PyTorch":
                    await asyncio.to_thread(
                        torch.save,
                        model.state_dict(),
                        series_dir / f"{name}.pt",
                        _use_new_zipfile_serialization=True,
                    )
                    if hasattr(model, "calibrator") and model.calibrator is not None:
                        await asyncio.to_thread(
                            joblib.dump,
                            model.calibrator,
                            series_dir / f"{name}_calibrator.joblib",
                        )
                else:
                    await asyncio.to_thread(
                        joblib.dump,
                        model,
                        series_dir / f"{name}.joblib",
                    )

            preprocessor_data = {
                "scaler": self.app_state.scaler[self.series]
                if self.series
                else self.app_state.scaler,
                "feature_cols": self.app_state.feature_cols[self.series]
                if self.series
                else self.app_state.feature_cols,
            }
            await asyncio.to_thread(
                joblib.dump,
                preprocessor_data,
                series_dir / "preprocessor.joblib",
            )

            LOGGER.info(f"Models saved successfully for {self.series or 'all series'}")
        except Exception as e:
            LOGGER.error(f"Error saving models: {e}")

    async def load_models(self) -> bool:
        """Load models from disk."""
        try:
            models_loaded = False

            # Load for specific series or all series
            series_to_load = [self.series] if self.series else ["f3_to_f2", "f2_to_f1"]

            for series in series_to_load:
                series_dir = MODELS_DIR / series
                if not series_dir.exists():
                    continue

                # Load preprocessor
                await self._load_preprocessor(series, series_dir)

                # Load models
                for model_file in series_dir.iterdir():
                    loaded = await self._load_model_file(series, series_dir, model_file)
                    models_loaded = models_loaded or loaded

                # Update models_available for this series
                if self.app_state.models[series]:
                    self.app_state.system_status["models_available"][series] = list(
                        self.app_state.models[series].keys(),
                    )

            if models_loaded:
                LOGGER.info(
                    f"Loaded models for series: {list(self.app_state.models.keys())}",
                )

            return models_loaded
        except Exception as e:
            LOGGER.error(f"Error loading models: {e}")
            return False

    async def _load_preprocessor(self, series: str, series_dir: Path) -> None:
        preprocessor_path = series_dir / "preprocessor.joblib"
        if preprocessor_path.exists():
            preprocessor = await asyncio.to_thread(joblib.load, preprocessor_path)
            self.app_state.scaler[series] = preprocessor["scaler"]
            self.app_state.feature_cols[series] = preprocessor["feature_cols"]

    async def _load_model_file(
        self,
        series: str,
        series_dir: Path,
        model_file: str,
    ) -> bool:
        if model_file == "preprocessor.joblib" or "_calibrator" in model_file:
            return False

        name = Path(model_file).stem
        model_path = series_dir / model_file

        if model_file.endswith(".joblib"):
            model = await asyncio.to_thread(joblib.load, model_path)
            self.app_state.models[series][name] = model
            return True
        if model_file.endswith(".pt"):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = RacingPredictor(len(self.app_state.feature_cols[series]))
            state_dict = await asyncio.to_thread(
                torch.load,
                model_path,
                map_location=device,
                weights_only=True,
            )
            model.load_state_dict(state_dict)
            model = model.to(device)
            model.eval()
            calibrator_path = series_dir / f"{name}_calibrator.joblib"
            if calibrator_path.exists():
                model.calibrator = await asyncio.to_thread(joblib.load, calibrator_path)
            self.app_state.models[series][name] = model
            return True

        return False

    async def train_models(self, trainable_df):
        """Train models on provided data."""
        LOGGER.info(f"Training models for {self.series} on {len(trainable_df)} records")
        from app.core.predictor import train_models

        (models, feature_cols, scaler) = await asyncio.to_thread(
            train_models,
            trainable_df,
        )

        # Store in series-specific slots
        self.app_state.models[self.series] = models
        self.app_state.feature_cols[self.series] = feature_cols
        self.app_state.scaler[self.series] = scaler

        self.app_state.system_status["last_training"] = datetime.now()
        self.app_state.system_status["last_trained_season"] = trainable_df["year"].max()

        # Update available models for this series
        self.app_state.system_status["models_available"][self.series] = list(
            models.keys(),
        )

        self.app_state.system_status["data_health"][self.series] = {
            "historical_records": len(trainable_df),
            "current_records": 0,
        }
