import os
import joblib
import torch
from datetime import datetime

from app.core.state import AppState
from app.config import MODELS_DIR, LOGGER
from app.core.predictor import RacingPredictor


class ModelService:
    def __init__(self, app_state: AppState, series: str = None):
        self.app_state = app_state
        self.series = series

    async def save_models(self):
        """Save models to disk"""
        try:
            # Create series-specific directory
            series_dir = (
                os.path.join(MODELS_DIR, self.series) if self.series else MODELS_DIR
            )
            os.makedirs(series_dir, exist_ok=True)

            models_to_save = (
                self.app_state.models[self.series]
                if self.series
                else self.app_state.models
            )

            for name, model in models_to_save.items():
                if name == "PyTorch":
                    torch.save(
                        model.state_dict(),
                        os.path.join(series_dir, f"{name}.pt"),
                        _use_new_zipfile_serialization=True,
                    )
                    if hasattr(model, "calibrator") and model.calibrator is not None:
                        joblib.dump(
                            model.calibrator,
                            os.path.join(series_dir, f"{name}_calibrator.joblib"),
                        )
                else:
                    joblib.dump(model, os.path.join(series_dir, f"{name}.joblib"))

            # Save preprocessor
            preprocessor_data = {
                "scaler": (
                    self.app_state.scaler[self.series]
                    if self.series
                    else self.app_state.scaler
                ),
                "feature_cols": (
                    self.app_state.feature_cols[self.series]
                    if self.series
                    else self.app_state.feature_cols
                ),
            }
            joblib.dump(
                preprocessor_data, os.path.join(series_dir, "preprocessor.joblib")
            )

            LOGGER.info(f"Models saved successfully for {self.series or 'all series'}")

        except Exception as e:
            LOGGER.error(f"Error saving models: {e}")

    async def load_models(self):
        """Load models from disk"""
        try:
            models_loaded = False

            # Load for specific series or all series
            series_to_load = [self.series] if self.series else ["f3_to_f2", "f2_to_f1"]

            for series in series_to_load:
                series_dir = os.path.join(MODELS_DIR, series)
                if not os.path.exists(series_dir):
                    continue

                # Load preprocessor
                preprocessor_path = os.path.join(series_dir, "preprocessor.joblib")
                if os.path.exists(preprocessor_path):
                    preprocessor = joblib.load(preprocessor_path)
                    self.app_state.scaler[series] = preprocessor["scaler"]
                    self.app_state.feature_cols[series] = preprocessor["feature_cols"]

                # Load models
                for model_file in os.listdir(series_dir):
                    if model_file == "preprocessor.joblib" or "_calibrator" in model_file:
                        continue

                    name = os.path.splitext(model_file)[0]
                    model_path = os.path.join(series_dir, model_file)

                    if model_file.endswith(".joblib"):
                        model = joblib.load(model_path)
                        self.app_state.models[series][name] = model
                        models_loaded = True
                    elif model_file.endswith(".pt"):
                        device = torch.device(
                            "cuda" if torch.cuda.is_available() else "cpu"
                        )
                        model = RacingPredictor(len(self.app_state.feature_cols[series]))
                        state_dict = torch.load(
                            model_path,
                            map_location=device,  # Load directly to target device
                            weights_only=False,
                        )
                        model.load_state_dict(state_dict)
                        model = model.to(device)  # Ensure model is on correct device
                        model.eval()  # Set to evaluation mode
                        calibrator_path = os.path.join(
                            series_dir, f"{name}_calibrator.joblib"
                        )
                        if os.path.exists(calibrator_path):
                            model.calibrator = joblib.load(calibrator_path)
                        self.app_state.models[series][name] = model
                        models_loaded = True

                # Update models_available for this series
                if self.app_state.models[series]:
                    self.app_state.system_status["models_available"][series] = list(
                        self.app_state.models[series].keys()
                    )

            if models_loaded:
                LOGGER.info(
                    f"Loaded models for series: {list(self.app_state.models.keys())}"
                )

            return models_loaded

        except Exception as e:
            LOGGER.error(f"Error loading models: {e}")
            return False

    async def train_models(self, trainable_df):
        """Train models on provided data"""
        LOGGER.info(f"Training models for {self.series} on {len(trainable_df)} records")
        from app.core.predictor import train_models

        (models, feature_cols, scaler) = train_models(trainable_df)

        # Store in series-specific slots
        self.app_state.models[self.series] = models
        self.app_state.feature_cols[self.series] = feature_cols
        self.app_state.scaler[self.series] = scaler

        self.app_state.system_status["last_training"] = datetime.now()
        self.app_state.system_status["last_trained_season"] = trainable_df["year"].max()

        # Update available models for this series
        self.app_state.system_status["models_available"][self.series] = list(
            models.keys()
        )

        self.app_state.system_status["data_health"][self.series] = {
            "historical_records": len(trainable_df),
            "current_records": 0,
        }
