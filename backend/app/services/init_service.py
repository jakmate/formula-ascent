import logging

from app.config import CURRENT_YEAR
from app.core.feature_creator import calculate_qualifying_features, engineer_features
from app.core.loader import load_data, load_qualifying_data, load_standings_data
from app.core.predictor import create_target_variable
from app.core.state import AppState
from app.services.data_service import DataService
from app.services.model_service import ModelService
from app.services.prediction_service import PredictionService

log = logging.getLogger(__name__)


class InitService:
    def __init__(
        self,
        app_state: AppState,
        data_service: DataService,
    ) -> None:
        self.app_state = app_state
        self.data_service = data_service

    async def initialize_system(self):
        """Initial data loading, training, and prediction generation."""
        for series in ["f3_to_f2", "f2_to_f1"]:
            try:
                log.info("Initializing system for %s...", series)
                feeder_series, parent_series = self.data_service._parse_series(series)

                feeder_df = load_data(feeder_series)
                parent_df = load_standings_data(parent_series, "drivers")
                feeder_quali_df = load_qualifying_data(feeder_series)
                feeder_df = calculate_qualifying_features(feeder_df, feeder_quali_df)
                feeder_df = create_target_variable(feeder_df, parent_df, parent_series)
                features_df = engineer_features(feeder_df)
                features_df["promoted"] = feeder_df["promoted"]

                self.data_service.data_cache[f"full_data_{series}"] = features_df

                trainable_df = features_df[features_df["year"] < CURRENT_YEAR]

                if not trainable_df.empty:
                    series_model_service = ModelService(self.app_state, series)
                    await series_model_service.train_models(trainable_df)
                    await series_model_service.save_models()
                else:
                    log.warning("No historical data available for training %s", series)

                prediction_service = PredictionService(
                    self.app_state, series, self.data_service
                )
                for model_name in self.app_state.models[series]:
                    await prediction_service.get_prediction_for_model(model_name)

            except Exception:
                log.exception("Failed to initialize %s", series)

        self.app_state.save_state()
