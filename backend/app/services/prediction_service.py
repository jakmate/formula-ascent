from datetime import UTC, datetime

import pandas as pd
import torch

from app.config import CURRENT_YEAR, LOGGER
from app.core.state import AppState
from app.models.predictions import ModelResults, PredictionResponse
from app.services.data_service import DataService


class PredictionService:
    def __init__(
        self,
        app_state: AppState,
        series: str,
        data_service: DataService,
    ) -> None:
        self.app_state = app_state
        self.series = series
        self.data_service = data_service
        self.prediction_cache = {}

    async def get_prediction_for_model(self, model_name: str) -> ModelResults:
        """Return predictions for single series+model combo."""
        if not self.app_state.models.get(self.series, {}).get(model_name):
            raise ValueError(
                f"Model {model_name} not available for series {self.series}"
            )

        current_df, x_current = await self._get_cached_features()
        raw_probas = self._get_model_predictions(model_name, x_current)
        predictions = self._create_prediction_responses(current_df, raw_probas)
        return ModelResults(
            model_name=model_name,
            predictions=predictions,
            accuracy_metrics={"total_predictions": len(predictions)},
        )

    async def _get_cached_features(self):
        """Return (current_df, x_current) from cache or load."""
        cache_key = f"{self.series}_processed_features"
        if cache_key not in self.prediction_cache:
            current_df = await self.data_service.load_current_data(self.series)
            feature_cols = self.app_state.feature_cols[self.series]
            if not feature_cols:
                raise ValueError(f"No feature columns for {self.series}")
            x_current = current_df[feature_cols].fillna(0)
            self.prediction_cache[cache_key] = {
                "current_df": current_df,
                "x_current": x_current,
                "timestamp": datetime.now(UTC),
            }
        cached = self.prediction_cache[cache_key]
        return cached["current_df"], cached["x_current"]

    def _get_model_predictions(self, model_name: str, x_current):
        """Extract prediction logic for reusability."""
        if model_name not in self.app_state.models[self.series]:
            raise ValueError(f"Model {model_name} not found for series {self.series}")

        model = self.app_state.models[self.series][model_name]

        if "PyTorch" in model_name:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            x_current_scaled = self.app_state.scaler[self.series].transform(x_current)
            model.eval()
            with torch.no_grad():
                x_torch = torch.FloatTensor(x_current_scaled).to(device)
                logits = model(x_torch)
                raw_predictions = torch.sigmoid(logits).cpu().numpy().flatten()
        else:
            raw_predictions = model.predict_proba(x_current)[:, 1]

        if hasattr(model, "calibrator") and model.calibrator is not None:
            return model.calibrator.transform(raw_predictions)

        return raw_predictions

    def _create_prediction_responses(
        self,
        current_df,
        calibrated_probas,
    ) -> list[PredictionResponse]:
        """Create standardized prediction response objects."""
        prediction_values = calibrated_probas * 100.0  # Percentage for classification
        predictions = []

        for idx, (_, row) in enumerate(current_df.iterrows()):
            predictions.append(
                PredictionResponse(
                    driver=row["Driver"],
                    nationality=row.get("nationality")
                    if pd.notna(row.get("nationality"))
                    else None,
                    position=int(row["pos"]),
                    points=float(row["points"]),
                    avg_quali_pos=float(row.get("avg_quali_pos", 0)),
                    wins=int(row["wins"]),
                    win_rate=float(row["win_rate"]),
                    podiums=int(row["podiums"]),
                    dnf_rate=float(row["dnf_rate"]),
                    experience=int(row["experience"]),
                    dob=row.get("dob") if pd.notna(row.get("dob")) else None,
                    age=float(row["age"])
                    if row.get("age") is not None and not pd.isna(row.get("age"))
                    else None,
                    participation_rate=float(row["participation_rate"]),
                    teammate_h2h=float(row["teammate_h2h_rate"]),
                    team=str(row["team"]),
                    team_pos=int(row["team_pos"]),
                    team_points=float(row["team_points"]),
                    empirical_percentage=float(prediction_values[idx]),
                ),
            )

        predictions.sort(key=lambda x: x.empirical_percentage, reverse=True)
        return predictions

    async def update_predictions(self, features_df=None):
        """Generate predictions for current season."""
        try:
            if features_df is None:
                current_df = await self.data_service.load_current_data(self.series)
            else:
                current_df = features_df[
                    features_df["year"]
                    >= self.app_state.system_status.get(
                        "current_year",
                        CURRENT_YEAR - 1,
                    )
                ].copy()

            if current_df.empty:
                LOGGER.warning(f"No current data for {self.series} predictions")
                return

            x_current = current_df[self.app_state.feature_cols[self.series]].fillna(0)
            predictions = []
            for model_name in self.app_state.models[self.series]:
                try:
                    result = self._get_model_predictions(model_name, x_current)
                    predictions.append(
                        {
                            "model": model_name,
                            "series": self.series,
                            "predictions": result,
                            "timestamp": datetime.now(UTC),
                        },
                    )
                except Exception as e:
                    LOGGER.error(
                        f"Prediction failed for {model_name} in {self.series}: {e}",
                    )

            # Store predictions with series key
            if not hasattr(self.app_state, "current_predictions"):
                self.app_state.current_predictions = {}
            self.app_state.current_predictions[self.series] = predictions
            LOGGER.info(
                f"Generated {len(predictions)} prediction sets for {self.series}",
            )

        except Exception as e:
            LOGGER.error(f"Prediction update failed for {self.series}: {e}")

    def clear_prediction_cache(self):
        """Clear cached predictions and features."""
        self.prediction_cache.clear()
        LOGGER.info(f"Cleared prediction cache for {self.series}")
