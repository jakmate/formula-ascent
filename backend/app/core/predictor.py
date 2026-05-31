import logging
import os
import random

import numpy as np
import pandas as pd
import torch

from app.config import CURRENT_YEAR, SEED
from app.core.feature_creator import (
    calculate_participation_stats,
    calculate_qualifying_features,
    engineer_features,
)
from app.core.loader import load_data, load_qualifying_data, load_standings_data
from app.core.trainer import train_models
from app.core.utils import get_race_columns

log = logging.getLogger(__name__)

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.RandomState(SEED)
torch.manual_seed(SEED)
torch._dynamo.disable()
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_target_variable(feeder_df, parent_df, series):
    """Create target variable for parent series participation."""
    if feeder_df.empty or parent_df.empty:
        feeder_df["promoted"] = np.nan
        return feeder_df

    feeder_df["promoted"] = 0
    max_parent_year = parent_df["year"].max()

    # Get last feeder season per driver
    last_feeder_seasons = feeder_df.groupby("Driver")["year"].max()

    # Build participation lookup
    participation_lookup = {}
    for year, year_df in parent_df.groupby("year"):
        race_cols = get_race_columns(year_df)
        if not race_cols:
            continue

        threshold = 0 if year == CURRENT_YEAR else len(race_cols) * 0.4
        stats = calculate_participation_stats(year_df, race_cols)

        for stat in stats:
            key = (stat["Driver"], year)
            participation_lookup[key] = stat["participated_races"] > threshold

    # Target assignment
    years_to_check = [1, 2, 3] if series == "F1" else [1]

    def check_promotion(row):
        driver = row["Driver"]
        year = row["year"]
        last_year = last_feeder_seasons.get(driver)

        # Only process last feeder season
        if year != last_year:
            return 0

        # Can't observe future
        if year + 1 > max_parent_year:
            return np.nan

        # Check future years
        for offset in years_to_check:
            target_year = year + offset
            if target_year > max_parent_year:
                break
            if participation_lookup.get((driver, target_year), False):
                return 1
        return 0

    feeder_df["promoted"] = feeder_df.apply(check_promotion, axis=1)
    return feeder_df


def predict_drivers(models, df, feature_cols, scaler=None):
    """Make predictions for current year drivers."""
    current_df = df[df["year"] == CURRENT_YEAR].copy()
    if current_df.empty:
        current_df = df[df["year"] == df["year"].max()].copy()
    if current_df.empty:
        log.warning("No current data found for predictions")
        return pd.DataFrame()

    x_current = current_df[feature_cols].fillna(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = None

    for name, model in models.items():
        try:
            # Get raw probabilities based on model type
            if name == "PyTorch":
                if scaler is not None:
                    x_processed = scaler.transform(x_current)
                else:
                    x_processed = x_current
                model.eval()
                with torch.no_grad():
                    x_torch = torch.FloatTensor(x_processed).to(device)
                    logits = model(x_torch)
                    raw_probas = torch.sigmoid(logits).cpu().numpy().flatten()
            else:  # Traditional models
                x_processed = x_current
                raw_probas = model.predict_proba(x_processed)[:, 1]

            # Apply calibration if available
            if hasattr(model, "calibrator") and model.calibrator is not None:
                calibrated_probas = model.calibrator.transform(raw_probas)
            else:
                calibrated_probas = raw_probas

            empirical_pct = calibrated_probas * 100.0

            # Create results DataFrame
            results = pd.DataFrame(
                {
                    "Driver": current_df["Driver"],
                    "Nat.": current_df["nationality"],
                    "Nat_encoded": current_df["nationality_encoded"],
                    "Academy": current_df["academy"],
                    "Academy_encoded": current_df["academy_encoded"],
                    "Pos": current_df["pos"],
                    "Points": current_df["points"],
                    "Wins": current_df["wins"],
                    "Podiums": current_df["podiums"],
                    "Win %": current_df["win_rate"],
                    "DNF %": current_df["dnf_rate"],
                    "Participation %": current_df["participation_rate"],
                    "Exp": current_df["experience"],
                    "DoB": current_df["dob"],
                    "Age": current_df["age"],
                    "Teammate_h2h": current_df["teammate_h2h_rate"],
                    "Team": current_df["team"],
                    "Team Pos": current_df["team_pos"],
                    "Team Points": current_df["team_points"],
                    "Raw_Prob": raw_probas,
                    "Empirical_%": empirical_pct,
                },
            ).sort_values("Empirical_%", ascending=False)

            log.info("\n%s Predictions:", name)
            log.info("=" * 70)
            log.info(results.head(3).to_string(index=False, float_format="%.3f"))

        except Exception:
            log.exception("Error with %s model:", name)
            continue

    if results is not None:
        return results
    return pd.DataFrame()


def main():
    series = ["F3", "F2"]

    log.info("Loading %s qualifying data...", series[0])
    feeder_quali_data = load_qualifying_data(series[0])

    feeder_df = load_data(series[0])
    parent_df = load_standings_data(series[1], "drivers")

    log.info("Adding qualifying features...")
    feeder_df = calculate_qualifying_features(feeder_df, feeder_quali_data)

    log.info("Creating target variable based on %s participation...", series[1])
    feeder_df = create_target_variable(feeder_df, parent_df, series[1])

    log.info("Engineering features...")
    features_df = engineer_features(feeder_df)
    features_df["promoted"] = feeder_df["promoted"]
    del feeder_df, parent_df, feeder_quali_data

    log.info("Training all models...")
    models, feature_cols, scaler = train_models(features_df)

    log.info("Making predictions for %s %s drivers...", series[0], CURRENT_YEAR)
    predict_drivers(models, features_df, feature_cols, scaler)


if __name__ == "__main__":  # pragma: no cover
    main()
