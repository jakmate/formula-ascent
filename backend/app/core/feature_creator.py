import logging
import re

import numpy as np
import pandas as pd

from app.config import (
    EXPERIENCE_SEASON_PARTICIPATION_THRESHOLD,
    F2_WEIGHTED_SPRINT_WEIGHT,
    NOT_PARTICIPATED_CODES,
    RETIREMENT_CODES,
)
from app.core.utils import calculate_age, extract_position, get_race_columns

log = logging.getLogger(__name__)


def get_points_system(year):
    """Return points system parameters for a given year."""
    if year <= 2011:
        return {
            "feature_max": 12,  # 10 + 2 pole
            "sprint_max": 6,
            "feature_positions": [10, 8, 6, 5, 4, 3, 2, 1],
            "sprint_positions": [6, 5, 4, 3, 2, 1],
        }
    if year <= 2020:
        return {
            "feature_max": 31,  # 25 + 4 pole + 2 FL
            "sprint_max": 17,  # 15 + 2 FL
            "feature_positions": [25, 18, 15, 12, 10, 8, 6, 4, 2, 1],
            "sprint_positions": [15, 12, 10, 8, 6, 4, 2, 1],
        }
    if year == 2021:
        return {
            "race12_max": 17,  # 15 + 2 FL each
            "race3_max": 31,  # 25 + 4 pole + 2 FL
            "race12_positions": [15, 12, 10, 8, 6, 5, 4, 3, 2, 1],
            "race3_positions": [25, 18, 15, 12, 10, 8, 6, 4, 2, 1],
        }
    # 2022-2025
    return {
        "feature_max": 28,  # 25 + 2 pole + 1 FL
        "sprint_max": 11,  # 10 + 1 FL
        "feature_positions": [25, 18, 15, 12, 10, 8, 6, 4, 2, 1],
        "sprint_positions": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    }


def identify_race_type(col_name, year):
    """Identify if column is sprint or feature race."""
    col_lower = col_name.lower()

    try:
        if year == 2021:
            # Triple header year - need different logic
            return "race3" if "r3" in col_lower else "race12"
        if year >= 2022:
            return "sprint" if "sr" in col_lower else "feature"
        if year <= 2020:
            # R1/FR = Feature, R2/SR = Sprint for 2010-2020
            return "feature" if "r1" in col_lower or "fr" in col_lower else "sprint"
    except Exception:
        log.exception("Failed to identify race type: %s, %s", col_name, year)
        return None


def calculate_participation_stats(df, race_cols):
    """Calculate participation statistics for a dataframe."""
    stats = []

    for _, row in df.iterrows():
        # Vectorize race result checks
        race_results = [str(row[col]).strip() for col in race_cols]

        # Single pass through results
        participated_races = 0
        positions = []

        for result in race_results:
            if not result or result in NOT_PARTICIPATED_CODES:
                continue

            participated_races += 1

            if not any(x in result for x in RETIREMENT_CODES):
                pos = extract_position(result)
                if pos:
                    positions.append(pos)

        stats.append(
            {
                "Driver": row["Driver"],
                "year": row["year"],
                "participated_races": participated_races,
                "positions": positions,
            },
        )

    return stats


def calculate_teammate_performance(df):
    """Calculate performance metrics relative to teammates."""
    if "Team" not in df.columns:
        return df

    race_cols = get_race_columns(df)
    if not race_cols:
        return df

    # Extract positions for all drivers at once (vectorized)
    position_matrix = np.full((len(df), len(race_cols)), np.nan)
    for i, col in enumerate(race_cols):
        position_matrix[:, i] = (
            df[col]
            .apply(
                lambda x: extract_position(str(x).strip()) if pd.notna(x) else np.nan,
            )
            .to_numpy()
        )

    # Add positions to df temporarily
    df["_positions_matrix"] = list(position_matrix)

    team_performance = []

    # Group once
    grouped = df.groupby(["year", "Team"])

    for (year, team), team_df in grouped:
        if len(team_df) < 2:
            continue

        driver_indices = team_df.index.tolist()
        driver_names = team_df["Driver"].tolist()
        positions_list = team_df["_positions_matrix"].tolist()

        # Convert to numpy array for vectorized operations
        team_positions = np.array(positions_list)  # shape: (n_drivers, n_races)

        # Calculate pairwise comparisons
        n_drivers = len(driver_names)
        h2h_rates = np.full((n_drivers, n_drivers), np.nan)

        for i in range(n_drivers):
            for j in range(i + 1, n_drivers):
                pos_i = team_positions[i]
                pos_j = team_positions[j]

                # Valid races where both participated
                valid_mask = ~(np.isnan(pos_i) | np.isnan(pos_j))
                valid_count = valid_mask.sum()

                if valid_count > 0:
                    wins_i = (pos_i[valid_mask] < pos_j[valid_mask]).sum()
                    wins_j = (pos_j[valid_mask] < pos_i[valid_mask]).sum()
                    h2h_rates[i, j] = wins_i / valid_count
                    h2h_rates[j, i] = wins_j / valid_count

        # Calculate overall H2H rate for each driver
        for idx, (driver_idx, driver_name) in enumerate(
            zip(driver_indices, driver_names, strict=False),
        ):
            # Get all valid comparisons for this driver
            other_rates = h2h_rates[idx]
            valid_others = ~np.isnan(other_rates)

            if valid_others.any():
                # Weight by number of valid races against each teammate
                total_wins = 0
                total_races = 0

                for j, is_valid in enumerate(valid_others):
                    if is_valid and j != idx:
                        # Count valid races against teammate j
                        pos_self = team_positions[idx]
                        pos_other = team_positions[j]
                        valid_races = (
                            ~(np.isnan(pos_self) | np.isnan(pos_other))
                        ).sum()

                        total_races += valid_races
                        total_wins += other_rates[j] * valid_races

                h2h_rate = total_wins / total_races if total_races > 0 else 0.5
            else:
                h2h_rate = 0.5

            is_multi_team = df.loc[driver_idx].get("team_count", 1) > 1

            team_performance.append(
                {
                    "Driver": driver_name,
                    "year": year,
                    "Team": team,
                    "teammate_h2h_rate": h2h_rate,
                    "is_multi_team": is_multi_team,
                },
            )

    # Clean up
    df = df.drop("_positions_matrix", axis=1)

    # Convert to DataFrame and merge with original
    if team_performance:
        team_perf_df = pd.DataFrame(team_performance)
        df = df.merge(
            team_perf_df[["Driver", "year", "teammate_h2h_rate", "is_multi_team"]],
            on=["Driver", "year"],
            how="left",
        )

    # Fill defaults
    df["teammate_h2h_rate"] = df["teammate_h2h_rate"].fillna(0.5)
    df["is_multi_team"] = df["is_multi_team"].fillna(False)

    return df


def calculate_qualifying_features(df, qualifying_df):
    """Calculate qualifying statistics for each driver-year combination."""
    position_columns = ["Pos.", "Grid"]

    # Vectorized position extraction function
    def extract_position_from_row(row):
        for col in position_columns:
            if col not in row.index or pd.isna(row[col]):
                continue

            # Handle numeric values directly
            if isinstance(row[col], (int, float)):
                return int(row[col])

            # String processing
            str_value = str(row[col]).strip()
            if str_value not in NOT_PARTICIPATED_CODES:
                match = re.search(r"\b\d{1,2}\b", str_value)
                if match:
                    return int(match.group())
        return np.nan

    # Extract positions for all rows at once
    qualifying_df["_extracted_pos"] = qualifying_df.apply(
        extract_position_from_row,
        axis=1,
    )

    # Group and aggregate in one operation
    qualifying_stats = (
        qualifying_df.groupby(["Driver", "year"])["_extracted_pos"]
        .agg(
            [
                ("avg_quali_pos", lambda x: x.mean() if x.notna().any() else np.nan),
                ("std_quali_pos", lambda x: x.std() if x.notna().any() else np.nan),
            ],
        )
        .reset_index()
    )

    # Merge with main data
    if not qualifying_stats.empty:
        df = df.merge(qualifying_stats, on=["Driver", "year"], how="left")

    # Fill missing values
    df[["avg_quali_pos", "std_quali_pos"]] = df.get(
        ["avg_quali_pos", "std_quali_pos"],
        np.nan,
    )

    return df


def encode_nationality_and_academy(df, features_df, alpha=10):
    if "promoted" not in df.columns:
        raise ValueError("'promoted' not in df")
    global_mean = df["promoted"].mean()

    # Safe missing value handling
    train_nat = df["nationality"].fillna("NO_NATIONALITY")
    train_acad = df["academy"].fillna("NO_ACADEMY")
    feat_nat = features_df["nationality"].fillna("NO_NATIONALITY")
    feat_acad = features_df["academy"].fillna("NO_ACADEMY")

    if "promoted" in df.columns:
        # Nationality encoding
        nat_stats = df.groupby(train_nat)["promoted"].agg(["sum", "count"])
        nat_stats["smoothed"] = (nat_stats["sum"] + alpha * global_mean) / (
            nat_stats["count"] + alpha
        )
        features_df["nationality_encoded"] = feat_nat.map(nat_stats["smoothed"]).fillna(
            global_mean,
        )

        # Academy encoding
        acad_stats = df.groupby(train_acad)["promoted"].agg(["sum", "count"])
        acad_stats["smoothed"] = (acad_stats["sum"] + alpha * global_mean) / (
            acad_stats["count"] + alpha
        )
        features_df["academy_encoded"] = feat_acad.map(acad_stats["smoothed"]).fillna(
            global_mean,
        )

        features_df["has_academy"] = (feat_acad != "NO_ACADEMY").astype(int)
    else:
        features_df["nationality_encoded"] = global_mean
        features_df["academy_encoded"] = global_mean
        features_df["has_academy"] = (feat_acad != "NO_ACADEMY").astype(int)

    return features_df


def engineer_features(df):
    """Create features for ML models with race type separation."""
    if df.empty:
        return pd.DataFrame()

    df = calculate_teammate_performance(df)
    df = calculate_age(df)
    df = df.sort_values(by=["Driver", "year"])
    df["experience"] = df.groupby("Driver").cumcount()

    features_df = pd.DataFrame(
        {
            "year": df["year"],
            "Driver": df["Driver"],
            "series": df["series"],
            "dob": df["dob"],
            "nationality": df["nationality"],
            "pos": pd.to_numeric(df["Pos"], errors="coerce").fillna(-1).astype(int),
            "points": pd.to_numeric(df["Points"], errors="coerce").fillna(0),
            "experience": df["experience"],
            "age": df.get("age", np.nan),
            "team": df.get("Team"),
            "team_pos": df.get("team_pos", np.nan),
            "team_points": df.get("team_points", 0),
            "teammate_h2h_rate": df.get("teammate_h2h_rate", 0.5),
            "avg_quali_pos": df.get("avg_quali_pos", 0),
            "std_quali_pos": df.get("std_quali_pos", 0),
            "academy": df.get("academy", None),
        },
    )

    # Calculate race statistics
    race_stats = []
    cache_key_to_data = {}

    for _, row in df.iterrows():
        cache_key = (row["year"], row.get("series", "F3"))
        if cache_key not in cache_key_to_data:
            year_series_data = df[
                (df["year"] == row["year"])
                & (df.get("series", "F3") == row.get("series", "F3"))
            ]
            race_cols = get_race_columns(year_series_data)

            valid_race_cols = [
                col
                for col in race_cols
                if not year_series_data[col].astype(str).str.strip().eq("C").any()
            ]
            cache_key_to_data[cache_key] = (race_cols, valid_race_cols)

        race_cols, valid_race_cols = cache_key_to_data[cache_key]
        points_system = get_points_system(row["year"])

        stats = {
            "sprint_points": 0,
            "feature_points": 0,
            "sprint_races": 0,
            "feature_races": 0,
            "sprint_wins": 0,
            "feature_wins": 0,
            "sprint_podiums": 0,
            "feature_podiums": 0,
            "sprint_point_finishes": 0,
            "feature_point_finishes": 0,
            "dnfs": 0,
            "finish_positions": [],
        }

        for col in race_cols:
            if col not in row or pd.isna(row[col]):
                continue

            result = str(row[col]).strip()
            if not result or result in NOT_PARTICIPATED_CODES:
                continue

            race_type = identify_race_type(col, row["year"])

            if any(x in result for x in RETIREMENT_CODES) and result != "NC":
                stats["dnfs"] += 1

            # Determine race type category
            is_sprint = (row["year"] == 2021 and race_type == "race12") or (
                row["year"] != 2021 and race_type == "sprint"
            )

            # Count participation
            if is_sprint:
                stats["sprint_races"] += 1
            else:
                stats["feature_races"] += 1

            # Skip further processing for retirements
            if any(x in result for x in RETIREMENT_CODES):
                continue

            # Extract position
            pos = extract_position(result)
            if not pos:
                continue

            stats["finish_positions"].append(pos)

            # Get appropriate points system
            if row["year"] == 2021:
                positions = points_system[
                    "race12_positions" if is_sprint else "race3_positions"
                ]
            else:
                positions = points_system.get(f"{race_type}_positions", [])

            # Calculate points and achievements
            if pos <= len(positions):
                points = positions[pos - 1]
                if is_sprint:
                    stats["sprint_points"] += points
                    stats["sprint_point_finishes"] += 1
                else:
                    stats["feature_points"] += points
                    stats["feature_point_finishes"] += 1

            if pos == 1:
                stats["sprint_wins" if is_sprint else "feature_wins"] += 1
            if pos <= 3:
                stats["sprint_podiums" if is_sprint else "feature_podiums"] += 1

        stats["races_completed"] = stats["feature_races"] + stats["sprint_races"]
        stats["participation_rate"] = (
            stats["races_completed"] / len(valid_race_cols) if valid_race_cols else 0
        )

        race_stats.append(stats)

    # Add race statistics
    for stat_name in [
        "sprint_points",
        "feature_points",
        "sprint_races",
        "feature_races",
        "sprint_wins",
        "feature_wins",
        "sprint_podiums",
        "feature_podiums",
        "sprint_point_finishes",
        "feature_point_finishes",
        "dnfs",
        "races_completed",
        "participation_rate",
    ]:
        features_df[stat_name] = [stats[stat_name] for stats in race_stats]

    features_df = features_df[features_df["races_completed"] > 0]

    # Make late-season cameo entries do not inflate season-level experience.
    features_df["counts_for_experience"] = (
        features_df["participation_rate"] >= EXPERIENCE_SEASON_PARTICIPATION_THRESHOLD
    ).astype(int)
    experience_cumsum = features_df.groupby("Driver")["counts_for_experience"].cumsum()
    features_df["experience"] = (
        experience_cumsum - features_df["counts_for_experience"]
    ).astype(int)
    features_df = features_df.drop(columns=["counts_for_experience"])

    # Race-type specific rates
    features_df["sprint_races"] = features_df["sprint_races"].fillna(0)
    features_df["feature_races"] = features_df["feature_races"].fillna(0)

    # Win rates by race type
    features_df["sprint_win_rate"] = np.where(
        features_df["sprint_races"] > 0,
        features_df["sprint_wins"].fillna(0) / features_df["sprint_races"],
        0,
    )
    features_df["feature_win_rate"] = np.where(
        features_df["feature_races"] > 0,
        features_df["feature_wins"].fillna(0) / features_df["feature_races"],
        0,
    )
    weighted_numerator = features_df["feature_wins"].fillna(0) + (
        F2_WEIGHTED_SPRINT_WEIGHT * features_df["sprint_wins"].fillna(0)
    )
    weighted_denominator = features_df["feature_races"].fillna(0) + (
        F2_WEIGHTED_SPRINT_WEIGHT * features_df["sprint_races"].fillna(0)
    )
    features_df["weighted_win_rate"] = np.where(
        weighted_denominator > 0,
        weighted_numerator / weighted_denominator,
        0,
    )

    features_df["wins"] = features_df["feature_wins"] + features_df["sprint_wins"]
    features_df["podiums"] = (
        features_df["feature_podiums"] + features_df["sprint_podiums"]
    )

    # Overall rates
    features_df["win_rate"] = features_df["wins"] / features_df["races_completed"]
    features_df["dnf_rate"] = features_df["dnfs"] / features_df["races_completed"]

    # Championship position percentile
    features_df["champ_pos_pct"] = features_df.groupby("year")["pos"].rank(pct=True)

    # Encode nationalities and academies
    return encode_nationality_and_academy(df, features_df)
