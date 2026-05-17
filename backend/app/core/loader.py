import json
import re

import pandas as pd

from app.config import DATA_DIR, LOGGER, PROFILES_DIR

FILE_PATTERNS = {
    "drivers": "{series}_{year}_drivers_standings.csv",
    "entries": "{series}_{year}_entries.csv",
    "teams": "{series}_{year}_teams_standings.csv",
    "qualifying": "{series}_{year}_qualifying_round_{round}.csv",
}


def get_file_pattern(file_type, series, year, round_num=None):
    """Get file pattern based on series type."""
    pattern = FILE_PATTERNS[file_type]
    if round_num:
        return pattern.format(series=series.lower(), year=year, round=round_num)
    return pattern.format(series=series.lower(), year=year)


def get_series_directories(series):
    """Get all directories for a series and their patterns."""
    series_path = DATA_DIR / series
    if not series_path.exists():
        LOGGER.warning(f"Series directory not found: {series_path}")
        return []
    return [p for p in series_path.iterdir() if p.is_dir() and p.name.isdigit()]


def load_all_entries_data(series):
    """Load all entries data for a series at once."""
    all_entries = []
    directories = get_series_directories(series)

    for year_dir in directories:
        try:
            year_int = int(year_dir.name)
            entries_file = year_dir / get_file_pattern("entries", series, year_dir.name)

            try:
                entries_df = pd.read_csv(entries_file)
            except FileNotFoundError as e:
                LOGGER.warning(f"Skipping entries for {year_dir.name} ({series}): {e}")
                continue

            entries_df["year"] = year_int
            entries_df["series"] = series
            all_entries.append(entries_df)

        except ValueError as e:
            LOGGER.error(f"Failed to process entries in {year_dir}: {e}")
            continue

    return pd.concat(all_entries, ignore_index=True) if all_entries else pd.DataFrame()


def load_year_data(year_dir, series, data_type):
    """Load data for a specific year and series."""
    try:
        year_int = int(year_dir.name)
        data_file = year_dir / get_file_pattern(data_type, series, year_dir.name)

        try:
            df = pd.read_csv(data_file)
        except FileNotFoundError as e:
            LOGGER.warning(
                f"Skipping {data_type} data for {year_dir.name} ({series}): {e}",
            )
            return None

        # Case of Konstantin Tereshchenko
        if data_type == "drivers" and "Pos" in df.columns:
            df = df.dropna(subset=["Pos"]).copy()

        df.loc[:, "year"] = year_int
        df.loc[:, "series"] = series
        return df

    except ValueError as e:
        LOGGER.error(f"Error processing {data_type} data in {year_dir}: {e}")
        return None


def load_standings_data(series, data_type):
    """Load standings data (drivers or teams) for a racing series."""
    all_data = []
    directories = get_series_directories(series)

    for year_dir in directories:
        df = load_year_data(year_dir, series, data_type)
        if df is not None:
            all_data.append(df)

    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


def load_qualifying_data(series):
    """Load qualifying data for a racing series across years."""
    quali_data = []
    directories = get_series_directories(series)

    for year_dir in directories:
        try:
            year_int = int(year_dir.name)
            quali_dir = year_dir / "qualifying"
            if not quali_dir.exists():
                continue

            pattern = f"{series.lower()}_{year_dir.name}_qualifying_round_*.csv"
            for quali_file in quali_dir.glob(pattern):
                try:
                    df = pd.read_csv(quali_file)
                    df["year"] = year_int
                    df["series"] = series
                    df["round"] = quali_file.stem.split("_")[-1]  # e.g., "5"
                    quali_data.append(df)
                except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                    LOGGER.warning(f"Error loading qualifying file {quali_file}: {e}")
                    continue

        except ValueError as e:
            LOGGER.error(f"Error processing qualifying data for {year_dir}: {e}")
            continue

    return pd.concat(quali_data, ignore_index=True) if quali_data else pd.DataFrame()


def get_driver_filename(driver_name):
    safe_name = re.sub(r"[^\w\s-]", "", driver_name)
    safe_name = re.sub(r"[-\s]+", "_", safe_name)
    return f"{safe_name.lower()}.json"


def load_driver_data(df):
    """Add features from cached JSON profiles."""
    default_profile = {"dob": None, "nationality": None}

    # Load cached profiles
    profiles = {}
    if PROFILES_DIR.exists():
        for driver in df["Driver"].unique():
            profile_file = PROFILES_DIR / get_driver_filename(driver)
            try:
                with profile_file.open(encoding="utf-8") as f:
                    profile_data = json.load(f)
                    profiles[driver] = (
                        profile_data
                        if profile_data.get("scraped", True)
                        else default_profile
                    )
            except Exception:
                profiles[driver] = default_profile

    # Map profiles to dataframe
    df["dob"] = df["Driver"].map(lambda d: profiles.get(d, default_profile)["dob"])
    df["nationality"] = df["Driver"].map(
        lambda d: profiles.get(d, default_profile).get("nationality", "Unknown"),
    )
    return df


def merge_entries(driver_df, entries_df):
    """Merge all entries data with driver standings at once."""
    if entries_df.empty:
        return driver_df

    # Add team count and round count in one go
    entries_df["team_count"] = entries_df.groupby(["Driver", "year", "series"])[
        "Team"
    ].transform("count")
    entries_df["round_count"] = entries_df["Rounds"].apply(parse_round_count)

    # For multi-team drivers: pick team with max round_count
    primary_idx = entries_df.groupby(["Driver", "year", "series"])[
        "round_count"
    ].idxmax()
    primary_teams = entries_df.loc[primary_idx]

    return driver_df.merge(
        primary_teams[["Driver", "Team", "team_count", "year", "series"]],
        on=["Driver", "year", "series"],
        how="left",
    )


def parse_round_count(rounds_str):
    """Parse rounds string to count."""
    if rounds_str is None or rounds_str == "All":
        return float("inf")

    rounds_str = str(rounds_str).replace("–", "-")
    count = 0
    for part in rounds_str.split(","):
        p = part.strip()
        if "-" in p:
            start, end = map(int, p.split("-"))
            count += end - start + 1
        else:
            count += 1
    return count


def merge_team_data(driver_df, team_df):
    team_df = (
        team_df.groupby(["year", "Team"], as_index=False)
        .agg({"Pos": "first", "Points": "first"})
        .rename(columns={"Pos": "team_pos", "Points": "team_points"})
    )

    return driver_df.merge(team_df, on=["Team", "year"], how="left")


def load_academies_data():
    """Load all driver academy/development program data."""
    academies_dir = DATA_DIR / "academies"

    if not academies_dir.exists():
        LOGGER.warning(f"Academies directory not found: {academies_dir}")
        return pd.DataFrame()

    all_academy_data = []

    for csv_file in academies_dir.glob("*_drivers.csv"):
        try:
            df = pd.read_csv(csv_file)

            # Extract academy name from filename
            academy_name = csv_file.stem.replace("_drivers", "").replace("_", " ")
            df["academy"] = academy_name

            all_academy_data.append(df)

        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            LOGGER.warning(f"Error loading academy file {csv_file}: {e}")
            continue

    if not all_academy_data:
        return pd.DataFrame()

    academy_df = pd.concat(all_academy_data, ignore_index=True)

    # Convert Year to int, handling any non-numeric values
    academy_df["Year"] = pd.to_numeric(academy_df["Year"], errors="coerce")
    academy_df = academy_df.dropna(subset=["Year"])
    academy_df["Year"] = academy_df["Year"].astype(int)

    return academy_df


def merge_academy_data(driver_df, academy_df):
    """Merge academy data with driver standings."""
    if academy_df.empty:
        driver_df["academy"] = None
        return driver_df

    # Rename Year column to match driver_df's 'year' column
    academy_merge = academy_df.rename(columns={"Year": "year"})

    # Concatenate multiple academies for same driver-year
    academy_merge = (
        academy_merge.groupby(["Driver", "year"])["academy"]
        .apply(lambda x: " | ".join(x.dropna().unique()))
        .reset_index()
    )

    # Replace empty strings with None
    academy_merge["academy"] = academy_merge["academy"].replace("", None)

    # Left join on Driver and year
    return driver_df.merge(
        academy_merge[["Driver", "year", "academy"]],
        on=["Driver", "year"],
        how="left",
    )


def load_data(series):
    driver_df = load_standings_data(series, "drivers")
    team_df = load_standings_data(series, "teams")
    entries_df = load_all_entries_data(series)
    academy_df = load_academies_data()
    df = merge_entries(driver_df, entries_df)
    df = merge_team_data(df, team_df)
    df = load_driver_data(df)
    return merge_academy_data(df, academy_df)


if __name__ == "__main__":  # pragma: no cover
    load_data("F3")
